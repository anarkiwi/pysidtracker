//! The call machine: the play routine of a tune on jennings' native core over the tune's RAM,
//! with every call into a modelled entry run as that entry's program.
//!
//! A frame that needs anything Python answers (a device register, the CPU port, a program not
//! yet built) is undone from a log of the stores it made and reported as a [`Bail`], so Python
//! can play it and the machine continue from the frame after.

use std::cell::Cell;

use jennings_core::{run_until, Access, AddrSet, Bus, Cpu, FlatBus, FlatMemory, Hooks, Until};

use crate::vm::{
    chip_input, chip_output, port_read, Bail, Env, Flow, Native, Vm, REGISTERS, SID_BASE,
};

pub const SID_REG_COUNT: usize = 25;
const PW_HI: [usize; 3] = [0x03, 0x0A, 0x11];
const BANKED_PAGES: [u8; 5] = [0xA, 0xB, 0xD, 0xE, 0xF];
const FREE: u16 = 0;

/// Notes a device access, which Python alone answers, and lets the access complete as RAM.
struct Gate<'a> {
    hit: &'a Cell<Option<u16>>,
    port: &'a Cell<bool>,
}

impl Hooks for Gate<'_> {
    fn read_hit(&mut self, mem: &FlatMemory, addr: u16, _access: Access) -> Option<u8> {
        (addr == 1).then(|| port_read(mem.ram(0), mem.ram(1)))
    }

    fn write_hit(&mut self, _mem: &FlatMemory, _addr: u16, v: u8, _access: Access) -> u8 {
        self.port.set(true);
        v
    }

    fn io_read(&mut self, mem: &FlatMemory, addr: u16) -> u8 {
        if chip_output(addr) {
            self.hit.set(Some(addr));
        }
        mem.ram(addr)
    }

    fn io_write(&mut self, mem: &FlatMemory, addr: u16, v: u8) {
        if chip_input(addr) {
            self.hit.set(Some(addr));
        }
        mem.set_ram(addr, v)
    }
}

/// The CPU's bus, logging the old value of every byte it stores.
struct Logged<'a, 'b> {
    inner: FlatBus<'a, Gate<'b>>,
    log: &'a mut Vec<(u16, u8)>,
}

impl Logged<'_, '_> {
    #[inline(always)]
    fn note(&mut self, addr: u16) {
        self.log.push((addr, self.inner.mem.ram(addr)));
    }
}

impl Bus for Logged<'_, '_> {
    #[inline(always)]
    fn read(&mut self, addr: u16) -> u8 {
        self.inner.read(addr)
    }

    #[inline(always)]
    fn write(&mut self, addr: u16, v: u8) {
        self.note(addr);
        self.inner.write(addr, v)
    }

    #[inline(always)]
    fn fetch(&mut self, addr: u16) -> u8 {
        self.inner.fetch(addr)
    }

    #[inline(always)]
    fn load(&mut self, addr: u16) -> u8 {
        self.inner.load(addr)
    }

    #[inline(always)]
    fn store(&mut self, addr: u16, v: u8) {
        self.note(addr);
        self.inner.store(addr, v)
    }

    #[inline(always)]
    fn stack_read(&mut self, addr: u16) -> u8 {
        self.inner.stack_read(addr)
    }

    #[inline(always)]
    fn stack_write(&mut self, addr: u16, v: u8) {
        self.note(addr);
        self.inner.stack_write(addr, v)
    }

    #[inline(always)]
    fn sync(&mut self, cpu: &Cpu) {
        self.inner.sync(cpu)
    }
}

/// Runs the CPU on `mem` until a condition of `until` holds, logging its stores; whether it
/// wrote the CPU port. A device access ends the run as a [`Bail`].
fn drive(
    cpu: &mut Cpu,
    mem: &FlatMemory,
    log: &mut Vec<(u16, u8)>,
    until: &mut Until,
) -> Result<bool, Bail> {
    let hit = Cell::new(None);
    let port = Cell::new(false);
    let mut gate = Gate {
        hit: &hit,
        port: &port,
    };
    let mut bus = Logged {
        inner: mem.bus(&mut gate),
        log,
    };
    run_until(cpu, &mut bus, until, || hit.get().is_some() || port.get());
    match hit.get() {
        Some(addr) => Err(Bail::Device(addr)),
        None => Ok(port.get()),
    }
}

/// The port setting the CPU port registers in RAM select.
fn port_bits(mem: &FlatMemory) -> usize {
    ((mem.ram(1) | !mem.ram(0)) & 7) as usize
}

/// The registers a `native` block overwrote, to give back when the spec ends.
#[derive(Clone, Copy)]
struct Saved {
    pc: u16,
    sp: u8,
    a: u8,
    x: u8,
    y: u8,
    p: u8,
}

/// The tune's own code for the `native` statements of one call of a program.
struct CpuNative<'a> {
    cpu: &'a mut Cpu,
    mem: &'a FlatMemory,
    bits: usize,
    max_cycles: u64,
    base: Option<u8>,
    saved: Option<Saved>,
}

impl Native for CpuNative<'_> {
    fn run(
        &mut self,
        log: &mut Vec<(u16, u8)>,
        addr: u16,
        regs: [Option<u8>; 3],
        resume: u16,
    ) -> Result<bool, Bail> {
        let cpu = &mut *self.cpu;
        if self.saved.is_none() {
            self.saved = Some(Saved {
                pc: cpu.pc,
                sp: cpu.sp,
                a: cpu.a,
                x: cpu.x,
                y: cpu.y,
                p: cpu.p(),
            });
            if self.base.is_none() {
                for byte in [0x00, 0x01] {
                    let at = 0x0100 | cpu.sp as u16;
                    log.push((at, self.mem.ram(at)));
                    self.mem.set_ram(at, byte);
                    cpu.sp = cpu.sp.wrapping_sub(1);
                }
                self.base = Some(cpu.sp);
            }
        }
        let base = self.base.expect("a base");
        for (slot, value) in [&mut cpu.a, &mut cpu.x, &mut cpu.y].into_iter().zip(regs) {
            if let Some(v) = value {
                *slot = v;
            }
        }
        cpu.pc = addr;
        let start = cpu.cycles;
        let stop: AddrSet = [resume].into_iter().collect();
        while cpu.pc != resume {
            if cpu.sp > base {
                return Ok(true);
            }
            if cpu.cycles - start >= self.max_cycles {
                return Err(Bail::Spec);
            }
            let mut until = Until {
                sp_above: Some(base as i32),
                max_cycles: Some(self.max_cycles - (cpu.cycles - start) - 1),
                stop_pcs: Some(&stop),
                stop_on_jam: true,
                ..Until::default()
            };
            if drive(cpu, self.mem, log, &mut until)? && port_bits(self.mem) != self.bits {
                return Err(Bail::Port);
            }
        }
        Ok(false)
    }
}

/// The memory map and modelled entries under one setting of the CPU port's low bits.
struct Bank {
    io: bool,
    pages: Vec<(u8, Vec<u8>)>,
    overlay: u16,
    entries: AddrSet,
    slot: Vec<u16>,
}

pub struct Machine {
    mem: FlatMemory,
    pub cpu: Cpu,
    vms: Vec<Vm>,
    banks: Vec<Option<Bank>>,
    bits: usize,
    mapped: bool,
    play: u16,
    entry: u16,
    max_cycles: u64,
    log: Vec<(u16, u8)>,
    pub sid: Vec<(u8, u8)>,
    saved: Vec<i64>,
}

impl Machine {
    /// A machine over the 64 KiB at `ram`, the play routine at `play` and `entry` the modelled
    /// entry it is (or reaches). Banks and programs are added before it runs.
    ///
    /// # Safety
    /// `ram` must stay valid for reads and writes of 64 KiB for the life of the machine and be
    /// accessed by nothing else while it runs.
    pub unsafe fn new(ram: *mut u8, play: u16, entry: u16, max_cycles: u64) -> Machine {
        let mem = FlatMemory::with_shared_ram(ram);
        mem.watch_read(1, true);
        mem.watch_write(0, true);
        mem.watch_write(1, true);
        Machine {
            mem,
            cpu: Cpu::new(),
            vms: Vec::new(),
            banks: (0..8).map(|_| None).collect(),
            bits: 0,
            mapped: false,
            play,
            entry,
            max_cycles,
            log: Vec::new(),
            sid: Vec::new(),
            saved: Vec::new(),
        }
    }

    /// Adds the memory map of port setting `bits`: whether I/O is mapped, the pages read from
    /// ROM images and the modelled entries.
    pub fn add_bank(
        &mut self,
        bits: usize,
        io: bool,
        pages: Vec<(u8, Vec<u8>)>,
        entries: &[u16],
    ) -> Result<(), String> {
        let bad = pages
            .iter()
            .any(|(p, image)| !BANKED_PAGES.contains(p) || image.len() != 0x1000);
        if bits > 7 || bad {
            return Err("a bank is a port setting below 8 with its ROM pages".into());
        }
        let overlay = pages.iter().fold(0u16, |mask, (p, _)| mask | 1 << p);
        self.mapped &= bits != self.bits;
        let slot = match self.banks[bits].take() {
            Some(old) => old.slot,
            None => vec![FREE; 0x10000],
        };
        self.banks[bits] = Some(Bank {
            io,
            pages,
            overlay,
            entries: entries.iter().copied().collect(),
            slot,
        });
        Ok(())
    }

    /// Whether the memory map of port setting `bits` has been added.
    pub fn has_bank(&self, bits: usize) -> bool {
        self.banks.get(bits).is_some_and(|b| b.is_some())
    }

    /// Makes `bits` the port setting the memory is mapped under.
    pub fn select(&mut self, bits: usize) -> Result<(), String> {
        if self.mapped && self.bits == bits {
            return Ok(());
        }
        let bank = self
            .banks
            .get(bits)
            .and_then(|b| b.as_ref())
            .ok_or("no such bank")?;
        for page in BANKED_PAGES {
            match bank.pages.iter().find(|(p, _)| *p == page) {
                Some((_, image)) => self.mem.set_page(page, Some(image)),
                None => self.mem.set_page(page, None),
            }
        }
        self.mem.set_io(bank.io);
        self.bits = bits;
        self.mapped = true;
        Ok(())
    }

    /// Adds the program of the modelled entry `entry` under port setting `bits`.
    pub fn add(&mut self, bits: usize, entry: u16, vm: Vm) -> Result<(), String> {
        let bank = self
            .banks
            .get_mut(bits)
            .and_then(|b| b.as_mut())
            .ok_or("no such bank")?;
        match bank.slot[entry as usize] {
            FREE => {
                self.vms.push(vm);
                bank.slot[entry as usize] = self.vms.len() as u16;
            }
            n => self.vms[n as usize - 1] = vm,
        }
        Ok(())
    }

    pub fn vm(&mut self, bits: usize, entry: u16) -> Option<&mut Vm> {
        let n = self.banks.get(bits)?.as_ref()?.slot[entry as usize];
        (n != FREE).then(|| &mut self.vms[n as usize - 1])
    }

    /// The port setting the memory is mapped under.
    pub fn bits(&self) -> usize {
        self.bits
    }

    fn bank(&self) -> &Bank {
        self.banks[self.bits].as_ref().expect("a selected bank")
    }

    /// Plays up to `frames` frames, appending each one's SID registers (25 bytes) to `out`;
    /// the frames played and what stopped the rest, the machine as it was before that frame.
    pub fn run(&mut self, frames: usize, out: &mut [u8]) -> (usize, Option<Bail>) {
        for done in 0..frames {
            if let Err(why) = self.frame() {
                return (done, Some(why));
            }
            snapshot(
                &self.mem,
                &mut out[done * SID_REG_COUNT..(done + 1) * SID_REG_COUNT],
            );
        }
        (frames, None)
    }

    fn frame(&mut self) -> Result<(), Bail> {
        self.log.clear();
        self.sid.clear();
        let cpu = self.cpu.clone();
        let bits = self.bits;
        self.saved.clear();
        for vm in &self.vms {
            self.saved.extend_from_slice(vm.controls());
        }
        let played = if self.play == self.entry {
            self.spec(self.entry, false).map(|_| ())
        } else {
            self.call()
        };
        if played.is_err() {
            for &(addr, old) in self.log.iter().rev() {
                self.mem.set_ram(addr, old);
            }
            self.cpu = cpu;
            if self.bits != bits {
                self.select(bits).expect("the bank it began in");
            }
            let mut at = 0;
            for vm in &mut self.vms {
                let n = vm.controls().len();
                vm.vars[..n].copy_from_slice(&self.saved[at..at + n]);
                at += n;
            }
        }
        played
    }

    /// Runs the program of `entry`; whether native code returned from the play call.
    fn spec(&mut self, entry: u16, in_call: bool) -> Result<bool, Bail> {
        let bank = self.bank();
        let n = bank.slot[entry as usize] as usize;
        if n == 0 {
            return Err(Bail::Need(entry));
        }
        let (io, overlay, bits) = (bank.io, bank.overlay, self.bits);
        let cpu = &self.cpu;
        let regs: [i64; REGISTERS] = [
            cpu.a as i64,
            cpu.x as i64,
            cpu.y as i64,
            cpu.c as i64,
            cpu.z as i64,
            cpu.i as i64,
            cpu.d as i64,
            cpu.v as i64,
            cpu.n as i64,
        ];
        let mut machine = CpuNative {
            base: in_call.then_some(cpu.sp),
            cpu: &mut self.cpu,
            mem: &self.mem,
            bits,
            max_cycles: self.max_cycles,
            saved: None,
        };
        let mut env = Env {
            mem: &self.mem,
            io,
            overlay,
            log: &mut self.log,
            sid: &mut self.sid,
            native: &mut machine,
        };
        let flow = self.vms[n - 1].exec(&mut env, &regs)?;
        if let Some(saved) = machine.saved {
            if flow == Flow::Done || !in_call {
                let cpu = &mut *machine.cpu;
                (cpu.pc, cpu.sp, cpu.a, cpu.x, cpu.y) =
                    (saved.pc, saved.sp, saved.a, saved.x, saved.y);
                cpu.set_p(saved.p);
            }
        }
        Ok(flow == Flow::Escape)
    }

    fn call(&mut self) -> Result<(), Bail> {
        let top = self.cpu.sp;
        for byte in [0x00, 0x01] {
            let at = 0x0100 | self.cpu.sp as u16;
            self.log.push((at, self.mem.ram(at)));
            self.mem.set_ram(at, byte);
            self.cpu.sp = self.cpu.sp.wrapping_sub(1);
        }
        self.cpu.pc = self.play;
        let start = self.cpu.cycles;
        while self.cpu.sp < top && self.cpu.cycles - start < self.max_cycles {
            if !self.bank().entries.contains(self.cpu.pc) {
                self.cpu_until(top, self.max_cycles - (self.cpu.cycles - start) - 1)?;
                continue;
            }
            if self.spec(self.cpu.pc, true)? {
                continue;
            }
            let sp = self.cpu.sp.wrapping_add(2);
            self.cpu.sp = sp;
            let lo = self.mem.ram(0x00FF + sp as u16) as u16;
            let hi = self.mem.ram(0x0100 + sp as u16) as u16;
            self.cpu.pc = (lo | hi << 8).wrapping_add(1);
        }
        Ok(())
    }

    fn cpu_until(&mut self, top: u8, cycles: u64) -> Result<(), Bail> {
        let bank = self.banks[self.bits].as_ref().expect("a selected bank");
        let mut until = Until {
            sp_above: Some(top as i32 - 1),
            max_cycles: Some(cycles),
            stop_pcs: Some(&bank.entries),
            stop_on_jam: true,
            ..Until::default()
        };
        if drive(&mut self.cpu, &self.mem, &mut self.log, &mut until)? {
            let bits = port_bits(&self.mem);
            if bits != self.bits {
                if !self.has_bank(bits) {
                    return Err(Bail::Bank(bits as u8));
                }
                self.select(bits).expect("a bank that was added");
            }
        }
        Ok(())
    }
}

/// The SID registers in RAM, the pulse-width highs to their nibble.
fn snapshot(mem: &FlatMemory, out: &mut [u8]) {
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = mem.ram((SID_BASE as u16) + i as u16);
    }
    for r in PW_HI {
        out[r] &= 0x0F;
    }
}
