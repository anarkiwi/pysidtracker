//! The interpreter of one lowered playspec program: a flat stream of `(op, arg)` instructions
//! over a value stack, executed on the 64 KiB memory of a [`FlatMemory`].
//!
//! Every value is an `i64` with the semantics of the Python integers the reference executor
//! uses; an operation Python would reject (a shift by a negative count, an address outside the
//! 64 KiB, a byte store of a non-byte) is a [`Bail`], and the caller runs that frame in Python.
//!
//! [`Vm::new`] checks a program once: operands are in range, every statement leaves the stack
//! empty and every jump lands where it is, so [`Vm::exec`] runs without bounds checks.

use std::cell::Cell;

use jennings_core::{FlatMemory, NoHooks};

use crate::objects::Step;

macro_rules! ops {
    ($($name:ident),* $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(u16)]
        pub enum Op { $($name),* }
        const ALL: &[Op] = &[$(Op::$name),*];
        /// The opcode names in numbering order, the vocabulary of the Python encoder.
        pub const NAMES: &[&str] = &[$(stringify!($name)),*];
        impl Op {
            pub fn from_u16(v: u16) -> Option<Op> {
                ALL.get(v as usize).copied()
            }
        }
    };
}

// Expressions push a value. `*K` forms take the right operand from the argument; `AddK`,
// `SubK` and `ShlK` pack the mask above the operand's low 32 bits. Statements leave the stack
// empty. Jumps take their target from the low 24 bits of the argument, the `*K` compare-jumps
// their right operand from the bits above; `J<cmp>` jumps when the left operand `<cmp>` the right.
// `Native` takes the 6502 code's address, the values of the registers `a`, `x`, `y` its argument
// mask names (in that order) and the address to resume at, and runs the tune's own code.
// `Obj` pops the operands of the object reaction its argument names (index in the low 24 bits,
// operand count above) and steps it.
// `JzKeep` and `JnzKeep` jump keeping the top value if it is zero (non-zero), else pop it: the
// short-circuit of `and`/`or`, with `Jz`/`Jmp` around the arms of a conditional expression.
ops!(
    Konst, Var, Rd, RdK, Add, Sub, Mul, And, Or, Xor, Shl, Shr, Eq, Ne, Ult, Ule, Slt, Sle, Word,
    AddK, SubK, ShlK, AndK, OrK, XorK, ShrK, EqK, NeK, UltK, UleK, Not, Neg, Lo, Hi, Lnot, Sext,
    Note, Wr, WrK, Set, SetK, Jmp, Jz, Jnz, JzKeep, JnzKeep, JEq, JNe, JLt, JLe, JGe, JGt, JEqK,
    JNeK, JLtK, JLeK, JGeK, JGtK, Call, Ret, Halt, Assert, Fail, Native, Obj,
);

pub const SID_BASE: i64 = 0xD400;
pub const SID_REGS: i64 = 25;
pub const REGISTERS: usize = 9;
const TARGET: i64 = 0xFF_FFFF;
const SETK_VAR: i64 = 0xF_FFFF;

/// Why a frame cannot be finished natively.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bail {
    /// An access Python answers with a device (I/O registers).
    Device(u16),
    /// The spec stored to the CPU port.
    Port,
    /// A value or address outside what Python's memory accepts, or a negative shift.
    Range,
    /// A failed assertion or an exhausted loop bound: Python raises it.
    Spec,
    /// The play call reached the modelled entry with this address, which has no program yet.
    Need(u16),
    /// The CPU port selected a memory map that has not been added.
    Bank(u8),
}

impl Bail {
    /// A number for the Python side to report: the reason a frame was not played natively.
    pub fn code(self) -> u16 {
        match self {
            Bail::Device(_) => 1,
            Bail::Port => 2,
            Bail::Range => 3,
            Bail::Spec => 4,
            Bail::Need(_) => 5,
            Bail::Bank(_) => 6,
        }
    }
}

/// How a program ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    /// It ran to its end (or a `finish`).
    Done,
    /// Native code returned from the play call itself: the CPU has finished the call.
    Escape,
}

/// The tune's own 6502 code, run for a `native` statement.
pub trait Native {
    /// Runs the code at `addr` with the registers `a`, `x`, `y` given, until it reaches `resume`;
    /// whether it returned from the play call first.
    fn run(
        &mut self,
        log: &mut Vec<(u16, u8)>,
        addr: u16,
        regs: [Option<u8>; 3],
        resume: u16,
    ) -> Result<bool, Bail>;
}

#[derive(Clone, Copy, Debug)]
pub struct Ins {
    pub op: Op,
    pub arg: i64,
}

/// A program: its code, note tables and variables. The variables are the persistent controls
/// followed by the locals, which start every call at zero.
pub struct Vm {
    code: Vec<Ins>,
    tables: Vec<Vec<i64>>,
    objects: Vec<Step>,
    inputs: Vec<(usize, usize)>,
    pub vars: Vec<i64>,
    controls: usize,
    stack: Vec<i64>,
    calls: Vec<usize>,
}

/// Where the stores of a build go: `slot` maps an address to `entries` (target or none, mask,
/// forced bits), and a store into the SID's registers then moves to the `ghost` copy.
#[derive(Default)]
pub struct Routes {
    pub ghost: Option<u16>,
    slot: Vec<u32>,
    entries: Vec<(Option<u16>, u8, u8)>,
}

impl Routes {
    pub fn new(ghost: Option<u16>, routes: &[(u16, Option<u16>, u8, u8)]) -> Routes {
        let mut slot = vec![0; if routes.is_empty() { 0 } else { 0x10000 }];
        let mut entries = Vec::new();
        for &(from, to, keep, force) in routes {
            entries.push((to, keep, force));
            slot[from as usize] = entries.len() as u32;
        }
        Routes {
            ghost,
            slot,
            entries,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.ghost.is_none() && self.entries.is_empty()
    }

    /// Where a store of `byte` to `addr` lands, if anywhere.
    fn apply(&self, mut addr: u16, mut byte: u8) -> Result<Option<(u16, u8)>, Bail> {
        if let Some(&n) = self.slot.get(addr as usize).filter(|&&n| n != 0) {
            let (to, keep, force) = self.entries[n as usize - 1];
            let Some(to) = to else { return Ok(None) };
            (addr, byte) = (to, byte & keep | force);
        }
        if let Some(ghost) = self.ghost {
            let offset = addr.wrapping_sub(SID_BASE as u16);
            if (offset as i64) < SID_REGS {
                addr = ghost.checked_add(offset).ok_or(Bail::Range)?;
            }
        }
        Ok(Some((addr, byte)))
    }
}

/// Where a program reads and stores.
pub struct Env<'a> {
    pub routes: &'a Routes,
    pub mem: &'a FlatMemory,
    pub io: bool,
    pub overlay: u16,
    pub log: &'a mut Vec<(u16, u8)>,
    pub sid: &'a mut Vec<(u8, u8)>,
    pub native: &'a mut dyn Native,
}

/// The 6510 port `$01` as read, given its direction and data registers.
pub fn port_read(ddr: u8, data: u8) -> u8 {
    let pins = (0x3F & !ddr) | (data & ddr);
    let mut value = (data | !ddr) & (pins | 0x17);
    if ddr & 0x20 == 0 {
        value &= !0x20;
    }
    value & !(!ddr & 0xC0)
}

#[cold]
fn special(addr: u16) -> bool {
    matches!(addr, 0xD011 | 0xD012)
        || (0xD400..0xD800).contains(&addr) && matches!(addr & 0x1F, 0x1B | 0x1C)
}

/// Whether a CPU write to the I/O page `addr` feeds the chip that answers `$D41B`/`$D41C`.
#[cold]
pub fn chip_input(addr: u16) -> bool {
    (0xD400..0xD800).contains(&addr) && (0x07..0x15).contains(&(addr & 0x1F))
}

/// Whether a CPU read of the I/O page `addr` is answered by a device.
#[cold]
pub fn chip_output(addr: u16) -> bool {
    special(addr)
}

/// `(pops, pushes)` of an instruction and whether it ends a statement (leaves the stack empty).
fn effect(ins: &Ins) -> (usize, usize, bool) {
    let op = ins.op;
    use Op::*;
    match op {
        Konst | Var | RdK => (0, 1, false),
        Rd | AddK | SubK | ShlK | AndK | OrK | XorK | ShrK | EqK | NeK | UltK | UleK | Not
        | Neg | Lo | Hi | Lnot | Sext | Note => (1, 1, false),
        Add | Sub | Mul | And | Or | Xor | Shl | Shr | Eq | Ne | Ult | Ule | Slt | Sle | Word => {
            (2, 1, false)
        }
        JEq | JNe | JLt | JLe | JGe | JGt => (2, 0, false),
        Jz | Jnz | JEqK | JNeK | JLtK | JLeK | JGeK | JGtK => (1, 0, false),
        JzKeep | JnzKeep | Jmp => (0, 0, false),
        Wr => (2, 0, true),
        WrK | Set | Assert => (1, 0, true),
        SetK | Call | Ret | Halt | Fail => (0, 0, true),
        Native => (2 + (ins.arg & 7).count_ones() as usize, 0, true),
        Obj => ((ins.arg >> 24) as usize, 0, true),
    }
}

fn is_jump(op: Op) -> bool {
    use Op::*;
    matches!(
        op,
        Jmp | Jz
            | Jnz
            | JzKeep
            | JnzKeep
            | JEq
            | JNe
            | JLt
            | JLe
            | JGe
            | JGt
            | JEqK
            | JNeK
            | JLtK
            | JLeK
            | JGeK
            | JGtK
            | Call
    )
}

/// The deepest the stack gets, if `code` is well formed: operands in range, every instruction
/// finding the stack as the code before it and each jump to it leave it, every statement ending
/// with it empty.
fn verify(code: &[Ins], vars: usize, tables: usize, objects: &[Step]) -> Result<usize, String> {
    let len = code.len();
    let mut expect: Vec<Option<usize>> = vec![None; len + 1];
    let mut before = vec![0usize; len];
    let (mut depth, mut deepest, mut live) = (0usize, 0usize, true);
    for (at, ins) in code.iter().enumerate() {
        let bad = |what: &str| Err(format!("instruction {at} ({:?}): {what}", ins.op));
        let in_range = match ins.op {
            Op::Var | Op::Set => (0..vars as i64).contains(&ins.arg),
            Op::SetK => (ins.arg & SETK_VAR) < vars as i64,
            Op::Note => (0..tables as i64).contains(&ins.arg),
            Op::Native => (0..8).contains(&ins.arg),
            Op::Obj => objects
                .get((ins.arg & TARGET) as usize)
                .is_some_and(|o| o.operands as i64 == ins.arg >> 24),
            Op::Call => (0..len as i64).contains(&ins.arg),
            op if is_jump(op) => (ins.arg & TARGET) < len as i64,
            _ => true,
        };
        if !in_range {
            return bad("an operand is out of range");
        }
        match (expect[at], live) {
            (Some(d), true) if d != depth => return bad("the stack differs between paths"),
            (Some(d), false) => depth = d,
            (None, false) => depth = 0,
            _ => {}
        }
        before[at] = depth;
        let (pops, pushes, ends) = effect(ins);
        let keeps = matches!(ins.op, Op::JzKeep | Op::JnzKeep);
        if depth < pops || keeps && depth < 1 {
            return bad("pops an empty stack");
        }
        let after = depth - pops + pushes;
        if is_jump(ins.op) {
            let target = if ins.op == Op::Call {
                ins.arg
            } else {
                ins.arg & TARGET
            } as usize;
            if ins.op == Op::Call && after != 0 {
                return bad("calls inside a statement");
            }
            let arrives = after;
            let known = if target > at {
                expect[target].get_or_insert(arrives)
            } else {
                &mut before[target]
            };
            if *known != arrives {
                return bad("the stack differs at the jump target");
            }
        }
        depth = after - keeps as usize;
        deepest = deepest.max(depth).max(after);
        if ends && depth != 0 {
            return bad("leaves values on the stack");
        }
        live = !matches!(ins.op, Op::Jmp | Op::Ret | Op::Halt | Op::Fail);
    }
    if live && depth != 0 {
        return Err("the code ends inside a statement".into());
    }
    Ok(deepest)
}

impl Vm {
    pub fn new(
        code: Vec<Ins>,
        tables: Vec<Vec<i64>>,
        inputs: Vec<(usize, usize)>,
        vars: usize,
        controls: usize,
        objects: &[Vec<i64>],
    ) -> Result<Vm, String> {
        let objects = objects
            .iter()
            .map(|o| Step::parse(o, vars))
            .collect::<Result<Vec<_>, _>>()?;
        if controls > vars || inputs.iter().any(|&(v, r)| v >= vars || r >= REGISTERS) {
            return Err("a program variable is out of range".into());
        }
        let deepest = verify(&code, vars, tables.len(), &objects)?;
        Ok(Vm {
            code,
            tables,
            objects,
            inputs,
            vars: vec![0; vars],
            controls,
            stack: vec![0; deepest + 1],
            calls: Vec::with_capacity(16),
        })
    }

    pub fn controls(&self) -> &[i64] {
        &self.vars[..self.controls]
    }

    pub fn set_controls(&mut self, values: &[i64]) -> Result<(), String> {
        if values.len() != self.controls {
            return Err("wrong number of control states".into());
        }
        self.vars[..self.controls].copy_from_slice(values);
        Ok(())
    }

    /// Runs the program once with the entry registers `regs`.
    pub fn exec(&mut self, env: &mut Env, regs: &[i64; REGISTERS]) -> Result<Flow, Bail> {
        let Vm {
            code,
            tables,
            objects,
            inputs,
            vars,
            controls,
            stack,
            calls,
        } = self;
        vars[*controls..].fill(0);
        for &(var, reg) in inputs.iter() {
            vars[var] = regs[reg];
        }
        calls.clear();
        let ram = env.mem.ram_cells();
        let slow = env.overlay | if env.io { 1 << 0xD } else { 0 };
        let code = code.as_slice();
        let vars = vars.as_mut_ptr();
        let stack = stack.as_mut_ptr();
        let mut sp = 0usize;
        let mut pc = 0usize;
        // SAFETY (all `unsafe` below): `Vm::new` proved that `pc` stays inside `code`, that
        // every variable index is below `vars.len()`, that the stack never holds more than
        // its length minus one nor pops when empty, and that jumps land at statement starts.
        macro_rules! push {
            ($v:expr) => {{
                let v = $v;
                unsafe { *stack.add(sp) = v };
                sp += 1;
            }};
        }
        macro_rules! pop {
            () => {{
                sp -= 1;
                unsafe { *stack.add(sp) }
            }};
        }
        macro_rules! top {
            () => {
                unsafe { &mut *stack.add(sp - 1) }
            };
        }
        macro_rules! bin {
            ($f:expr) => {{
                let b = pop!();
                let a = top!();
                *a = $f(*a, b);
            }};
        }
        macro_rules! binc {
            ($arg:expr, $f:expr) => {{
                let a = top!();
                *a = $f(*a, $arg);
            }};
        }
        macro_rules! cjump {
            ($arg:expr, $cmp:expr) => {{
                let b = pop!();
                let a = pop!();
                if $cmp(a, b) {
                    pc = ($arg & TARGET) as usize;
                }
            }};
        }
        macro_rules! cjumpk {
            ($arg:expr, $cmp:expr) => {{
                let a = pop!();
                if $cmp(a, $arg >> 24) {
                    pc = ($arg & TARGET) as usize;
                }
            }};
        }
        loop {
            let Ins { op, arg } = unsafe { *code.get_unchecked(pc) };
            pc += 1;
            match op {
                Op::Konst => push!(arg),
                Op::Var => push!(unsafe { *vars.add(arg as usize) }),
                Op::Rd => {
                    let a = top!();
                    *a = read(env, ram, slow, *a)?;
                }
                Op::RdK => push!(read(env, ram, slow, arg)?),
                Op::Add => bin!(|a: i64, b: i64| a.wrapping_add(b) & arg),
                Op::Sub => bin!(|a: i64, b: i64| a.wrapping_sub(b) & arg),
                Op::Mul => bin!(|a: i64, b: i64| a.wrapping_mul(b) & arg),
                Op::And => bin!(|a: i64, b: i64| a & b),
                Op::Or => bin!(|a: i64, b: i64| a | b),
                Op::Xor => bin!(|a: i64, b: i64| a ^ b),
                Op::Shl => {
                    let b = pop!();
                    let a = top!();
                    *a = shl(*a, b)? & arg;
                }
                Op::Shr => {
                    let b = pop!();
                    let a = top!();
                    *a = shr(*a, b)?;
                }
                Op::Eq => bin!(|a: i64, b: i64| (a == b) as i64),
                Op::Ne => bin!(|a: i64, b: i64| (a != b) as i64),
                Op::Ult => bin!(|a: i64, b: i64| (a < b) as i64),
                Op::Ule => bin!(|a: i64, b: i64| (a <= b) as i64),
                Op::Slt => bin!(|a: i64, b: i64| ((a ^ arg) < (b ^ arg)) as i64),
                Op::Sle => bin!(|a: i64, b: i64| ((a ^ arg) <= (b ^ arg)) as i64),
                Op::Word => bin!(|a: i64, b: i64| a | b << 8),
                Op::AddK => binc!(arg, |a: i64, p: i64| a.wrapping_add(p as i32 as i64)
                    & (p >> 32)),
                Op::SubK => binc!(arg, |a: i64, p: i64| a.wrapping_sub(p as i32 as i64)
                    & (p >> 32)),
                Op::ShlK => {
                    let a = top!();
                    *a = shl(*a, arg as i32 as i64)? & (arg >> 32);
                }
                Op::AndK => binc!(arg, |a: i64, k: i64| a & k),
                Op::OrK => binc!(arg, |a: i64, k: i64| a | k),
                Op::XorK => binc!(arg, |a: i64, k: i64| a ^ k),
                Op::ShrK => {
                    let a = top!();
                    *a = shr(*a, arg)?;
                }
                Op::EqK => binc!(arg, |a: i64, k: i64| (a == k) as i64),
                Op::NeK => binc!(arg, |a: i64, k: i64| (a != k) as i64),
                Op::UltK => binc!(arg, |a: i64, k: i64| (a < k) as i64),
                Op::UleK => binc!(arg, |a: i64, k: i64| (a <= k) as i64),
                Op::Not => {
                    let a = top!();
                    *a ^= arg;
                }
                Op::Neg => {
                    let a = top!();
                    *a = a.wrapping_neg() & arg;
                }
                Op::Lo => {
                    let a = top!();
                    *a &= 0xFF;
                }
                Op::Hi => {
                    let a = top!();
                    *a >>= 8;
                }
                Op::Lnot => {
                    let a = top!();
                    *a = (*a == 0) as i64;
                }
                Op::Sext => {
                    let a = top!();
                    if *a & 0x80 != 0 {
                        *a |= 0xFF00;
                    }
                }
                Op::Note => {
                    let i = top!();
                    let table = unsafe { tables.get_unchecked(arg as usize) };
                    *i = *usize::try_from(*i)
                        .ok()
                        .and_then(|i| table.get(i))
                        .ok_or(Bail::Range)?;
                }
                Op::Wr => {
                    let v = pop!();
                    let a = pop!();
                    store(env, a, v)?;
                }
                Op::WrK => {
                    let v = pop!();
                    store(env, arg, v)?;
                }
                Op::Set => {
                    let v = pop!();
                    unsafe { *vars.add(arg as usize) = v };
                }
                Op::SetK => unsafe { *vars.add((arg & SETK_VAR) as usize) = arg >> 20 },
                Op::Jmp => pc = (arg & TARGET) as usize,
                Op::Jz => {
                    if pop!() == 0 {
                        pc = (arg & TARGET) as usize;
                    }
                }
                Op::Jnz => {
                    if pop!() != 0 {
                        pc = (arg & TARGET) as usize;
                    }
                }
                Op::JzKeep => {
                    if top!() == &0 {
                        pc = (arg & TARGET) as usize;
                    } else {
                        sp -= 1;
                    }
                }
                Op::JnzKeep => {
                    if top!() != &0 {
                        pc = (arg & TARGET) as usize;
                    } else {
                        sp -= 1;
                    }
                }
                Op::JEq => cjump!(arg, |a: i64, b: i64| a == b),
                Op::JNe => cjump!(arg, |a: i64, b: i64| a != b),
                Op::JLt => cjump!(arg, |a: i64, b: i64| a < b),
                Op::JLe => cjump!(arg, |a: i64, b: i64| a <= b),
                Op::JGe => cjump!(arg, |a: i64, b: i64| a >= b),
                Op::JGt => cjump!(arg, |a: i64, b: i64| a > b),
                Op::JEqK => cjumpk!(arg, |a: i64, b: i64| a == b),
                Op::JNeK => cjumpk!(arg, |a: i64, b: i64| a != b),
                Op::JLtK => cjumpk!(arg, |a: i64, b: i64| a < b),
                Op::JLeK => cjumpk!(arg, |a: i64, b: i64| a <= b),
                Op::JGeK => cjumpk!(arg, |a: i64, b: i64| a >= b),
                Op::JGtK => cjumpk!(arg, |a: i64, b: i64| a > b),
                Op::Call => {
                    calls.push(pc);
                    pc = arg as usize;
                }
                Op::Ret => pc = calls.pop().ok_or(Bail::Spec)?,
                Op::Halt => return Ok(Flow::Done),
                Op::Assert => {
                    if pop!() == 0 {
                        return Err(Bail::Spec);
                    }
                }
                Op::Fail => return Err(Bail::Spec),
                Op::Obj => {
                    let step = unsafe { objects.get_unchecked((arg & TARGET) as usize) };
                    sp -= step.operands;
                    let ops = unsafe { std::slice::from_raw_parts(stack.add(sp), step.operands) };
                    let flag = step.run(env, ram, slow, ops)?;
                    if let Some(out) = step.out {
                        unsafe { *vars.add(out) = flag };
                    }
                }
                Op::Native => {
                    let resume = pop!();
                    let mut given = [None; 3];
                    for (i, slot) in given.iter_mut().enumerate().rev() {
                        if arg >> i & 1 != 0 {
                            *slot = Some(byte(pop!())?);
                        }
                    }
                    let addr = pop!();
                    let (addr, resume) = (word(addr)?, word(resume)?);
                    if env.native.run(env.log, addr, given, resume)? {
                        return Ok(Flow::Escape);
                    }
                }
            }
        }
    }
}

fn byte(v: i64) -> Result<u8, Bail> {
    u8::try_from(v).map_err(|_| Bail::Range)
}

fn word(v: i64) -> Result<u16, Bail> {
    u16::try_from(v).map_err(|_| Bail::Range)
}

/// Python's `a << b` kept to the bits a mask of at most 62 bits keeps.
#[inline(always)]
fn shl(a: i64, b: i64) -> Result<i64, Bail> {
    if b < 0 {
        return Err(Bail::Range);
    }
    Ok(if b >= 64 { 0 } else { a.wrapping_shl(b as u32) })
}

/// Python's `a >> b`.
#[inline(always)]
fn shr(a: i64, b: i64) -> Result<i64, Bail> {
    if b < 0 {
        return Err(Bail::Range);
    }
    Ok(a >> b.min(63))
}

#[inline(always)]
pub(crate) fn read(env: &Env, ram: &[Cell<u8>], slow: u16, addr: i64) -> Result<i64, Bail> {
    if addr as u64 > 0xFFFF {
        return Err(Bail::Range);
    }
    let addr = addr as u16;
    if addr != 1 && slow >> (addr >> 12) & 1 == 0 {
        // SAFETY: `ram` is the 64 KiB and `addr` is below it.
        return Ok(unsafe { ram.get_unchecked(addr as usize) }.get() as i64);
    }
    read_mapped(env, addr)
}

#[cold]
fn read_mapped(env: &Env, addr: u16) -> Result<i64, Bail> {
    if addr == 1 {
        return Ok(port_read(env.mem.ram(0), env.mem.ram(1)) as i64);
    }
    if env.io && addr >> 12 == 0xD && special(addr) {
        return Err(Bail::Device(addr));
    }
    Ok(env.mem.peek(&mut NoHooks, addr) as i64)
}

#[inline(always)]
pub(crate) fn store(env: &mut Env, addr: i64, value: i64) -> Result<(), Bail> {
    if addr as u64 > 0xFFFF || value as u64 > 0xFF {
        return Err(Bail::Range);
    }
    let (addr, byte) = match env.routes.apply(addr as u16, value as u8)? {
        Some(landed) => landed,
        None => return Ok(()),
    };
    if addr <= 1 {
        return Err(Bail::Port);
    }
    if (SID_BASE..SID_BASE + SID_REGS).contains(&(addr as i64)) {
        env.sid.push(((addr - SID_BASE as u16) as u8, byte));
    }
    env.log.push((addr, env.mem.ram(addr)));
    env.mem.set_ram(addr, byte);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A native hook that records its calls and returns from the play call when asked.
    #[derive(Default)]
    struct Hook {
        calls: Vec<(u16, [Option<u8>; 3], u16)>,
        escape: bool,
    }

    impl Native for Hook {
        fn run(
            &mut self,
            _log: &mut Vec<(u16, u8)>,
            addr: u16,
            regs: [Option<u8>; 3],
            resume: u16,
        ) -> Result<bool, Bail> {
            self.calls.push((addr, regs, resume));
            Ok(self.escape)
        }
    }

    type Run = (Result<(), Bail>, FlatMemory, Vec<(u8, u8)>);

    fn run_io(code: &[(Op, i64)], vars: usize, io: bool) -> Run {
        let ins = code.iter().map(|&(op, arg)| Ins { op, arg }).collect();
        let mut vm = Vm::new(ins, vec![vec![10, 20, 30]], vec![], vars, 0, &[]).unwrap();
        let mem = FlatMemory::new();
        let (mut log, mut sid) = (Vec::new(), Vec::new());
        let mut hook = Hook::default();
        let r = vm.exec(
            &mut Env {
                routes: &Routes::default(),
                mem: &mem,
                io,
                overlay: 0,
                log: &mut log,
                sid: &mut sid,
                native: &mut hook,
            },
            &[0; REGISTERS],
        );
        (r.map(|_| ()), mem, sid)
    }

    fn run(code: &[(Op, i64)], vars: usize) -> Run {
        run_io(code, vars, false)
    }

    fn eval(code: &[(Op, i64)]) -> Result<i64, Bail> {
        let mut full = code.to_vec();
        full.extend([(Op::Set, 0), (Op::Halt, 0)]);
        let ins = full.iter().map(|&(op, arg)| Ins { op, arg }).collect();
        let mut vm = Vm::new(ins, vec![vec![10, 20, 30]], vec![], 1, 1, &[]).unwrap();
        let mem = FlatMemory::new();
        let (mut log, mut sid) = (Vec::new(), Vec::new());
        let mut hook = Hook::default();
        let mut env = Env {
            routes: &Routes::default(),
            mem: &mem,
            io: false,
            overlay: 0,
            log: &mut log,
            sid: &mut sid,
            native: &mut hook,
        };
        vm.exec(&mut env, &[0; REGISTERS]).map(|_| vm.vars[0])
    }

    #[test]
    fn stores_arithmetic_and_sid_writes() {
        let (r, mem, sid) = run(
            &[
                (Op::Konst, 0xD404),
                (Op::Konst, 200),
                (Op::Konst, 100),
                (Op::Add, 0xFF),
                (Op::Wr, 0),
                (Op::Konst, 7),
                (Op::WrK, 0x2000),
                (Op::Halt, 0),
            ],
            0,
        );
        assert_eq!(r, Ok(()));
        assert_eq!(mem.ram(0xD404), 44);
        assert_eq!(mem.ram(0x2000), 7);
        assert_eq!(sid, vec![(4, 44)]);
    }

    #[test]
    fn python_semantics_of_logic_and_shifts() {
        let and = |a| eval(&[(Op::Konst, a), (Op::JzKeep, 3), (Op::Konst, 7)]);
        let or = |a| eval(&[(Op::Konst, a), (Op::JnzKeep, 3), (Op::Konst, 7)]);
        assert_eq!((and(0), and(3)), (Ok(0), Ok(7)));
        assert_eq!((or(0), or(3)), (Ok(7), Ok(3)));
        let ite = |c| {
            eval(&[
                (Op::Konst, c),
                (Op::Jz, 4),
                (Op::Konst, 11),
                (Op::Jmp, 5),
                (Op::Konst, 22),
            ])
        };
        assert_eq!((ite(1), ite(0)), (Ok(11), Ok(22)));
        assert_eq!(
            eval(&[(Op::Konst, 5), (Op::Konst, 70), (Op::Shl, 0xFF)]),
            Ok(0)
        );
        assert_eq!(
            eval(&[(Op::Konst, -8), (Op::Konst, 70), (Op::Shr, 0)]),
            Ok(-1)
        );
        assert_eq!(
            eval(&[(Op::Konst, 1), (Op::Konst, -1), (Op::Shl, 0xFF)]),
            Err(Bail::Range)
        );
        assert_eq!(eval(&[(Op::Konst, 0x80), (Op::Sext, 0)]), Ok(0xFF80));
        assert_eq!(
            eval(&[(Op::Konst, 0x7F), (Op::Konst, 0x80), (Op::Slt, 0x80)]),
            Ok(0)
        );
        assert_eq!(eval(&[(Op::Konst, 1), (Op::Note, 0)]), Ok(20));
        assert_eq!(eval(&[(Op::Konst, 3), (Op::Note, 0)]), Err(Bail::Range));
        assert_eq!(
            eval(&[(Op::Konst, 1), (Op::Konst, 0), (Op::Sub, 0xFF)]),
            Ok(1)
        );
        assert_eq!(
            eval(&[(Op::Konst, 0), (Op::Konst, 1), (Op::Sub, 0xFF)]),
            Ok(255)
        );
    }

    #[test]
    fn immediate_forms_agree_with_the_general_ones() {
        let packed = |mask: i64, k: i64| mask << 32 | (k as u32 as i64);
        assert_eq!(
            eval(&[(Op::Konst, 250), (Op::AddK, packed(0xFF, 10))]),
            Ok(4)
        );
        assert_eq!(
            eval(&[(Op::Konst, 3), (Op::SubK, packed(0xFF, 5))]),
            Ok(254)
        );
        assert_eq!(
            eval(&[(Op::Konst, 3), (Op::SubK, packed(0xFFFF, -5))]),
            Ok(8)
        );
        assert_eq!(
            eval(&[(Op::Konst, 3), (Op::ShlK, packed(0xFF, 7))]),
            Ok(0x80)
        );
        assert_eq!(eval(&[(Op::Konst, 6), (Op::AndK, 3)]), Ok(2));
        assert_eq!(eval(&[(Op::Konst, 6), (Op::EqK, 6)]), Ok(1));
        assert_eq!(eval(&[(Op::Konst, 6), (Op::UltK, 6)]), Ok(0));
        assert_eq!(eval(&[(Op::Konst, 6), (Op::UleK, 6)]), Ok(1));
        assert_eq!(eval(&[(Op::Konst, 6), (Op::ShrK, 1)]), Ok(3));
    }

    #[test]
    fn bails_where_python_answers() {
        let store_to = |addr: i64, value: i64| {
            run(
                &[
                    (Op::Konst, addr),
                    (Op::Konst, value),
                    (Op::Wr, 0),
                    (Op::Halt, 0),
                ],
                0,
            )
            .0
        };
        assert_eq!(store_to(1, 0x35), Err(Bail::Port));
        assert_eq!(store_to(0x10000, 1), Err(Bail::Range));
        assert_eq!(store_to(0x1000, 256), Err(Bail::Range));
        assert_eq!(store_to(0x1000, -1), Err(Bail::Range));
        assert_eq!(run(&[(Op::Fail, 0)], 0).0, Err(Bail::Spec));
        assert_eq!(
            run_io(&[(Op::RdK, 0xD012), (Op::Set, 0), (Op::Halt, 0)], 1, true).0,
            Err(Bail::Device(0xD012))
        );
        assert_eq!(
            run(&[(Op::RdK, 0xD012), (Op::Set, 0), (Op::Halt, 0)], 1).0,
            Ok(())
        );
        assert_eq!(
            run(&[(Op::RdK, 1), (Op::Set, 0), (Op::Halt, 0)], 1).0,
            Ok(())
        );
        assert_eq!(
            run(&[(Op::RdK, 0x10000), (Op::Set, 0), (Op::Halt, 0)], 1).0,
            Err(Bail::Range)
        );
    }

    #[test]
    fn jumps_calls_and_compare_branches() {
        let k = |target: i64, k: i64| k << 24 | target;
        let (r, mem, _) = run(
            &[
                (Op::Call, 7),
                (Op::Var, 0),
                (Op::JNeK, k(6, 9)),
                (Op::Konst, 1),
                (Op::WrK, 0x2000),
                (Op::Halt, 0),
                (Op::Halt, 0),
                (Op::SetK, 9 << 20),
                (Op::Ret, 0),
            ],
            1,
        );
        assert_eq!(r, Ok(()));
        assert_eq!(mem.ram(0x2000), 1);
        let (_, mem, _) = run(
            &[
                (Op::Konst, 3),
                (Op::Konst, 4),
                (Op::JLt, 5),
                (Op::Konst, 1),
                (Op::WrK, 0x2001),
                (Op::Halt, 0),
            ],
            0,
        );
        assert_eq!(mem.ram(0x2001), 0);
    }

    #[test]
    fn rejects_malformed_programs() {
        let one = |code: &[(Op, i64)]| {
            let ins = code.iter().map(|&(op, arg)| Ins { op, arg }).collect();
            Vm::new(ins, vec![], vec![], 1, 0, &[]).is_err()
        };
        assert!(one(&[(Op::Jmp, 5)]));
        assert!(one(&[(Op::Var, 1), (Op::Set, 0)]));
        assert!(one(&[(Op::Konst, 1), (Op::Note, 0), (Op::Set, 0)]));
        assert!(one(&[(Op::Set, 0)]));
        assert!(one(&[(Op::Konst, 1)]));
        assert!(one(&[(Op::Konst, 1), (Op::Konst, 2), (Op::Set, 0)]));
        assert!(one(&[
            (Op::Konst, 1),
            (Op::Jz, 3),
            (Op::Konst, 2),
            (Op::Set, 0)
        ]));
        assert!(Vm::new(vec![], vec![], vec![(0, 9)], 1, 0, &[]).is_err());
        assert!(Vm::new(vec![], vec![], vec![], 1, 2, &[]).is_err());
        assert!(!one(&[(Op::Halt, 0)]));
    }

    #[test]
    fn native_statements_run_the_tunes_code() {
        let ins = |escape| {
            let code: Vec<Ins> = [
                (Op::Konst, 0x1234),
                (Op::Konst, 5),
                (Op::Konst, 6),
                (Op::Konst, 0x1240),
                (Op::Native, 0b101),
                (Op::Konst, 1),
                (Op::WrK, 0x2000),
                (Op::Halt, 0),
            ]
            .iter()
            .map(|&(op, arg)| Ins { op, arg })
            .collect();
            let mut vm = Vm::new(code, vec![], vec![], 0, 0, &[]).unwrap();
            let mem = FlatMemory::new();
            let (mut log, mut sid) = (Vec::new(), Vec::new());
            let mut hook = Hook {
                escape,
                ..Hook::default()
            };
            let flow = vm
                .exec(
                    &mut Env {
                        routes: &Routes::default(),
                        mem: &mem,
                        io: false,
                        overlay: 0,
                        log: &mut log,
                        sid: &mut sid,
                        native: &mut hook,
                    },
                    &[0; REGISTERS],
                )
                .unwrap();
            (flow, hook.calls, mem.ram(0x2000))
        };
        let (flow, calls, wrote) = ins(false);
        assert_eq!(flow, Flow::Done);
        assert_eq!(calls, vec![(0x1234, [Some(5), None, Some(6)], 0x1240)]);
        assert_eq!(wrote, 1);
        let (flow, _, wrote) = ins(true);
        assert_eq!((flow, wrote), (Flow::Escape, 0));
    }

    #[test]
    fn routed_stores_land_where_the_build_plays_them() {
        let routes = Routes::new(
            Some(0x3000),
            &[(0xD401, Some(0xD405), 0x0F, 0x80), (0xD402, None, 0, 0)],
        );
        let store = |addr, v| {
            vec![(Op::Konst, 0), (Op::Konst, v), (Op::Wr, 0), (Op::Halt, 0)]
                .into_iter()
                .map(|(op, arg)| Ins {
                    op,
                    arg: if op == Op::Konst && arg == 0 {
                        addr
                    } else {
                        arg
                    },
                })
                .collect::<Vec<_>>()
        };
        let mem = FlatMemory::new();
        for (addr, v) in [(0xD401, 0xFF), (0xD402, 5), (0xD404, 7), (0x2000, 9)] {
            let mut vm = Vm::new(store(addr, v), vec![], vec![], 0, 0, &[]).unwrap();
            let (mut log, mut sid) = (Vec::new(), Vec::new());
            let mut hook = Hook::default();
            vm.exec(
                &mut Env {
                    routes: &routes,
                    mem: &mem,
                    io: false,
                    overlay: 0,
                    log: &mut log,
                    sid: &mut sid,
                    native: &mut hook,
                },
                &[0; REGISTERS],
            )
            .unwrap();
        }
        assert_eq!(mem.ram(0x3005), 0x8F);
        assert_eq!((mem.ram(0xD402), mem.ram(0x3002)), (0, 0));
        assert_eq!(mem.ram(0x3004), 7);
        assert_eq!(mem.ram(0x2000), 9);
        assert!(!routes.is_empty() && Routes::default().is_empty());
    }

    #[test]
    fn controls_persist_and_locals_reset() {
        let ins = [
            (Op::Var, 0),
            (Op::AddK, 0xFF_i64 << 32 | 1),
            (Op::Set, 0),
            (Op::Var, 1),
            (Op::AddK, 0xFF_i64 << 32 | 1),
            (Op::Set, 1),
            (Op::Halt, 0),
        ]
        .iter()
        .map(|&(op, arg)| Ins { op, arg })
        .collect();
        let mut vm = Vm::new(ins, vec![], vec![], 2, 1, &[]).unwrap();
        let mem = FlatMemory::new();
        let (mut log, mut sid) = (Vec::new(), Vec::new());
        for _ in 0..3 {
            let mut hook = Hook::default();
            let mut env = Env {
                routes: &Routes::default(),
                mem: &mem,
                io: false,
                overlay: 0,
                log: &mut log,
                sid: &mut sid,
                native: &mut hook,
            };
            vm.exec(&mut env, &[0; REGISTERS]).unwrap();
        }
        assert_eq!(vm.controls(), &[3]);
        assert_eq!(vm.vars[1], 1);
        assert!(vm.set_controls(&[1, 2]).is_err());
    }

    /// Runs `code` with the object description `object` on `mem`; the variables.
    fn react(code: &[(Op, i64)], object: &[i64], vars: usize, mem: &FlatMemory) -> Vec<i64> {
        let ins = code.iter().map(|&(op, arg)| Ins { op, arg }).collect();
        let mut vm = Vm::new(ins, vec![], vec![], vars, 0, &[object.to_vec()]).unwrap();
        let (mut log, mut sid) = (Vec::new(), Vec::new());
        let mut hook = Hook::default();
        let mut env = Env {
            routes: &Routes::default(),
            mem,
            io: false,
            overlay: 0,
            log: &mut log,
            sid: &mut sid,
            native: &mut hook,
        };
        vm.exec(&mut env, &[0; REGISTERS]).unwrap();
        vm.vars.clone()
    }

    fn operands(values: &[i64], object: usize) -> Vec<(Op, i64)> {
        let mut code: Vec<_> = values.iter().map(|&v| (Op::Konst, v)).collect();
        code.push((Op::Obj, object as i64 | (values.len() as i64) << 24));
        code.push((Op::Halt, 0));
        code
    }

    #[test]
    fn counters_wrap_up_and_down_and_flag_it() {
        let mem = FlatMemory::new();
        mem.set_ram(0x10, 4);
        let up = vec![0, 0, 2, 0x10, -1, 1, 1];
        let down = vec![0, 0, 2, 0x10, -1, 2, 2];
        assert_eq!(react(&operands(&[4, 0], 0), &up, 1, &mem)[0], 1);
        assert_eq!(mem.ram(0x10), 0);
        assert_eq!(react(&operands(&[4, 9], 0), &down, 1, &mem)[0], 1);
        assert_eq!(mem.ram(0x10), 9);
        assert_eq!(react(&operands(&[4, 9], 0), &down, 1, &mem)[0], 0);
        assert_eq!(mem.ram(0x10), 7);
        assert_eq!(react(&operands(&[9, 0], 0), &up, 1, &mem)[0], 0);
        assert_eq!(mem.ram(0x10), 8);
    }

    #[test]
    fn owners_claim_and_release_their_own_id() {
        let mem = FlatMemory::new();
        let claim = vec![1, -1, 3, 0x20, -1, 0];
        let release = vec![1, -1, 3, 0x20, -1, 1];
        react(&operands(&[2, 0xFF, 1], 0), &claim, 0, &mem);
        assert_eq!(mem.ram(0x20), 2);
        react(&operands(&[3, 0xFF, 1], 0), &release, 0, &mem);
        assert_eq!(mem.ram(0x20), 2);
        react(&operands(&[2, 0xFF, 0], 0), &release, 0, &mem);
        assert_eq!(mem.ram(0x20), 2);
        react(&operands(&[2, 0xFF, 1], 0), &release, 0, &mem);
        assert_eq!(mem.ram(0x20), 0xFF);
    }

    fn walker(reset: i64) -> Vec<i64> {
        // two columns, rows of width 2: 0..=0x7f sweeps value 0 by column 1 (count column 0),
        // 0x80 sets it, 0xfe jumps to column 1, anything else stops
        let x = |c: i64| [c, 0xFF, 0, 0];
        let none = [0, 0, 0, 0];
        let row = |lo: i64, hi: i64, kind: i64, a: [i64; 4], b: [i64; 4], sets: Vec<i64>| {
            let mut r = vec![lo, hi, kind, 0, 0];
            r.extend(a);
            r.extend(b);
            r.extend(sets);
            r.push(0);
            r
        };
        let set = vec![1, 0, 0, 1, 0xFF, 0, 0];
        let mut d = vec![2, -1, 3, 2, 2, 0x30, -1, 0x32, -1, 1, 0x34, -1, 4];
        d.extend(row(0, 0x7F, 2, x(0), x(1), vec![0]));
        d.extend(row(0x80, 0x80, 4, none, none, set));
        d.extend(row(0xFE, 0xFE, 1, x(1), none, vec![0]));
        d.extend(row(0xFF, 0xFF, 0, none, none, vec![0]));
        d.push(reset);
        d
    }

    #[test]
    fn walkers_sweep_set_jump_and_stop() {
        let mem = FlatMemory::new();
        for (i, b) in [1u8, 0x10, 0x80, 0x55, 0xFE, 0, 0xFF, 0].iter().enumerate() {
            mem.set_ram(0x2000 + i as u16, *b);
        }
        let d = walker(0);
        let step = operands(&[0x2000, 0x2001, 0], 0);
        let frame = || react(&step, &d, 0, &mem);
        let at = || (mem.ram(0x30), mem.ram(0x32), mem.ram(0x34));
        let seen: Vec<_> = (0..4)
            .map(|_| {
                frame();
                at()
            })
            .collect();
        assert_eq!(seen, [(0, 1, 16), (2, 0, 16), (4, 0, 0x55), (0, 1, 0x65)]);
        react(&operands(&[0x2000, 0x2001, 6], 0), &walker(1), 0, &mem);
        assert_eq!((mem.ram(0x30), mem.ram(0x32)), (6, 0));
    }

    #[test]
    fn a_walker_that_jumps_forever_bails() {
        let mem = FlatMemory::new();
        mem.set_ram(0x2000, 0xFE);
        mem.set_ram(0x2001, 2);
        mem.set_ram(0x2002, 0xFE);
        mem.set_ram(0x2003, 0);
        let ins = operands(&[0x2000, 0x2001, 0], 0)
            .iter()
            .map(|&(op, arg)| Ins { op, arg })
            .collect();
        let mut vm = Vm::new(ins, vec![], vec![], 0, 0, &[walker(0)]).unwrap();
        let (mut log, mut sid) = (Vec::new(), Vec::new());
        let mut hook = Hook::default();
        let mut env = Env {
            routes: &Routes::default(),
            mem: &mem,
            io: false,
            overlay: 0,
            log: &mut log,
            sid: &mut sid,
            native: &mut hook,
        };
        assert_eq!(vm.exec(&mut env, &[0; REGISTERS]), Err(Bail::Spec));
    }

    #[test]
    fn cursors_read_runs_and_chained_fields() {
        let mem = FlatMemory::new();
        for (i, b) in [0x82u8, 0x05, 0xF2, 0x03].iter().enumerate() {
            mem.set_ram(0x2000 + i as u16, *b);
        }
        // run 0xF0..=0xF7 from 0xF0 into 0x41; field 0 -> 0x42 (& 0x7f), field 1 when bit 7 -> 0x43
        let d = vec![
            3, -1, 2, 0x40, -1, 0, 0xF0, 0xF7, 0xF0, 0x41, -1, 2, 0, 0, 1, 1, 0x42, -1, 0x7F, 0, 0,
            0x80, 0, 0, 1, 0x43, -1, 0xFF, 0, 0, 0,
        ];
        let step = operands(&[0x2000, 0], 0);
        react(&step, &d, 0, &mem);
        assert_eq!((mem.ram(0x40), mem.ram(0x42), mem.ram(0x43)), (2, 2, 5));
        react(&step, &d, 0, &mem);
        assert_eq!((mem.ram(0x40), mem.ram(0x41), mem.ram(0x42)), (3, 2, 0));
        react(&step, &d, 0, &mem);
        assert_eq!((mem.ram(0x40), mem.ram(0x41)), (3, 1));
        react(&step, &d, 0, &mem);
        assert_eq!((mem.ram(0x40), mem.ram(0x41)), (3, 0));
    }

    #[test]
    fn descriptions_and_operand_counts_are_checked() {
        let bad = |d: Vec<i64>, n: i64| {
            let ins = vec![Ins {
                op: Op::Obj,
                arg: n << 24,
            }];
            Vm::new(ins, vec![], vec![], 1, 0, &[d]).is_err()
        };
        assert!(bad(vec![0, 0, 2, 0x10, -1, 1, 1], 1));
        assert!(bad(vec![0, 5, 2, 0x10, -1, 1, 1], 2));
        assert!(bad(vec![9, -1, 0], 0));
        assert!(bad(vec![0, -1, 2, 0x10, -1, 1], 2));
        assert!(bad(vec![0, -1, 2, 0x10, -1, 1, 1, 7], 2));
        assert!(bad(vec![0, -1, 2, 0x1_0000, -1, 1, 1], 2));
        assert!(Step::parse(&[2, -1, 1, 1, 1, 0x30, -1, 0x31, -1, 0, 0, 0], 0).is_err());
    }
}
