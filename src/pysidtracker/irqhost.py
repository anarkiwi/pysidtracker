"""Run a tune's real interrupt structure, cycle for cycle, as the sidtrace oracle does.

:class:`InterruptHost` boots a C64 the way sidplayfp does and runs a PSID or RSID
on jennings, with every VIC-II and CIA interrupt and VIC DMA modelled; see
``docs/irqhost.md`` for what is modelled and how it is validated.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterator, List, Optional, Tuple, Union

from vagrant_moses.cia import Cia  # core dependency
from vagrant_moses.eventsched import PHI1, Event, Scheduler  # core dependency
from vagrant_moses.vicii import NTSC_GEOMETRY, PAL_GEOMETRY, Vic  # core dependency

from . import registers as reg
from .asm6502 import assemble
from .cadence import Cadence, TriggerSource, _driver_base, _iomap
from .emu import SidReads
from .errors import EmulatorUnavailable, SidParseError
from .image import SidImage
from .oracle import (
    SidtraceRow,
    _raise_frames,
    nested,
    play_source,
    raise_cycles,
    sidtrace_cadence,
    sidtrace_grid,
)
from .player import MemPlayer
from .poweron import PSID_DRIVER_CLEARED, ram_pattern
from .roms import poweron_writes
from .vicii import NTSC_GEOMETRY, PAL_GEOMETRY, Vic

_MAX = 65536
# Entry kinds of libsidplayfp's per-opcode cycle tables; those below _READS write.
W, PUSHPCH, LOWPC, PUSHSR = range(4)
R, X2, F, CLI, SEI, POPSR, BR, BRK0, VECLO, VECHI, FN, RTI, HALT, ANE = range(4, 18)
_ANE_MAGIC = 0xEF
_READS = R
_FETCHES = frozenset((F, CLI, SEI, RTI, FN))
BRK_ENTRIES = (BRK0, PUSHPCH, LOWPC, PUSHSR, VECLO, VECHI, FN)
_CLI_BASE, _SEI_BASE = 0x58 << 3, 0x78 << 3
_NMI, _VIDEO, _CIA = 0, 1, 2
_RAISE = ("nmi", "video_irq", "cia_irq")
_IRQ_I = 0x04

_READ_OPS = frozenset(
    "ADC AND CMP EOR LDA LDX LDY ORA SBC BIT CPX CPY LAX LAS ANC ALR ARR ANE LXA "
    "SBX NOP".split()
)
_RMW_OPS = frozenset("ASL LSR ROL ROR INC DEC SLO RLA SRE RRA DCP ISC".split())
_STORE_OPS = frozenset("STA STX STY SAX SHA SHX SHY TAS".split())
_BRANCHES = {
    "BPL": (0x80, 0),
    "BMI": (0x80, 1),
    "BVC": (0x40, 0),
    "BVS": (0x40, 1),
    "BCC": (0x01, 0),
    "BCS": (0x01, 1),
    "BNE": (0x02, 0),
    "BEQ": (0x02, 1),
}
_OP_TAIL = {
    "CLI": (CLI,),
    "SEI": (SEI,),
    "PHA": (W,),
    "PHP": (W,),
    "PLA": (R, R),
    "PLP": (R, POPSR),
    "JSR": (R, W, W, R, F),
    "JMP": (F,),
    "RTS": (R, R, R, R),
    "RTI": (R, POPSR, R, R, RTI),
}
_PREFIX = {
    "impl": (R,),
    "acc": (R,),
    "imm": (R,),
    "rel": (R,),
    "zp": (R,),
    "zpx": (R, R),
    "zpy": (R, R),
    "abs": (R, R),
    "ind": (R, R, R, R),
    "indx": (R, R, R, R),
}


def opcode_layout(op: int) -> Tuple[Tuple[int, ...], int, str]:
    """libsidplayfp's cycle entries after ``op``'s fetch, its exec entry and kind.

    The exec entry is where jennings runs the instruction (its data access); the
    kind is ``"rmw"``, ``"absx"``/``"absy"``/``"indy"`` (page-cross skip), ``"branch"``
    or ``""``.
    """
    from jennings.opcodes import MEM_MODES, OPCODES

    mn, mode = OPCODES[op]
    if mn == "JAM":
        return (HALT,), -1, ""
    if mn == "BRK":
        return BRK_ENTRIES, -1, ""
    if mn == "ANE":
        return (ANE, F), -1, ""
    reads = mn in _READ_OPS and mode in MEM_MODES
    rmw = mn in _RMW_OPS and mode != "acc"
    if mode in ("absx", "absy", "indy"):
        skip = X2 if reads else R
        prefix = (R, skip, R) if mode != "indy" else (R, R, skip, R)
    elif mn == "JSR":
        prefix = (R,)
    else:
        prefix = _PREFIX[mode]
    ents = list(prefix) + ([R] if reads or rmw else [])
    if mn in _BRANCHES:
        ents += [BR, R]
    elif rmw:
        ents += [W, W]
    elif mn in _STORE_OPS:
        ents += [W]
    elif mn in _OP_TAIL:
        ents += _OP_TAIL[mn]
    elif mn != "NOP":
        ents += [F]
    if ents[-1] not in _FETCHES:
        ents.append(F)
    exec_at = next(i for i, k in enumerate(ents) if k not in (R, X2)) - 1
    kind = "branch" if mn in _BRANCHES else "rmw" if rmw else mode if X2 in ents else ""
    return tuple(ents), exec_at, kind


def _layouts():
    from jennings.opcodes import OPCODES

    out = []
    for op in range(256):
        ents, exec_at, kind = opcode_layout(op)
        out.append((ents, exec_at + 1, kind, op << 3, _BRANCHES.get(OPCODES[op][0])))
    return out


# KERNAL jump-table entries and the routines they call, each an RTS in the stub.
_KERNAL_CALLS = (
    0xFF81, 0xFF5B, 0xFF84, 0xFDA3, 0xFF87, 0xFD50, 0xFF8A, 0xFD15, 0xFF8D, 0xFD1A,
    0xFF90, 0xFE18, 0xFF93, 0xEDB9, 0xFF96, 0xEDC7, 0xFF99, 0xFE25, 0xFF9C, 0xFE34,
    0xFF9F, 0xEA87, 0xFFA2, 0xFE21, 0xFFA5, 0xEE13, 0xFFA8, 0xEDDD, 0xFFAB, 0xEDEF,
    0xFFAE, 0xEDFE, 0xFFB1, 0xED0C, 0xFFB4, 0xED09, 0xFFB7, 0xFE07, 0xFFBA, 0xFE00,
    0xFFBD, 0xFDF9, 0xFFC0, 0xF34A, 0xFFC3, 0xF291, 0xFFC6, 0xF20E, 0xFFC9, 0xF250,
    0xFFCC, 0xF333, 0xFFCF, 0xF157, 0xFFD2, 0xF1CA, 0xFFD5, 0xF49E, 0xFFD8, 0xF5DD,
    0xFFDB, 0xF6E4, 0xFFDE, 0xF6DD, 0xFFE1, 0xF6ED, 0xFFE4, 0xF13E, 0xFFE7, 0xF32F,
    0xFFEA, 0xF69B, 0xFFED, 0xE505, 0xFFF0, 0xE50A, 0xFFF3, 0xE500,
)  # fmt: skip


def kernal_stub() -> bytes:
    """sidplayfp's ROM-less KERNAL (``SystemROMBanks.h``): NOPs and a few stubs."""
    rom = bytearray(b"\xea" * 0x2000)
    save, restore = bytes((0x48, 0x8A, 0x48, 0x98, 0x48)), bytes(
        (0x68, 0xA8, 0x68, 0xAA, 0x68)
    )

    def put(addr, data):
        rom[addr - 0xE000 : addr - 0xE000 + len(data)] = data

    put(0xEA31, b"\x4c\x7e\xea")
    put(0xEA7E, b"\x0c\x0d\xdc")
    put(0xEA81, restore + b"\x40")
    put(0xFCE2, b"\x02")
    put(0xFE43, b"\x78\x6c\x18\x03")
    put(0xFE47, save)
    put(0xFEBC, restore + b"\x40")
    put(0xFF48, save + b"\x6c\x14\x03")
    put(0xFFFA, bytes((0x43, 0xFE, 0xE2, 0xFC, 0x48, 0xFF)))
    for addr in _KERNAL_CALLS:
        rom[addr - 0xE000] = 0x60
    return bytes(rom)


def boot_cycle(ntsc: bool) -> int:
    """The cycle sidplayfp's CPU first runs: after its fixed power-on event count.

    sidtrace pins the power-on delay to its maximum, so ``Player::initialise``
    fires ``3 * (0x1FFF + 8000)`` VIC, CIA-timer and TOD events before the reset.
    """
    cpl = (NTSC_GEOMETRY if ntsc else PAL_GEOMETRY)[1]
    pattern = (0, 1, 11, 14, 15, 54, 55, 56, 57) + ((58,) if ntsc else ())
    tod = 17045 if ntsc else 19704
    target = 3 * (0x1FFF + 8000)
    lines = target // len(pattern) + 2
    times = [0, 1, 1, 1, 1]
    times += [n * cpl + lc for n in range(lines) for lc in pattern if n * cpl + lc]
    times += [k * tod for k in range(lines * cpl // tod + 1) for _ in (0, 1)]
    times.sort()
    return times[target - 1]


_DRIVER = """
playnum: .byte 0
speed: .byte 0
initvec: .word 0
playvec: .word 0
initiomap: .byte 0
playiomap: .byte 0
video: .byte 0
clock: .byte 0
flags: .byte 0
play: JMP (playvec)
init: JMP (initvec)
cold: SEI
  LDA #$00
  STA $D01A
  LDA $D019
  STA $D019
  LDA #$7F
  STA $DC0D
  STA $DD0D
  LDA $DC0D
  LDA $DD0D
  LDA #$0F
  STA $D418
  LDA video
  BEQ ntsc
  LDX #$25
  LDY #$40
  JMP timer
ntsc: LDX #$95
  LDY #$42
timer: STX $DC04
  STY $DC05
  LDX #$9B
  LDY #$37
  EOR clock
  ORA initiomap
  BEQ vicinit
  LDX #$1B
  LDY #$00
vicinit: STX $D011
  STY $D012
  LDA initiomap
  BEQ irqinit
  LDA playiomap
  BEQ irqinit
  LDX #<irqjob
  STX $0314
irqinit: LDA speed
  BNE ciainit
  LDA #$81
  STA $D01A
  JMP setiomap
ciainit: LDA #$81
  LDX #$01
  STA $DC0D
  STX $DC0E
setiomap: LDA initiomap
  BNE setbank
  LDA #$37
setbank: STA $01
  LDA flags
  PHA
  LDA playnum
  PLP
  JSR init
  LDA initiomap
  BEQ idle
  LDA playiomap
  BEQ run
  LDA #$37
  STA $01
run: CLI
idle: JMP idle
irqjob: LDA $01
  PHA
  LDA playiomap
  STA $01
  LDA #$00
  JSR play
  PLA
  STA $01
  DEC $D019
irqret: LDA $DC0D
  PLA
  TAY
  PLA
  TAX
  PLA
  RTI
exception: .byte $02
"""

# The KERNAL's RAM vectors $0314-$0333: an RSID's RAM image when libsidplayfp's
# power-on image is unavailable.
KERNAL_VECTORS = (
    (
        0x0314,
        b"".join(
            v.to_bytes(2, "little")
            for v in (0xEA31, 0xFE66, 0xFE47, 0xF34A, 0xF291, 0xF20E, 0xF250, 0xF333)
            + (0xF157, 0xF1CA, 0xF6ED, 0xF13E, 0xF32F, 0xFE66, 0xF4A5, 0xF5ED)
        ),
    ),
)


class C64:
    """The C64 under sidplayfp: jennings on a bus with the VIC-II and two CIAs.

    Mirrors libsidplayfp's ``MOS6510`` cycle tables, RDY stalls and interrupt
    delay; every SID write, raise, entry and RTI is logged as sidtrace logs it.
    """

    # pylint: disable=too-many-instance-attributes,too-many-public-methods

    LAYOUTS = None
    KERNAL = None

    def __init__(self, ntsc: bool, sid_models=("6581",), extra_sids=(), cia_new=False):
        try:
            from jennings.devices.mpu6502 import MPU
        except ImportError as exc:  # pragma: no cover - jennings is a core dependency
            raise EmulatorUnavailable("jennings is required to run a tune") from exc
        if C64.LAYOUTS is None:
            C64.LAYOUTS, C64.KERNAL = _layouts(), kernal_stub()
        self.ntsc = ntsc
        self.sched = Scheduler()
        self.vic = Vic(self.sched, ntsc, self._vic_irq, self._set_ba)
        self.cia1 = Cia(self.sched, self._cia_irq, cia_new)
        self.cia2 = Cia(self.sched, self._cia_nmi, cia_new)
        self.clear_int = Event(self._remove_irq)
        self.chips = [SidReads(model) for model in sid_models]
        self.sid_slots: Dict[int, int] = {}
        for i, addr in enumerate(extra_sids, 1):
            self.sid_slots[addr & 0xFFE0] = i
        self.lastpoke = [bytearray(0x20) for _ in range(1 + len(extra_sids))]
        self.ram = ram_pattern()
        self.view = bytearray(self.ram)
        self.romview = [False] * 16
        self.colour = bytearray(0x400)
        self.port_dir, self.port_data, self.bits, self.io = 0, 0x3F, -1, False
        self.seed = 3686734
        self.rows: List[tuple] = []
        self.asserted = [False, False, False]
        self.irq_cause = _CIA
        self.irq_count = 0
        self.ic, self.pin, self.nmi, self.iflag, self.rdy = (
            _MAX,
            False,
            False,
            False,
            True,
        )
        self.d1x1 = False
        self.pending: List[Tuple[int, int]] = []
        self.last_read = 0
        self.vec_addr = self.target = 0
        self.cross = self.taken = False
        self.ents, self.idx, self.base, self.exec_at, self.rmw = (
            BRK_ENTRIES,
            6,
            0,
            0,
            False,
        )
        self.t, self.cc, self.jammed = 0, 6, False
        self.mpu = MPU(memory=self)
        self._map()
        self.cia1.reset()
        self.cia2.reset()
        self.vic.reset()

    # --- bus -------------------------------------------------------------------

    def _map(self) -> None:
        bits = (self.port_data | ~self.port_dir) & 7
        if bits == self.bits:
            return
        self.bits = bits
        loram, hiram, charen = bits & 1, bits & 2, bits & 4
        ram, view, rom = self.ram, self.view, self.romview
        kernal = bool(hiram)
        basic = bool(loram and hiram)
        self.io = bool(charen and (loram or hiram))
        chars = bool(not charen and (loram or hiram))
        view[0xE000:] = C64.KERNAL if kernal else ram[0xE000:]
        view[0xA000:0xC000] = bytes(0x2000) if basic else ram[0xA000:0xC000]
        view[0xD000:0xE000] = bytes(0x1000) if chars else ram[0xD000:0xE000]
        rom[0xA] = rom[0xB] = basic
        rom[0xD] = chars or self.io
        rom[0xE] = rom[0xF] = kernal

    def _port_read(self, addr: int) -> int:
        if not addr:
            return self.port_dir
        data, direction = self.port_data, self.port_dir
        pins = (0x3F & ~direction) | (data & direction)
        value = (data | ~direction) & (pins | 0x17) & 0xFF
        if not direction & 0x20:
            value &= ~0x20
        return value & ~(~direction & 0xC0)

    def __getitem__(self, addr: int) -> int:
        if addr < 0xD000 or addr >= 0xE000 or not self.io:
            return self.view[addr] if addr > 1 else self._port_read(addr)
        return self._io_read(addr)

    def __setitem__(self, addr: int, value: int) -> None:
        if 0xD000 <= addr < 0xE000 and self.io:
            self.pending.append((addr, value))
            return
        self.ram[addr] = value
        if not self.romview[addr >> 12]:
            self.view[addr] = value
        if addr < 2:
            if addr:
                self.port_data = value
            else:
                self.port_dir = value
            self._map()

    def _sid_chip(self, addr: int) -> Optional[int]:
        chip = self.sid_slots.get(addr & 0xFFE0)
        if chip is None and addr < 0xD800:
            return 0
        return chip

    def _io_read(self, addr: int) -> int:
        page = addr >> 8
        if page < 0xD4:
            value = self.vic.read(addr)
        elif page < 0xD8 or page >= 0xDE:
            chip = self._sid_chip(addr)
            if chip is None:
                self.seed = (self.seed * 1664525 + 1013904223) & 0xFFFFFFFF
                value = self.seed & 0xFF
            else:
                value = self.chips[chip].read(self.sched.time(PHI1), addr & 0x1F)
        elif page < 0xDC:
            value = self.colour[addr & 0x3FF]
        else:
            value = (self.cia1 if page == 0xDC else self.cia2).read(addr)
        self.last_read = value
        return value

    def _io_write(self, addr: int, value: int) -> None:
        page = addr >> 8
        if page < 0xD4:
            self.vic.write(addr, value)
        elif page < 0xD8 or page >= 0xDE:
            chip = self._sid_chip(addr)
            if chip is not None:
                self._sid_write(chip, addr & 0x1F, value)
        elif page < 0xDC:
            self.colour[addr & 0x3FF] = value & 0x0F
        else:
            (self.cia1 if page == 0xDC else self.cia2).write(addr, value)

    def _sid_write(self, chip: int, register: int, value: int) -> None:
        cycle = self.sched.time(PHI1)
        if value != self.lastpoke[chip][register]:
            self.rows.append((cycle, chip, register, value))
            self.lastpoke[chip][register] = value
        sid = self.chips[chip]
        sid.write(cycle, register, value)
        if sid.pending() > 0xFFFF:
            sid.flush(cycle)

    # --- interrupt lines -------------------------------------------------------

    def _mark(self, kind: int) -> None:
        if kind != _NMI and not self.asserted[_VIDEO] and not self.asserted[_CIA]:
            self.irq_cause = kind
        self.asserted[kind] = True
        self.rows.append((self.sched.time(PHI1), None, None, _RAISE[kind]))

    def _irq(self, state: bool, kind: int) -> None:
        if state:
            self._mark(kind)
            if not self.irq_count:
                self.pin = True
                self._calc()
            self.irq_count += 1
        else:
            self.asserted[kind] = False
            other = _VIDEO if kind == _CIA else _CIA
            if self.asserted[other]:
                self.irq_cause = other
            self.irq_count -= 1
            if not self.irq_count:
                self.pin = False
                self.sched.schedule(self.clear_int, 2, PHI1)

    def _vic_irq(self, state: bool) -> None:
        self._irq(state, _VIDEO)

    def _cia_irq(self, state: bool) -> None:
        self._irq(state, _CIA)

    def _cia_nmi(self, state: bool) -> None:
        if state:
            self._mark(_NMI)
            self.nmi = True
            self._calc()

    def _set_ba(self, state: bool) -> None:
        self.rdy = state

    def _check(self) -> bool:
        return self.nmi or (self.pin and not self.iflag)

    def _calc(self) -> None:
        if self.ic == _MAX and self._check():
            self.ic = self.cc

    def _remove_irq(self) -> None:
        if not self.nmi and self.ic != _MAX:
            self.ic = _MAX

    # --- CPU -------------------------------------------------------------------

    def _decode(self) -> None:
        """Fetch the opcode at PC (``fetchNextOpcode``) and lay out its cycles."""
        mpu = self.mpu
        pc = mpu.pc
        op = self[pc]
        self.ents, self.exec_at, kind, self.base, branch = C64.LAYOUTS[op]
        self.idx, self.rmw = 0, kind == "rmw"
        if branch:
            mask, want = branch
            self.taken = bool(mpu.p & mask) == bool(want)
            rel = self[(pc + 1) & 0xFFFF]
            nxt = (pc + 2) & 0xFFFF
            target = (nxt + rel - (0x100 if rel & 0x80 else 0)) & 0xFFFF
            self.cross = (target ^ nxt) & 0xFF00 != 0
        elif kind:
            lo = self[(pc + 1) & 0xFFFF]
            if kind == "indy":
                lo = self[lo]
            self.cross = lo + (mpu.x if kind == "absx" else mpu.y) > 0xFF
        if not self._check():
            self.ic = _MAX
        elif self.ic != _MAX:
            self.ic = -_MAX

    def _fetch(self) -> None:
        """``interruptsAndNextOpcode``: start an interrupt or the next opcode."""
        ic = self.ic
        if ic != _MAX and self.cc > ic + 2:
            self.ents, self.idx, self.base = BRK_ENTRIES, 0, 0
            self.d1x1, self.ic = True, _MAX
        else:
            self._decode()

    def _push(self, value: int) -> None:
        mpu = self.mpu
        self.ram[0x100 | mpu.sp] = value
        self.view[0x100 | mpu.sp] = value
        mpu.sp = (mpu.sp - 1) & 0xFF

    def _stall(self, cc: int) -> None:
        if cc == _CLI_BASE:
            self.iflag = False
            if self.pin and self.ic == _MAX:
                self.ic = -_MAX
        elif cc == _SEI_BASE:
            self.iflag = True
            if not self.nmi and cc <= self.ic + 2:
                self.ic = _MAX
        if self.ic == cc:
            self.ic = cc - 1

    def _special(self, kind: int, t: int) -> None:
        # pylint: disable=too-many-branches
        mpu = self.mpu
        if kind == F:
            self._fetch()
        elif kind == W:
            if self.pending:
                self._io_write(*self.pending.pop(0))
        elif kind == X2:
            if not self.cross:
                self.idx += 1
                self.cc += 1
        elif kind == BR:
            if not self.taken:
                self._fetch()
            elif not self.cross:
                self.idx += 1
                self.cc += 1
                if self.ic >> 3 == self.cc >> 3:
                    self.ic += 2
        elif kind == CLI:
            self.iflag = False
            self._calc()
            self._fetch()
        elif kind == SEI:
            self.iflag = True
            self._fetch()
            if not self.nmi and self.ic != _MAX:
                self.ic = _MAX
        elif kind == POPSR:
            self.iflag = bool(mpu.p & _IRQ_I)
            self._calc()
        elif kind == RTI:
            self.rows.append((t + 1, None, None, "rti"))
            self._fetch()
        elif kind == BRK0:
            if not self.d1x1:
                mpu.pc = (mpu.pc + 2) & 0xFFFF
        elif kind == PUSHPCH:
            self._push(mpu.pc >> 8)
        elif kind == LOWPC:
            self._push(mpu.pc & 0xFF)
            nmi = self.nmi
            self.vec_addr = 0xFFFA if nmi else 0xFFFE
            self.nmi = False
            self._calc()
            name = "brk_entry"
            if nmi:
                name = "nmi_entry"
            elif self.d1x1:
                name = _RAISE[self.irq_cause] + "_entry"
            self.rows.append((t + 1, None, None, name))
        elif kind == PUSHSR:
            self._push((mpu.p & ~0x30) | (0x20 if self.d1x1 else 0x30))
        elif kind == VECLO:
            self.target = self[self.vec_addr]
            self.d1x1 = False
        elif kind == VECHI:
            self.target |= self[self.vec_addr + 1] << 8
            self.iflag = True
            mpu.p |= _IRQ_I
        elif kind == FN:
            mpu.pc = self.target
            self._decode()
        elif kind == HALT:
            self.jammed = True
        elif kind == ANE:
            value = (mpu.a | _ANE_MAGIC) & mpu.x & self[(mpu.pc + 1) & 0xFFFF]
            mpu.a = value
            mpu.p = (mpu.p & 0x7D) | (value & 0x80) | (0 if value else 0x02)
            mpu.pc = (mpu.pc + 2) & 0xFFFF

    def run(self, limit: int) -> None:
        """Run the CPU (and the chips) up to cycle ``limit``."""
        sched, mpu = self.sched, self.mpu
        t = self.t
        while t < limit and not self.jammed:
            t2 = t << 1
            if sched.next <= t2:
                self.cc = self.base + self.idx
                sched.run(t2)
            kind = self.ents[self.idx]
            if kind >= _READS and not self.rdy:
                sched.now = t2 + 1
                self._stall(self.base + self.idx)
                t += 1
                continue
            sched.now = t2 + 1
            self.idx += 1
            self.cc = self.base + self.idx
            if kind == R:
                if self.idx == self.exec_at:
                    mpu.step()
                    if self.rmw and self.pending:
                        self.pending.insert(0, (self.pending[0][0], self.last_read))
            else:
                self.t = t
                self._special(kind, t)
            t += 1
        self.t = t

    # --- boot ------------------------------------------------------------------

    def boot(
        self, image: SidImage, subtune: int, tune_ntsc: Optional[bool], poweron=None
    ) -> int:
        """Install sidplayfp's PSID driver and the tune; reset into the driver.

        ``poweron`` is the ``(address, bytes)`` RAM image an RSID starts from
        (:func:`~pysidtracker.roms.poweron_writes`).
        """
        header = image.header
        rsid = not header.is_psid
        base = _driver_base(image)
        code, labels = assemble(_DRIVER, base)
        ram = self.ram
        ram[:PSID_DRIVER_CLEARED] = bytes(PSID_DRIVER_CLEARED)
        if rsid:
            for addr, data in poweron or KERNAL_VECTORS:
                ram[addr : addr + len(data)] = data
        video = 0 if self.ntsc else 1
        ram[0x02A6] = video
        exception = labels["exception"]
        vectors = (labels["irqret"], exception, exception)
        for i, vector in enumerate(vectors[: 1 if rsid else 3]):
            ram[0x314 + 2 * i : 0x316 + 2 * i] = vector.to_bytes(2, "little")
        ram[0x328:0x32A] = labels["setiomap"].to_bytes(2, "little")
        init = header.init_address or image.load
        speed = 1 if rsid or header.cia_speed(subtune) else 0
        clock = video if tune_ntsc is None else int(not tune_ntsc)
        ram[base : base + len(code)] = code
        ram[base : base + 11] = bytes(
            (subtune & 0xFF, speed)
            + tuple(init.to_bytes(2, "little"))
            + tuple(header.play_address.to_bytes(2, "little"))
            + (_iomap(header, init), _iomap(header, header.play_address))
            + (video, clock, 0 if rsid else _IRQ_I)
        )
        end = image.end
        for addr, word in ((0x2D, end), (0x2F, end), (0x31, end), (0xAC, image.load)):
            ram[addr : addr + 2] = (word & 0xFFFF).to_bytes(2, "little")
        ram[0xAE:0xB0] = (end & 0xFFFF).to_bytes(2, "little")
        ram[image.load : image.end] = image.mem[image.load : image.end]
        self.view[:] = ram
        self.bits = -1
        cycle = boot_cycle(self.ntsc)
        self.sched.run(cycle << 1)
        self.t = cycle
        self[0], self[1] = 0x2F, 0x37
        mpu = self.mpu
        mpu.sp, mpu.p, mpu.a, mpu.x, mpu.y = 0xFF, 0, 0, 0, 0
        self.target = labels["cold"]
        self.ents, self.idx, self.base, self.cc = BRK_ENTRIES, 6, 0, 6
        return cycle


def _resolve_ntsc(image: SidImage, clock: Optional[str]) -> Tuple[bool, Optional[bool]]:
    """``(machine_ntsc, tune_ntsc)``: the header clock picks the machine."""
    tune = {1: False, 2: True}.get((image.header.flags >> 2) & 3)
    if clock is not None:
        key = clock.strip().upper()
        if key not in ("PAL", "NTSC"):
            raise ValueError(f"clock must be 'PAL' or 'NTSC', got {clock!r}")
        return key == "NTSC", tune
    return bool(tune), tune


def to_rows(raw: List[tuple]) -> List[SidtraceRow]:
    """Logged ``(cycle, chip, reg, value)`` tuples as sidtrace :class:`SidtraceRow`."""
    last: Dict[str, int] = {}
    out = []
    for cycle, chip, register, value in raw:
        if chip is None and value in _RAISE:
            last[value] = cycle
        since = [cycle - last[s] if s in last else None for s in _RAISE]
        out.append(SidtraceRow(cycle, *since, chip, register, value))
    return out


class InterruptHost:
    """A PSID or RSID booted and run as sidplayfp runs it, every interrupt live.

    :meth:`rows` are what sidtrace logs for the same tune; :meth:`grid` and
    :meth:`frames` frame them at the play source's raises as the oracle is framed.
    """

    def __init__(
        self,
        data: Union[bytes, bytearray, SidImage],
        subtune: int = 0,
        *,
        clock: Optional[str] = None,
        cia_new: bool = False,
        poweron=None,
    ):
        image = data if isinstance(data, SidImage) else SidImage.from_bytes(bytes(data))
        header = image.header
        if header is None:
            raise SidParseError("cannot run a bare .prg: image has no SID header")
        if header.is_rsid and header.flags & 0x02:
            raise SidParseError("RSID BASIC tunes need the BASIC interpreter")
        ntsc, tune_ntsc = _resolve_ntsc(image, clock)
        extra = [0xD000 | b << 4 for b in (header.second_sid, header.third_sid) if b]
        model = reg.sid_model_for_flags(header.flags)
        models = [model] + [
            {1: "6581", 2: "8580"}.get(header.flags >> shift & 3, model)
            for shift in (6, 8)[: len(extra)]
        ]
        self.machine = C64(ntsc, models, extra, cia_new)
        if poweron is None and header.is_rsid:
            poweron = poweron_writes() or KERNAL_VECTORS
        self.start = self.machine.boot(image, subtune, tune_ntsc, poweron)
        self.clock_hz = reg.NTSC_CLOCK_HZ if ntsc else reg.PAL_CLOCK_HZ
        self.frame_cycles = (
            reg.NTSC_CYCLES_PER_FRAME if ntsc else reg.PAL_CYCLES_PER_FRAME
        )
        self.framed: Optional[Tuple[str, int]] = None

    @property
    def cycle(self) -> int:
        """The machine's current cycle (sidplayfp's clock)."""
        return self.machine.t

    @property
    def jammed(self) -> bool:
        """The CPU hit a JAM opcode; sidplayfp stops there."""
        return self.machine.jammed

    def run(self, cycles: int) -> None:
        """Run ``cycles`` more cycles."""
        self.machine.run(self.machine.t + cycles)

    def rows(self) -> List[SidtraceRow]:
        """Every row logged so far, as sidtrace logs it."""
        return to_rows(self.machine.rows)

    def timing(self, chip: int = 0) -> Tuple[Optional[str], Optional[int]]:
        """``(source, cycles_per_frame)`` the oracle frames the rows so far by."""
        rows = self.rows()
        try:
            return play_source(rows, chip) or _most_raised(rows), sidtrace_cadence(
                rows, chip=chip
            )
        except ValueError:
            return None, None

    def grid(self, nframes: int, *, chip: int = 0) -> List[List[int]]:
        """``nframes`` register rows framed as the survey frames the oracle.

        Runs until the play source's raises cover ``nframes`` frames, then frames
        every row at the median raise gap (``oracle.sidtrace_grid``), forward-filled.
        """
        limit = self.start + 4 * nframes * self.frame_cycles + 4 * self.clock_hz
        self.run(self.clock_hz)
        while not self.jammed and self.cycle < limit:
            source, cpf = self.timing(chip)
            raises = raise_cycles(self.rows(), source) if source else []
            covered = _raise_frames(raises, cpf).get(raises[-1], 0) if cpf else 0
            if covered > nframes:
                break
            self.run(max(nframes + 1 - covered, 1) * (cpf or self.frame_cycles))
        rows = self.rows()
        try:
            cpf = sidtrace_cadence(rows, chip=chip)
            grid = sidtrace_grid(rows, chip=chip, cycles_per_frame=cpf)
        except ValueError as exc:
            raise SidParseError("no interrupt was seen to play the tune") from exc
        self.framed = (play_source(rows, chip) or _most_raised(rows), cpf)
        return [grid[min(i, len(grid) - 1)] for i in range(nframes)]

    def frames(
        self,
        *,
        chip: int = 0,
        source: Optional[str] = None,
        cycles_per_frame: Optional[int] = None,
        probe_seconds: float = 2.0,
    ) -> Iterator[List[int]]:
        """Yield register rows one frame at a time, framed at ``source`` raises.

        ``source``/``cycles_per_frame`` default to the oracle's choice over the
        first ``probe_seconds``; a gap of ``k`` frames between raises repeats a
        row ``k`` times. Once the source stops raising, rows advance one frame each.
        """
        if source is None or cycles_per_frame is None:
            self.run(
                max(0, self.start + int(probe_seconds * self.clock_hz) - self.cycle)
            )
            found, cpf = self.timing(chip)
            if found is None:
                raise SidParseError("no interrupt was seen to play the tune")
            source, cycles_per_frame = source or found, cycles_per_frame or cpf
        return self._frames(chip, source, cycles_per_frame)

    def _frames(self, chip: int, source: str, cpf: int) -> Iterator[List[int]]:
        raw, state = self.machine.rows, [0] * reg.SID_REG_COUNT
        pos, prev = 0, None

        def apply(until: int) -> None:
            nonlocal pos
            while pos < len(raw) and raw[pos][0] < until:
                _, ch, register, value = raw[pos]
                if ch == chip and register < reg.SID_REG_COUNT:
                    state[register] = (
                        value & 0x0F if register in reg.PW_HI_REGS else value
                    )
                pos += 1

        scan = 0
        while True:
            raise_at = None
            waited = 0
            while raise_at is None:
                while scan < len(raw) and raise_at is None:
                    if raw[scan][1] is None and raw[scan][3] == source:
                        raise_at = raw[scan][0]
                    scan += 1
                if raise_at is None:
                    if self.jammed or waited > 50:
                        apply(self.cycle + 1)
                        yield list(state)
                        self.run(cpf)
                        continue
                    self.run(cpf)
                    waited += 1
            if self.cycle <= raise_at:
                self.run(raise_at + 1 - self.cycle)
            apply(raise_at)
            if prev is not None:
                for _ in range(max(1, (raise_at - prev + cpf // 2) // cpf)):
                    yield list(state)
            prev = raise_at


def _most_raised(rows) -> Optional[str]:
    raised = Counter(r.value for r in rows if r.chip is None and r.value in _RAISE)
    return raised.most_common(1)[0][0] if raised else None


class InterruptPlayer(MemPlayer):
    """A :class:`~pysidtracker.player.MemPlayer` over an :class:`InterruptHost`.

    Each frame is one play-source raise, as the oracle frames its grid; a fresh
    player's :meth:`render_grid` is the host's oracle-framed :meth:`InterruptHost.grid`.
    """

    def __init__(
        self,
        data: Union[bytes, bytearray, SidImage],
        subtune: int = 0,
        *,
        clock: Optional[str] = None,
    ):
        self._data, self._clock = data, clock
        self.host: Optional[InterruptHost] = None
        self._iter: Optional[Iterator[List[int]]] = None
        self._row = [0] * reg.SID_REG_COUNT
        self._played = 0
        super().__init__(b"", 0, subtune)

    def _init(self, subtune: int) -> None:
        self.host = InterruptHost(self._data, subtune, clock=self._clock)

    def _frame(self) -> None:
        if self._iter is None:
            source, cpf = self.host.framed or (None, None)
            self._iter = self.host.frames(source=source, cycles_per_frame=cpf)
            for _ in range(self._played):
                next(self._iter)
        self._row = next(self._iter)
        self._played += 1

    def snapshot(self) -> List[int]:
        return list(self._row)

    def render_grid(self, nframes: int) -> List[List[int]]:
        if self._played:
            return super().render_grid(nframes)
        grid = self.host.grid(nframes)
        self._played = nframes
        self._row = list(grid[-1])
        self.regs = list(self._row)
        return grid


def host_cadence(
    image_or_bytes: Union[SidImage, bytes, bytearray],
    *,
    clock: Optional[str] = None,
    subtune: int = 0,
    seconds: float = 10.0,
) -> Cadence:
    """The :class:`~pysidtracker.cadence.Cadence` sidplayfp plays the tune at.

    Runs the host ``seconds`` and measures the play source (most distinct SID
    registers, raised at least 3 times) over its raises after the first.
    """
    host = InterruptHost(image_or_bytes, subtune, clock=clock)
    host.run(int(seconds * host.clock_hz))
    rows = host.rows()
    frame, hz = host.frame_cycles, host.clock_hz
    source = play_source(rows, min_raises=3) if nested(rows) else None
    steady = raise_cycles(rows, source)[1:] if source else []
    if len(steady) < 2:
        return Cadence(frame, None, hz)
    gaps = [b - a for a, b in zip(steady, steady[1:])]
    span = steady[-1] - steady[0]
    trigger = {
        "video_irq": (
            TriggerSource.NTSC_VIDEO
            if frame == reg.NTSC_CYCLES_PER_FRAME
            else TriggerSource.PAL_VIDEO
        ),
        "cia_irq": TriggerSource.CIA_TIMER,
        "nmi": TriggerSource.NMI_TIMER,
    }[source]
    dynamic = source != "video_irq" and len(set(gaps)) > 1
    per_frame = frame * len(gaps) / span
    return Cadence(round(span / len(gaps)), trigger, hz, None, dynamic, per_frame)


__all__ = [
    "C64",
    "KERNAL_VECTORS",
    "InterruptHost",
    "InterruptPlayer",
    "host_cadence",
    "opcode_layout",
    "kernal_stub",
    "ram_pattern",
    "boot_cycle",
    "to_rows",
]
