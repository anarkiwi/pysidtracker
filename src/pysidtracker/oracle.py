"""Ground-truth per-frame SID register grids for byte-exact player validation.

Two ways to build the reference grid every format package validates its player
against, plus the readers/aligner around them:

* :func:`register_grid` -- run a tune's ``init`` then ``play`` on a py65 6502
  (reusing :mod:`pysidtracker.trace`'s run-to-RTS mechanic) and sample the 25
  SID registers ``$D400..$D418`` per frame. Requires py65 (a core dependency).
* :func:`grid_from_writes` -- the pure-stdlib framer that turns a
  ``(clock, reg, val)`` write stream (e.g. a ``preframr-sidtrace`` capture read
  by :func:`read_sidwr`, or a :mod:`pysidtracker.reglog` log) into the same
  per-frame grid: anchor frame 0 to the first play call, forward-fill, and
  nibble-mask the pulse-width-high registers.

:func:`aligned_match` compares a rendered grid to an oracle grid, tolerating a
few leading silent frames. This consolidates the py65 oracle (pydefmon,
pyjch) and the sidtrace framer (pyjch/pymusicassembler/pyfuturecomposer
conftests, pydmcsid helpers).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from . import registers as reg
from .errors import EmulatorUnavailable, SidParseError
from .image import SidImage
from .trace import _run_to_rts

# preframr-sidtrace ``.sidwr.bin`` record: (clock, addr, reg, val).
_SIDWR_REC = struct.Struct("<qHBB")


def _patch_illegals(mpu) -> None:
    """Install the NMOS illegal opcodes some replays (e.g. defMON) execute.

    Implemented from documented NMOS 6502 behaviour: SBX/ANC/ALR/ARR/SBC/LAX/SAX
    plus the multi-byte NOP illegals a data-adjacent code stream can drift
    through.
    """
    mpu.instruct = list(mpu.instruct)
    mpu.cycletime = list(mpu.cycletime)
    mpu.extracycles = list(mpu.extracycles)

    def _set(op, fn, cyc=2):
        mpu.instruct[op] = fn
        mpu.cycletime[op] = cyc
        mpu.extracycles[op] = 0

    def i_sbx(self):  # SBX/AXS #imm: X = (A & X) - imm, CMP-style carry
        v = self.ByteAt(self.ProgramCounter())
        t = (self.a & self.x) - v
        self.x = t & 0xFF
        self.p &= ~(self.CARRY | self.ZERO | self.NEGATIVE)
        if t >= 0:
            self.p |= self.CARRY
        self.FlagsNZ(self.x)
        self.pc += 1

    def i_anc(self):  # ANC #imm: A &= imm; C = bit7
        self.a &= self.ByteAt(self.ProgramCounter())
        self.FlagsNZ(self.a)
        self.p = (self.p & ~self.CARRY) | (1 if self.a & 0x80 else 0)
        self.pc += 1

    def i_alr(self):  # ALR #imm: A = (A & imm) >> 1
        self.a &= self.ByteAt(self.ProgramCounter())
        self.p = (self.p & ~self.CARRY) | (self.a & 1)
        self.a >>= 1
        self.FlagsNZ(self.a)
        self.pc += 1

    def i_arr(self):  # ARR #imm
        self.a &= self.ByteAt(self.ProgramCounter())
        c = 1 if self.p & self.CARRY else 0
        self.a = (self.a >> 1) | (c << 7)
        self.FlagsNZ(self.a)
        self.p &= ~(self.CARRY | self.OVERFLOW)
        if self.a & 0x40:
            self.p |= self.CARRY
        if bool(self.a & 0x40) ^ bool(self.a & 0x20):
            self.p |= self.OVERFLOW
        self.pc += 1

    def i_sbc(self):  # SBC #imm alias ($EB)
        self.opSBC(self.ProgramCounter)
        self.pc += 1

    def i_lax_imm(self):  # LAX #imm -> A = X = imm
        v = self.ByteAt(self.ProgramCounter())
        self.a = self.x = v
        self.FlagsNZ(v)
        self.pc += 1

    def _sax(meth, pcadd):  # SAX: store A & X
        def f(self):
            self.memory[getattr(self, meth)()] = self.a & self.x
            self.pc += pcadd

        return f

    def _lax(meth, pcadd):  # LAX: A = X = mem
        def f(self):
            v = self.ByteAt(getattr(self, meth)())
            self.a = self.x = v
            self.FlagsNZ(v)
            self.pc += pcadd

        return f

    _set(0xCB, i_sbx)
    _set(0x0B, i_anc)
    _set(0x2B, i_anc)
    _set(0x4B, i_alr)
    _set(0x6B, i_arr)
    _set(0xEB, i_sbc)
    _set(0xAB, i_lax_imm)
    _set(0x83, _sax("IndirectXAddr", 1), 6)
    _set(0x87, _sax("ZeroPageAddr", 1), 3)
    _set(0x8F, _sax("AbsoluteAddr", 2), 4)
    _set(0x97, _sax("ZeroPageYAddr", 1), 4)
    _set(0xA3, _lax("IndirectXAddr", 1), 6)
    _set(0xA7, _lax("ZeroPageAddr", 1), 3)
    _set(0xAF, _lax("AbsoluteAddr", 2), 4)
    _set(0xB3, _lax("IndirectYAddr", 1), 5)
    _set(0xB7, _lax("ZeroPageYAddr", 1), 4)
    _set(0xBF, _lax("AbsoluteYAddr", 2), 4)
    for op in (0x1A, 0x3A, 0x5A, 0x7A, 0xDA, 0xFA):
        _set(op, lambda s: setattr(s, "pc", s.pc), 2)
    for op in (0x80, 0x82, 0x89, 0xC2, 0xE2):
        _set(op, lambda s: setattr(s, "pc", s.pc + 1), 2)
    for op in (0x04, 0x44, 0x64):
        _set(op, lambda s: setattr(s, "pc", s.pc + 1), 3)
    for op in (0x14, 0x34, 0x54, 0x74, 0xD4, 0xF4):
        _set(op, lambda s: setattr(s, "pc", s.pc + 1), 4)
    _set(0x0C, lambda s: setattr(s, "pc", s.pc + 2), 4)
    for op in (0x1C, 0x3C, 0x5C, 0x7C, 0xDC, 0xFC):
        _set(op, lambda s: setattr(s, "pc", s.pc + 2), 4)


def register_grid(
    image_or_bytes,
    nframes: int,
    *,
    subtune: int = 0,
    illegal_opcodes: bool = False,
    max_cycles: int = 8_000_000,
) -> List[List[int]]:
    """Per-frame SID register grid from running a tune on py65.

    ``image_or_bytes`` is a :class:`~pysidtracker.image.SidImage` or PSID/RSID
    (or ``.prg``) bytes. Runs ``init`` (accumulator = ``subtune``), then
    ``nframes`` ``play`` calls, sampling ``$D400..$D418`` (25 registers) after
    each play. ``illegal_opcodes=True`` installs the NMOS illegal opcodes that
    replays such as defMON need (default off, so other callers are unaffected).

    Requires py65; raises :class:`~pysidtracker.errors.EmulatorUnavailable` if
    it is missing and :class:`~pysidtracker.errors.SidParseError` if the image
    has no init address.
    """
    if isinstance(image_or_bytes, (bytes, bytearray)):
        image = SidImage.from_bytes(bytes(image_or_bytes))
    else:
        image = image_or_bytes
    if image.header is None:
        raise SidParseError("cannot build a register grid: image has no SID header")
    try:
        from py65.devices.mpu6502 import MPU
        from py65.memory import ObservableMemory
    except ImportError as exc:  # pragma: no cover - py65 is a core dependency
        raise EmulatorUnavailable(
            "py65 is required for register_grid: pip install pysidtracker[emu]"
        ) from exc

    subject = image.mem
    mem = ObservableMemory(subject=subject)

    def _on_raster(addr):
        line = (mpu.processorCycles // 63) % 312
        if addr == reg.VIC_RASTER:
            return line & 0xFF
        return (subject[reg.VIC_CONTROL_1] & 0x7F) | (((line >> 8) & 1) << 7)

    def _on_sidread(addr):  # pylint: disable=unused-argument
        return (mpu.processorCycles >> 3) & 0xFF

    mem.subscribe_to_read([reg.VIC_CONTROL_1, reg.VIC_RASTER], _on_raster)
    mem.subscribe_to_read([0xD41B, 0xD41C], _on_sidread)
    mpu = MPU(memory=mem)
    if illegal_opcodes:
        _patch_illegals(mpu)

    init_address = image.header.init_address or image.header.real_load_address
    _run_to_rts(mpu, mem, init_address, subtune, max_cycles)

    play_address = image.header.play_address or init_address
    rows: List[List[int]] = []
    for _ in range(nframes):
        _run_to_rts(mpu, mem, play_address, 0, max_cycles)
        rows.append([subject[reg.SID_BASE + i] for i in range(reg.SID_REG_COUNT)])
    return rows


def grid_from_writes(
    writes: Sequence[Tuple[int, int, int]],
    *,
    cycles_per_frame: int = reg.PAL_CYCLES_PER_FRAME,
    reg_count: int = 25,
    pw_hi_regs: Iterable[int] = reg.PW_HI_REGS,
    gap: int = 10000,
) -> List[List[int]]:
    """Frame a ``(clock, reg, val)`` write stream into a per-frame grid.

    Frame 0 is anchored to the first play call -- the first write after a
    ``> gap``-cycle gap; the leading init writes form frame 0's baseline. Each
    frame's registers forward-fill from the previous frame, and the
    pulse-width-high registers (``pw_hi_regs``) are masked to 4 bits. Frame
    assignment rounds to nearest (``(clock - t0 + cpf // 2) // cpf``).
    """
    if not writes:
        return []
    pw = set(pw_hi_regs)
    cyc = [w[0] for w in writes]
    t0 = cyc[0]
    for prev, cur in zip(cyc, cyc[1:]):
        if cur - prev > gap:
            t0 = cur
            break
    cur_row = [0] * reg_count
    rows: List[List[int]] = []
    idx = 0
    while idx < len(writes) and writes[idx][0] < t0:
        _c, register, val = writes[idx]
        if 0 <= register < reg_count:
            cur_row[register] = (val & 0x0F) if register in pw else val
        idx += 1

    def frame_of(clock: int) -> int:
        return (clock - t0 + cycles_per_frame // 2) // cycles_per_frame

    nframes = frame_of(writes[-1][0]) + 1
    for frame in range(nframes):
        while idx < len(writes) and frame_of(writes[idx][0]) == frame:
            _c, register, val = writes[idx]
            if 0 <= register < reg_count:
                cur_row[register] = (val & 0x0F) if register in pw else val
            idx += 1
        rows.append(cur_row[:])
    return rows


def read_sidwr(path, *, reg_count: int = 25) -> List[Tuple[int, int, int]]:
    """Read a ``preframr-sidtrace`` ``.sidwr.bin`` into ``(clock, reg, val)``.

    Each fixed-size record is ``struct.Struct("<qHBB")`` = (clock, addr, reg,
    val); the addr field is dropped and records whose ``reg >= reg_count`` are
    skipped, matching the existing per-repo readers.
    """
    blob = Path(path).read_bytes()
    out: List[Tuple[int, int, int]] = []
    for off in range(0, len(blob) - _SIDWR_REC.size + 1, _SIDWR_REC.size):
        clock, _addr, register, val = _SIDWR_REC.unpack_from(blob, off)
        if register < reg_count:
            out.append((clock, register, val))
    return out


def aligned_match(
    oracle: Sequence[Sequence[int]],
    rendered: Sequence[Sequence[int]],
    *,
    max_lead: int = 4,
) -> bool:
    """True if ``rendered`` matches ``oracle`` within ``max_lead`` silent frames.

    Tries aligning ``oracle`` at each lead offset ``0..max_lead`` for which the
    skipped leading frames of ``rendered`` are all equal to its first frame
    (silent lead-in), returning True on the first exact match.
    """
    if not rendered:
        return False
    baseline = rendered[0]
    for lead in range(max_lead + 1):
        if lead and (lead > len(rendered) or rendered[lead - 1] != baseline):
            break
        aligned = rendered[lead : lead + len(oracle)]
        if len(aligned) < len(oracle):
            continue
        if all(oracle[f] == aligned[f] for f in range(len(oracle))):
            return True
    return False
