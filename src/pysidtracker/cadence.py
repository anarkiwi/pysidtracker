"""Derive a tune's play cadence from how its interrupts are actually programmed.

:func:`playroutine_cadence` seeds the state sidplayfp's PSID driver leaves, then
runs the tune on jennings with VIC raster and CIA timer interrupts raised and
taken as the C64 would; see ``docs/cadence.md`` for the driver and the model.
"""

from __future__ import annotations

import enum
import zlib
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from vagrant_moses import memmap as mm
from vagrant_moses.c64 import dispatch_vector

from . import registers as reg
from .emu import access_cycle, live_ram, wire_mpu
from .errors import EmulatorUnavailable, SidParseError
from .header import RSID_MAGIC
from .image import SidImage
from .poweron import psid_init_bank

# A CIA Timer-A latch below this is a lo-byte-only artefact, not a play period.
_MIN_CIA_LATCH = 0x100
# psiddrv.a65 ``irqjob``/``irqret`` offsets from the relocated driver's first byte.
_IRQJOB, _IRQRET = 0xA8, 0xBB
_RET, _EXCEPTION = 0xA0, 0xC4
# sidplayfp's ROM-less KERNAL slides NOPs up to `NOP $DC0D` at $EA7E (an ack).
_KERNAL_ACK, _KERNAL_NMI = 0xEA7E, 0xFE47
_BRK, _RTI, _I_FLAG = 0x00, 0x40, 0x04
_VIC = (-1, 0)
# Opcodes that can end a masked IRQ's wait (CLI, PLP, RTI) or spin on one address.
_UNMASK = (0x58, 0x28, 0x40)
_JMP, _JMP_IND = 0x4C, 0x6C
_SELF_BRANCH = (0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0)
_KERNAL_PCS = frozenset(range(mm.KERNAL_BASE, 0x10000))
_RMW = frozenset(
    (0x0E, 0x1E, 0x2E, 0x3E, 0x4E, 0x5E, 0x6E, 0x7E, 0xCE, 0xDE, 0xEE, 0xFE)
)


class TriggerSource(enum.Enum):
    """What triggers each play-routine call."""

    PAL_VIDEO = "pal_video"  # PAL raster interrupt
    NTSC_VIDEO = "ntsc_video"  # NTSC raster interrupt
    CIA_TIMER = "cia_timer"  # a CIA #1 timer IRQ
    NMI_TIMER = "nmi_timer"  # a CIA #2 timer NMI


@dataclass(frozen=True)
class Cadence:
    """The play-routine cadence of a tune.

    Attributes:
      cycles_per_call: mean CPU cycles between consecutive play calls.
      source: the :class:`TriggerSource` calling play, ``None`` when no
        interrupt was seen to play (``cycles_per_call`` is then one frame).
      clock_hz: the CPU clock (``PAL_CLOCK_HZ`` / ``NTSC_CLOCK_HZ``).
      latch: the timer latch when timer-driven, else ``None``.
      dynamic: the timer period changed while it was measured.
      calls_per_frame: play calls per video frame (``k`` for a raster chain
        over ``k`` lines); derived from ``cycles_per_call`` when omitted.
      raster_lines: the raster compare lines that called play, if raster-driven.
    """

    cycles_per_call: int
    source: Optional[TriggerSource]
    clock_hz: int
    latch: Optional[int] = None
    dynamic: bool = False
    calls_per_frame: Optional[float] = None
    raster_lines: Tuple[int, ...] = ()

    def __post_init__(self):
        if self.calls_per_frame is None:
            per_frame = self.frame_cycles / self.cycles_per_call
            object.__setattr__(self, "calls_per_frame", per_frame)

    @property
    def frame_cycles(self) -> int:
        """CPU cycles per video frame at ``clock_hz``."""
        return mm.video(self.clock_hz == mm.NTSC.clock_hz).cycles_per_frame

    @property
    def calls_per_second(self) -> float:
        """The play-call rate in Hz."""
        return self.calls_per_frame * self.clock_hz / self.frame_cycles


def _resolve_standard(image: SidImage, clock: Optional[str]) -> bool:
    """True for NTSC: an explicit ``clock``, else the header clock bits, else PAL."""
    if clock is not None:
        key = clock.strip().upper()
        if key not in ("PAL", "NTSC"):
            raise ValueError(f"clock must be 'PAL' or 'NTSC', got {clock!r}")
        return key == "NTSC"
    return image.header is not None and (image.header.flags >> 2) & 3 == 0b10


def _iomap(header, addr: int) -> int:
    """psiddrv's ``$01`` bank value for code at ``addr`` (0: real C64 environment)."""
    if not header.is_psid or addr == 0:
        return 0
    return psid_init_bank(addr)


def _driver_base(image: SidImage) -> int:
    """The page psiddrv relocates itself to.

    The header's free page (PSID v2+), else the first page outside the tune and
    BASIC ROM.
    """
    first, last = image.load >> 8, (image.end - 1) >> 8
    page = image.container[0x78] if image.header.version >= 2 else 0
    if page in (0, 0xFF):
        page = next(
            p
            for p in range(4, 0xD0)
            if not first <= p <= last and not 0xA0 <= p <= 0xBF
        )
    return page << 8


def _kernal_exit(pc: int) -> int:
    """The ``irqret`` offset matching where a jump into the ROM-less KERNAL ends.

    Up to ``$EA7E`` it acknowledges CIA #1, pulls and returns (``irqret``),
    ``$FE47`` saves then restores the registers (``RTI``), else it pulls and returns.
    """
    if pc <= _KERNAL_ACK:
        return 0
    return 8 if pc == _KERNAL_NMI else 3


def _word(value: int) -> bytes:
    return bytes((value & 0xFF, value >> 8))


class _Cia:
    """One CIA's two timers, interrupt data and mask (reset: ``$FFFF``, stopped)."""

    def __init__(self):
        self.latch = [0xFFFF, 0xFFFF]
        self.cr = [0, 0]
        self.mask = 0
        self.flags = 0
        self.due = [0, 0]

    def period(self, timer: int) -> int:
        """Cycles between underflows; Timer B may count Timer A underflows."""
        if timer and self.cr[1] & 0x40:
            return (self.latch[1] + 1) * (self.latch[0] + 1)
        return self.latch[timer] + 1

    def armed(self, timer: int) -> bool:
        """The timer runs and its underflow raises the interrupt."""
        return bool(self.cr[timer] & 1 and self.mask >> timer & 1)

    def write(self, offset: int, value: int, now: int) -> None:
        """Apply a register write at cycle ``now``."""
        if 4 <= offset <= 7:
            timer, shift = (offset - 4) >> 1, 8 * (offset & 1)
            self.latch[timer] = self.latch[timer] & ~(0xFF << shift) | value << shift
        elif offset == 0xD:
            bits = value & 0x1F
            self.mask = self.mask | bits if value & 0x80 else self.mask & ~bits
        elif offset in (0xE, 0xF):
            timer = offset - 0xE
            started = value & 1 and not self.cr[timer] & 1
            self.cr[timer] = value & 0xEF
            if started or value & 0x10:
                self.due[timer] = now + self.period(timer)

    def underflow(self, timer: int, now: int) -> bool:
        """Underflow due by ``now``; True if it newly asserts the interrupt line."""
        asserted = self.flags & self.mask
        self.flags |= 1 << timer
        if self.cr[timer] & 8:
            self.cr[timer] &= 0xFE
        else:
            period = self.period(timer)
            self.due[timer] += period * ((now - self.due[timer]) // period + 1)
        return not asserted and bool(self.flags & self.mask)

    def counter(self, timer: int, now: int) -> int:
        """The timer's count at ``now``: down from the latch to 0, then underflow."""
        if not self.cr[timer] & 1:
            return self.latch[timer]
        return (self.due[timer] - now - 1) % (self.latch[timer] + 1)

    def read_icr(self, now: int) -> int:
        """Read and clear the interrupt data register."""
        for timer in (0, 1):
            if self.cr[timer] & 1 and self.due[timer] <= now:
                self.underflow(timer, now)
        flags, self.flags = self.flags, 0
        return flags | (0x80 if flags & self.mask else 0)


class _Machine:
    """A tune on jennings in psiddrv's environment, raising and taking interrupts."""

    def __init__(
        self,
        image: SidImage,
        ntsc: bool,
        max_cycles: int,
        native: Optional[bool] = None,
    ):
        header = image.header
        ram = bytearray(image.mem)
        self.mpu, self.mem = wire_mpu(
            ram,
            sid_model=reg.sid_model_for_flags(header.flags),
            ntsc=ntsc,
            native=native,
        )
        self.ram = live_ram(self.mpu, ram)
        self.bus = self.mem._subject  # pylint: disable=protected-access
        video = mm.video(ntsc)
        self.frame, self.line = video.cycles_per_frame, video.cycles_per_line
        self.max_cycles = max_cycles
        self.cias = (_Cia(), _Cia())
        self.vic_enable = self.raster = self.vic_seen = 0
        self.vic_flag = False
        self.events = []
        self.last = {}
        self.handlers = [None]
        self.frames = [0x1FF]
        self.opaque = set()
        self.dirty = True
        extra = [0xD000 | b << 4 for b in (header.second_sid, header.third_sid) if b]
        self.mem.subscribe_to_write(
            [*range(mm.SID_BASE, mm.SID_LAST + 1)]
            + [a + o for a in extra for o in range(reg.SID_REG_COUNT)],
            self._sid,
        )
        self.mem.subscribe_to_write(
            [mm.VIC_CONTROL_1, mm.VIC_RASTER, mm.VIC_IRQ_STATUS, mm.VIC_IRQ_ENABLE]
            + [base + o for base in (mm.CIA1_BASE, mm.CIA2_BASE) for o in range(16)],
            self._write,
        )
        self.mem.subscribe_to_read(
            [mm.VIC_CONTROL_1, mm.VIC_RASTER, mm.VIC_IRQ_STATUS]
            + [mm.CIA1_ICR, mm.CIA2_ICR, mm.SID_BASE + 0x1B, mm.SID_BASE + 0x1C]
            + [
                base + o
                for base in (mm.CIA1_BASE, mm.CIA2_BASE)
                for o in (4, 5, 6, 7, 14, 15)
            ],
            self._read,
        )
        self.brk = False
        if self.mpu.native:
            self.mem.subscribe_to_write((mm.CPU_DDR, mm.CPU_PORT), self._rebank)
            self.mem.subscribe_to_read((mm.CPU_IRQ_VECTOR,), self._vector)
        self.passthrough = (mm.VIC_CONTROL_1, mm.VIC_RASTER)
        self.passthrough += (mm.SID_BASE + 0x1B, mm.SID_BASE + 0x1C)
        self.span = slice(image.load, image.end)
        self.base = _driver_base(image)

    def _sid(self, addr, _value):
        if self.handlers[-1] and self.bus.io:
            self.handlers[-1][2] |= 1 << (addr & 0x1F)

    def _write(self, addr, value):
        self.dirty = self.mpu.stop = True
        if not self.bus.io:
            return
        if addr == mm.VIC_IRQ_ENABLE:
            self.vic_enable = value & 0x0F
        elif addr == mm.VIC_IRQ_STATUS and (value | self._rmw()) & 1:
            self.vic_flag, self.vic_seen = False, access_cycle(self.mpu)
        elif addr == mm.VIC_RASTER:
            self.raster = self.raster & 0x100 | value
        elif addr == mm.VIC_CONTROL_1:
            self.raster = self.raster & 0xFF | (value & 0x80) << 1
        elif mm.CIA1_BASE <= addr <= mm.CIA2_LAST:
            cia = self.cias[addr >= mm.CIA2_BASE]
            cia.write(addr & 0xF, value, access_cycle(self.mpu))

    def _rebank(self, _addr, _value):
        """A ``$00``/``$01`` write: the KERNAL window may have moved."""
        self.mpu.stop = True

    def _vector(self, _addr):
        """The executing instruction read the IRQ vector: a BRK, if it is one."""
        if self.ram[self.mpu.pc] == _BRK:
            self.brk = self.mpu.stop = True

    def _rmw(self) -> int:
        """1 if the executing instruction is a read-modify-write, whose dummy write
        stores the value it read before the modified one."""
        return int(self.ram[self.mpu.pc] in _RMW)

    def _read(self, addr):
        if not self.bus.io:
            return self.ram[addr]
        now = access_cycle(self.mpu)
        if addr in self.passthrough:
            return None
        if addr == mm.VIC_IRQ_STATUS:
            match = self._next_match(now - self.frame)
            flag = self.vic_flag or self.vic_seen < match <= now
            return flag | (0x80 if flag and self.vic_enable & 1 else 0)
        cia = self.cias[addr >= mm.CIA2_BASE]
        offset = addr & 0xF
        if offset >= 0xE:
            return cia.cr[offset - 0xE]
        if offset != 0xD:
            return cia.counter(offset - 4 >> 1, now) >> 8 * (offset & 1) & 0xFF
        self.dirty = self.mpu.stop = True
        return cia.read_icr(now)

    def _next_match(self, now: int) -> int:
        base = self.raster * self.line
        return base - (base - now - 1) // self.frame * self.frame

    def driver(self, header, subtune: int, ntsc: bool) -> None:
        """Seed psiddrv's cold-start state for ``subtune`` and install its stubs."""
        base, ram = self.base, self.ram
        init = header.init_address or header.real_load_address
        initiomap, playiomap = _iomap(header, init), _iomap(header, header.play_address)
        clock_bits = (header.flags >> 2) & 3
        tune_ntsc = ntsc if clock_bits in (0, 3) else clock_bits == 2
        tune = bytes(ram[self.span])
        ram[0:0x400] = bytes(0x400)
        ram[0x02A6] = 0 if ntsc else 1
        ram[mm.RAM_IRQ_VECTOR_LO : mm.RAM_NMI_VECTOR_HI + 1] = (
            _word(base + _IRQRET) + _word(base + _EXCEPTION) * 2
            if header.is_psid
            else _word(base + _IRQRET) + _word(0xFE66) + _word(_KERNAL_NMI)
        )
        ram[self.span] = tune
        self.mem[mm.CPU_DDR], self.mem[mm.CPU_PORT] = mm.DDR_DEFAULT, initiomap or 0x37
        if initiomap and playiomap:
            ram[mm.RAM_IRQ_VECTOR_LO] = base + _IRQJOB & 0xFF
        kernal = mm.video(ntsc).kernal_latch
        compat = tune_ntsc != ntsc or initiomap
        cia_speed = not header.is_psid or header.cia_speed(subtune)
        for addr, value in (
            (mm.VIC_IRQ_ENABLE, 0x00),
            (mm.CIA1_ICR, 0x7F),
            (mm.CIA2_ICR, 0x7F),
            (mm.SID_BASE + 0x18, 0x0F),
            (mm.CIA1_TIMER_A_LO, kernal & 0xFF),
            (mm.CIA1_TIMER_A_HI, kernal >> 8),
            (mm.VIC_CONTROL_1, 0x1B if compat else 0x9B),
            (mm.VIC_RASTER, 0x00 if compat else 0x37),
            *(
                ((mm.CIA1_ICR, 0x81), (mm.CIA1_CRA, 0x01))
                if cia_speed
                else ((mm.VIC_IRQ_ENABLE, 0x81),)
            ),
        ):
            self.mem[addr] = value
        ram[base + 7] = playiomap
        idle = base + _RET + 5
        ram[base + _RET : idle + 3] = (
            (bytes((0xA9, 0x37, 0x85, 0x01)) if playiomap else bytes((0xEA,) * 4))
            + bytes((0x58 if initiomap else 0xEA, 0x4C))
            + _word(idle)
        )
        ram[base + _IRQJOB : base + _EXCEPTION + 1] = (
            bytes((0xA5, 0x01, 0x48, 0xAD))
            + _word(base + 7)
            + bytes((0x85, 0x01, 0xA9, 0x00, 0x20))
            + _word(header.play_address)
            + bytes((0x68, 0x85, 0x01, 0xCE, 0x19, 0xD0, 0xAD, 0x0D, 0xDC))
            + bytes((0x68, 0xA8, 0x68, 0xAA, 0x68, _RTI, 0x02))
        )
        mpu = self.mpu
        mpu.sp, mpu.a = 0xFB, subtune
        ram[0x01FC:0x01FE] = _word(base + _RET - 1)
        mpu.p = mpu.p & ~_I_FLAG | (_I_FLAG if initiomap else 0)
        mpu.pc = init

    def pending(self, now: int):
        """``(cycle, source)`` of the earliest armed source's next raise."""
        due = [
            (max(cia.due[t], now), (chip, t))
            for chip, cia in enumerate(self.cias)
            for t in (0, 1)
            if cia.armed(t)
        ]
        if self.vic_enable & 1 and self.raster < self.frame // self.line:
            due.append((max(self._next_match(now - 1), now), _VIC))
        return min(due, default=None)

    def raise_(self, when: int, source) -> bool:
        """Raise ``source`` at ``when``; log it if it newly asserts its line."""
        if source == _VIC:
            raised, self.vic_flag, self.vic_seen = not self.vic_flag, True, when
            tag = self.raster
        else:
            cia = self.cias[source[0]]
            tag = cia.period(source[1])
            raised = cia.underflow(source[1], when)
        if raised:
            self.events.append([when, source, 0, tag])
            self.last[source[0]] = self.events[-1]
        return raised

    def irq_line(self) -> bool:
        """The CPU IRQ input: a pending raster IRQ or CIA #1 interrupt."""
        cia = self.cias[0]
        return bool(self.vic_flag and self.vic_enable & 1 or cia.flags & cia.mask)

    def entered(self, nmi: bool):
        """The raise an interrupt entry serves.

        That is the NMI's, else the earliest raise still asserting the IRQ line,
        as sidtrace's entry rows name it.
        """
        if nmi:
            return self.last.get(1)
        cia = self.cias[0]
        lines = (
            (_VIC[0], self.vic_flag and self.vic_enable & 1),
            (0, cia.flags & cia.mask),
        )
        held = [self.last[k] for k, on in lines if on and k in self.last]
        return min(held, key=lambda event: event[0], default=None)

    def take(self, nmi: bool) -> None:
        """Enter an interrupt through the vector the banked-in ROM or RAM gives.

        A handler vectored into I/O executes live register values, which only
        cycle-exact timing reproduces; its source is marked ``opaque``.
        """
        mpu, ram = self.mpu, self.ram
        self.handlers.append(self.entered(nmi))
        mpu.nmi()
        self.frames.append(mpu.sp)
        vector, saves = dispatch_vector(self.bus.bits, nmi)
        if saves:
            for value in (mpu.a, mpu.x, mpu.y):
                ram[0x0100 + mpu.sp] = value
                mpu.sp = mpu.sp - 1 & 0xFF
        mpu.pc = ram[vector] | ram[vector + 1] << 8
        if self.handlers[-1] and self.bus.io and mm.VIC_BASE <= mpu.pc < mm.KERNAL_BASE:
            self.opaque.add(self.handlers[-1][1])

    def execute(self, frames: int) -> int:
        """Run until ``frames`` frames after init first idles; return the last cycle.

        A native MPU runs to the next event (:meth:`execute_events`).
        """
        if self.mpu.native:
            return self.execute_events(frames)
        return self.execute_stepped(frames)

    def _watch(self, masked: bool) -> frozenset:
        """Addresses the native run stops before: possible spins, and while an
        asserted IRQ is masked, the instructions that can unmask it."""
        ram = np.frombuffer(self.ram, np.uint8)
        nxt, last = np.roll(ram, -1).astype(np.uint16), np.roll(ram, -2)
        here = np.arange(0x10000, dtype=np.uint16)
        spin = (ram == _JMP) & (nxt | last.astype(np.uint16) << 8 == here)
        spin |= np.isin(ram, _SELF_BRANCH) & (nxt == 0xFE)
        spin |= ram == _JMP_IND
        if masked:
            spin |= np.isin(ram, _UNMASK)
        return frozenset(np.flatnonzero(spin).tolist())

    def _unwind(self) -> None:
        """Leave the handlers whose frames the stack pointer has climbed past."""
        mpu, handlers, frames = self.mpu, self.handlers, self.frames
        while len(handlers) > 1 and mpu.sp > frames[-1]:
            handlers.pop()
            frames.pop()

    def execute_events(self, frames: int) -> int:
        """:meth:`execute_stepped` in native runs between the events that need Python.

        An event is a source falling due, a callback marking the interrupt state
        ``dirty`` (or a ``$00``/``$01`` write, or a BRK), the stack climbing out of
        a handler's frame, or the next instruction being one the reference loop
        examines (a possible spin, a possible unmask of an asserted IRQ, the ROM-less
        KERNAL in a handler): that one instruction is stepped from Python.
        """
        mpu, ram, bus = self.mpu, self.ram, self.bus
        end, nmi, due, irq = self.max_cycles, False, None, False
        handlers = self.handlers
        watched = {}
        while mpu.processorCycles < end:
            now = mpu.processorCycles
            if self.dirty:
                due, irq, self.dirty = self.pending(now), self.irq_line(), False
            if due and now >= due[0]:
                nmi |= self.raise_(*due) and due[1][0] == 1
                due, irq = self.pending(now + 1), self.irq_line()
                continue
            if nmi or irq and not mpu.p & _I_FLAG:
                self.take(nmi)
                nmi = False
                continue
            pc, sp = mpu.pc, mpu.sp
            inside = len(handlers) > 1
            if inside and bus.bits & mm.HIRAM and pc >= mm.KERNAL_BASE:
                mpu.pc = self.base + _IRQRET + _kernal_exit(pc)
                continue
            asserted = irq and bool(mpu.p & _I_FLAG)
            key = zlib.crc32(ram), asserted
            if key not in watched:
                watched = {key: self._watch(asserted)}
            watch = watched[key]
            stops = watch | _KERNAL_PCS if inside and bus.bits & mm.HIRAM else watch
            if pc not in stops:
                limit = min(end, due[0] if due else end) - now - 1
                mpu.stop, self.brk = False, False
                reason = mpu.run_until(
                    max_cycles=limit,
                    stop_pcs=stops,
                    sp_above=self.frames[-1] if inside else None,
                )
                if self.brk:
                    handlers.append(handlers[-1])
                    self.frames.append(mpu.sp)
                self._unwind()
                if reason != "jam":
                    continue
                pc, sp = mpu.pc, mpu.sp
            else:
                brk = ram[pc] == _BRK
                mpu.step()
                if brk:
                    handlers.append(handlers[-1])
                    self.frames.append(mpu.sp)
                self._unwind()
            if mpu.pc == pc and mpu.sp == sp:
                end = min(end, mpu.processorCycles + frames * self.frame)
                if due is None:
                    break
                mpu.processorCycles = max(mpu.processorCycles, due[0])
        return mpu.processorCycles if end == self.max_cycles else end

    def execute_stepped(self, frames: int) -> int:
        """:meth:`execute`, one instruction at a time (the reference loop)."""
        mpu, ram, bus = self.mpu, self.ram, self.bus
        end, nmi, due = self.max_cycles, False, None
        handlers = self.handlers
        while mpu.processorCycles < end:
            now = mpu.processorCycles
            if self.dirty:
                due, irq, self.dirty = self.pending(now), self.irq_line(), False
            if due and now >= due[0]:
                nmi |= self.raise_(*due) and due[1][0] == 1
                due, irq = self.pending(now + 1), self.irq_line()
                continue
            if nmi or irq and not mpu.p & _I_FLAG:
                self.take(nmi)
                nmi = False
                continue
            pc, sp = mpu.pc, mpu.sp
            if len(handlers) > 1 and bus.bits & mm.HIRAM and pc >= mm.KERNAL_BASE:
                mpu.pc = self.base + _IRQRET + _kernal_exit(pc)
                continue
            brk = ram[pc] == _BRK
            mpu.step()
            if brk:
                handlers.append(handlers[-1])
                self.frames.append(mpu.sp)
            while len(handlers) > 1 and mpu.sp > self.frames[-1]:
                handlers.pop()
                self.frames.pop()
            if mpu.pc == pc and mpu.sp == sp:
                end = min(end, mpu.processorCycles + frames * self.frame)
                if due is None:
                    break
                mpu.processorCycles = max(mpu.processorCycles, due[0])
        return mpu.processorCycles if end == self.max_cycles else end


def _cadence(machine: _Machine, start: int, frames: int) -> Cadence:
    """The :class:`Cadence` of the play source.

    That is the source whose raises drive the most distinct SID registers (a
    sample player drives few), then the one with the most writing raises; any
    recurring ``opaque`` source (see :meth:`_Machine.take`) makes it unknown.
    """
    frame = machine.frame
    ntsc = frame == mm.NTSC.cycles_per_frame
    clock_hz = mm.video(ntsc).clock_hz
    regs, calls, recurs = Counter(), Counter(), Counter()
    for t, src, wrote, _ in machine.events:
        regs[src] |= wrote
        calls[src] += bool(wrote)
        recurs[src] += t >= start
    live = [s for s in regs if recurs[s] > 1]
    source = max(
        (s for s in live if regs[s]),
        key=lambda s: (regs[s].bit_count(), calls[s]),
        default=None,
    )
    if machine.opaque.intersection(live):
        source = None
    if source is None:
        return Cadence(frame, None, clock_hz)
    steady = [(t, tag) for t, src, _, tag in machine.events if src == source]
    steady = [(t, tag) for t, tag in steady if t >= start]
    if source == _VIC:
        per_frame = len(steady) / frames
        video = TriggerSource.NTSC_VIDEO if ntsc else TriggerSource.PAL_VIDEO
        lines = tuple(sorted({line for _, line in steady}))
        cycles = round(frame / per_frame)
        return Cadence(cycles, video, clock_hz, None, False, per_frame, lines)
    chip, timer = source
    cia = machine.cias[chip]
    trigger = TriggerSource.NMI_TIMER if chip else TriggerSource.CIA_TIMER
    dynamic = len({period for _, period in steady}) > 1
    if (dynamic or cia.cr[timer] & 8) and len(steady) > 1:
        mean = (steady[-1][0] - steady[0][0]) / (len(steady) - 1)
        return Cadence(
            round(mean), trigger, clock_hz, cia.latch[timer], dynamic, frame / mean
        )
    return Cadence(cia.period(timer), trigger, clock_hz, cia.latch[timer])


def playroutine_cadence(
    image_or_bytes: Union[SidImage, bytes, bytearray],
    *,
    clock: Optional[str] = None,
    frames: int = 8,
    subtune: int = 0,
    max_cycles: int = 8_000_000,
) -> Cadence:
    """Derive the play-routine :class:`Cadence` from the tune's interrupt programming.

    Seeds sidplayfp's PSID driver state and runs the tune, raising and taking
    interrupts, until ``frames`` frames after init idles (or ``max_cycles``); the
    last ``frames - 1`` are measured. ``clock`` overrides the header clock bits.

    Raises:
      EmulatorUnavailable: jennings is missing.
      SidParseError: the image has no SID header.
    """
    image = (
        image_or_bytes
        if isinstance(image_or_bytes, SidImage)
        else SidImage.from_bytes(bytes(image_or_bytes))
    )
    header = image.header
    if header is None:
        raise SidParseError("cannot derive a cadence: image has no SID header")
    ntsc = _resolve_standard(image, clock)
    machine = _Machine(image, ntsc, max_cycles)
    machine.driver(header, subtune, ntsc)
    end = machine.execute(frames)
    return _cadence(machine, end - (frames - 1) * machine.frame, frames - 1)


def cadence_from_latch(latch: Optional[int], clock: Optional[str] = None) -> Cadence:
    """Derive a :class:`Cadence` from a static CIA timer latch (no-init path).

    A plausible latch (``>= 256``) is a CIA-timer cadence of ``latch + 1``
    cycles per call; otherwise the PAL/NTSC video frame. ``clock`` selects the
    video standard (``"PAL"``/``"NTSC"``, default PAL).
    """
    key = (clock or "PAL").strip().upper()
    if key not in ("PAL", "NTSC"):
        raise ValueError(f"clock must be 'PAL' or 'NTSC', got {clock!r}")
    ntsc = key == "NTSC"
    clock_hz = mm.video(ntsc).clock_hz
    if latch is not None and latch >= _MIN_CIA_LATCH:
        return Cadence(latch + 1, TriggerSource.CIA_TIMER, clock_hz, latch)
    frame = mm.video(ntsc).cycles_per_frame
    source = TriggerSource.NTSC_VIDEO if ntsc else TriggerSource.PAL_VIDEO
    return Cadence(frame, source, clock_hz)


def cadence_info(data: bytes, song: int = 1) -> dict:
    """The play-routine build a survey renderer reports for 1-based ``song`` of ``data``.

    ``rsid`` (an RSID container), ``trigger`` (the :class:`TriggerSource` value, or
    ``None``), ``speed`` (play calls per PAL frame, rounded) and ``dynamic`` from
    :func:`playroutine_cadence`, so ``survey triage`` breaks exact rates down by them.
    """
    cadence = playroutine_cadence(data, subtune=song - 1)
    return {
        "rsid": bytes(data[:4]) == RSID_MAGIC,
        "trigger": cadence.source and cadence.source.value,
        "speed": round(mm.PAL.cycles_per_frame / cadence.cycles_per_call),
        "dynamic": cadence.dynamic,
    }


__all__ = [
    "TriggerSource",
    "Cadence",
    "playroutine_cadence",
    "cadence_from_latch",
    "cadence_info",
    "EmulatorUnavailable",
]
