"""Derive a tune's play cadence from how its interrupts are actually programmed.

:func:`playroutine_cadence` seeds the state sidplayfp's PSID driver leaves, then
runs the tune on jennings with VIC raster and CIA timer interrupts raised and
taken as the C64 would; see ``docs/cadence.md`` for the driver and the model.
"""

from __future__ import annotations

import enum
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from . import registers as reg
from .emu import wire_mpu
from .errors import EmulatorUnavailable, SidParseError
from .image import SidImage

# A CIA Timer-A latch below this is a lo-byte-only artefact, not a play period.
_MIN_CIA_LATCH = 0x100
# psiddrv.a65 ``irqjob``/``irqret`` offsets from the relocated driver's first byte.
_IRQJOB, _IRQRET = 0xA8, 0xBB
_RET, _EXCEPTION = 0xA0, 0xC4
_KERNAL_ROM, _KERNAL_NMI = 0xE000, 0xFE47
_RTI, _I_FLAG = 0x40, 0x04
_VIC = (-1, 0)
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
        ntsc = self.clock_hz == reg.NTSC_CLOCK_HZ
        return reg.NTSC_CYCLES_PER_FRAME if ntsc else reg.PAL_CYCLES_PER_FRAME

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
    if addr < 0xA000:
        return 0x37
    if addr < 0xD000:
        return 0x36
    return 0x35 if addr >= _KERNAL_ROM else 0x34


def _driver_base(load: int, end: int) -> int:
    """The first free page psiddrv relocates itself to, outside the tune and BASIC."""
    first, last = load >> 8, (end - 1) >> 8
    return next(
        p << 8
        for p in range(4, 0xD0)
        if not first <= p <= last and not 0xA0 <= p <= 0xBF
    )


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

    def __init__(self, image: SidImage, ntsc: bool, max_cycles: int):
        header = image.header
        self.ram = bytearray(image.mem)
        self.mpu, self.mem = wire_mpu(
            self.ram, sid_model=reg.sid_model_for_flags(header.flags), ntsc=ntsc
        )
        self.frame = reg.NTSC_CYCLES_PER_FRAME if ntsc else reg.PAL_CYCLES_PER_FRAME
        self.line = reg.NTSC_CYCLES_PER_LINE if ntsc else reg.PAL_CYCLES_PER_LINE
        self.max_cycles = max_cycles
        self.cias = (_Cia(), _Cia())
        self.vic_enable = self.raster = self.vic_seen = 0
        self.vic_flag = False
        self.events = []
        self.dirty = True
        extra = [0xD000 | b << 4 for b in (header.second_sid, header.third_sid) if b]
        self.mem.subscribe_to_write(
            [*range(reg.SID_BASE, reg.SID_LAST + 1)]
            + [a + o for a in extra for o in range(reg.SID_REG_COUNT)],
            self._sid,
        )
        self.mem.subscribe_to_write(
            [reg.VIC_CONTROL_1, reg.VIC_RASTER, reg.VIC_IRQ_STATUS, reg.VIC_IRQ_ENABLE]
            + [base + o for base in (reg.CIA1_BASE, reg.CIA2_BASE) for o in range(16)],
            self._write,
        )
        self.mem.subscribe_to_read(
            [reg.VIC_IRQ_STATUS, reg.CIA1_ICR, reg.CIA2_ICR]
            + [
                base + o
                for base in (reg.CIA1_BASE, reg.CIA2_BASE)
                for o in (4, 5, 6, 7, 14, 15)
            ],
            self._read,
        )
        self.base = _driver_base(image.load, image.end)

    def _sid(self, _addr, _value):
        if self.events:
            self.events[-1][2] += 1

    def _write(self, addr, value):
        self.dirty = True
        if addr == reg.VIC_IRQ_ENABLE:
            self.vic_enable = value & 0x0F
        elif addr == reg.VIC_IRQ_STATUS and (value | self._rmw()) & 1:
            self.vic_flag, self.vic_seen = False, self._now()
        elif addr == reg.VIC_RASTER:
            self.raster = self.raster & 0x100 | value
        elif addr == reg.VIC_CONTROL_1:
            self.raster = self.raster & 0xFF | (value & 0x80) << 1
        elif reg.CIA1_BASE <= addr <= reg.CIA2_LAST:
            cia = self.cias[addr >= reg.CIA2_BASE]
            cia.write(addr & 0xF, value, self._now())

    def _now(self) -> int:
        """The cycle of the executing instruction's last (data) access."""
        mpu = self.mpu
        return mpu.processorCycles + mpu.cycletime[self.ram[mpu.pc]] - 1

    def _rmw(self) -> int:
        """1 if the executing instruction is a read-modify-write, whose dummy write
        stores the value it read before the modified one."""
        return int(self.ram[self.mpu.pc] in _RMW)

    def _read(self, addr):
        now = self._now()
        if addr == reg.VIC_IRQ_STATUS:
            match = self._next_match(now - self.frame)
            flag = self.vic_flag or self.vic_seen < match <= now
            return flag | (0x80 if flag and self.vic_enable & 1 else 0)
        cia = self.cias[addr >= reg.CIA2_BASE]
        offset = addr & 0xF
        if offset >= 0xE:
            return cia.cr[offset - 0xE]
        if offset != 0xD:
            return cia.counter(offset - 4 >> 1, now) >> 8 * (offset & 1) & 0xFF
        self.dirty = True
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
        ram[0:2] = bytes((0x2F, initiomap or 0x37))
        kernal = reg.KERNAL_CIA1_LATCH_NTSC if ntsc else reg.KERNAL_CIA1_LATCH_PAL
        compat = tune_ntsc != ntsc or initiomap
        cia_speed = not header.is_psid or header.cia_speed(subtune)
        for addr, value in (
            (reg.VIC_IRQ_ENABLE, 0x00),
            (reg.CIA1_ICR, 0x7F),
            (reg.CIA2_ICR, 0x7F),
            (reg.SID_BASE + 0x18, 0x0F),
            (reg.CIA1_TIMER_A_LO, kernal & 0xFF),
            (reg.CIA1_TIMER_A_HI, kernal >> 8),
            (reg.VIC_CONTROL_1, 0x1B if compat else 0x9B),
            (reg.VIC_RASTER, 0x00 if compat else 0x37),
            *(
                ((reg.CIA1_ICR, 0x81), (reg.CIA1_CRA, 0x01))
                if cia_speed
                else ((reg.VIC_IRQ_ENABLE, 0x81),)
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
        vectors = (
            _word(base + (_IRQJOB if initiomap and playiomap else _IRQRET))
            + _word(base + _EXCEPTION) * 2
            if header.is_psid
            else _word(base + _IRQRET) + _word(0xFE66) + _word(_KERNAL_NMI)
        )
        ram[reg.RAM_IRQ_VECTOR_LO : reg.RAM_NMI_VECTOR_HI + 1] = vectors
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
        return raised

    def irq_line(self) -> bool:
        """The CPU IRQ input: a pending raster IRQ or CIA #1 interrupt."""
        cia = self.cias[0]
        return bool(self.vic_flag and self.vic_enable & 1 or cia.flags & cia.mask)

    def take(self, nmi: bool) -> None:
        """Enter an interrupt through the vector the banked-in ROM or RAM gives."""
        mpu, ram = self.mpu, self.ram
        mpu.nmi()
        if ram[1] & 2:
            if not nmi:
                for value in (mpu.a, mpu.x, mpu.y):
                    ram[0x0100 + mpu.sp] = value
                    mpu.sp = mpu.sp - 1 & 0xFF
            vector = reg.RAM_NMI_VECTOR_LO if nmi else reg.RAM_IRQ_VECTOR_LO
        else:
            vector = reg.CPU_NMI_VECTOR if nmi else reg.CPU_IRQ_VECTOR
        mpu.pc = ram[vector] | ram[vector + 1] << 8

    def execute(self, frames: int) -> int:
        """Run until ``frames`` frames after init first idles; return the last cycle."""
        mpu, ram = self.mpu, self.ram
        end, depth, nmi, due = self.max_cycles, 0, False, None
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
                depth, nmi = depth + 1, False
                continue
            pc, sp = mpu.pc, mpu.sp
            if depth and ram[1] & 2 and pc >= _KERNAL_ROM:
                mpu.pc = self.base + _IRQRET + (8 if pc == _KERNAL_NMI else 3)
                continue
            depth -= depth and ram[pc] == _RTI
            mpu.step()
            if mpu.pc == pc and mpu.sp == sp:
                end = min(end, mpu.processorCycles + frames * self.frame)
                if due is None:
                    break
                mpu.processorCycles = max(mpu.processorCycles, due[0])
        return mpu.processorCycles if end == self.max_cycles else end


def _cadence(machine: _Machine, start: int, frames: int) -> Cadence:
    """The :class:`Cadence` of the source whose raises the SID writes follow."""
    frame = machine.frame
    ntsc = frame == reg.NTSC_CYCLES_PER_FRAME
    clock_hz = reg.NTSC_CLOCK_HZ if ntsc else reg.PAL_CLOCK_HZ
    writes = Counter()
    for _, src, wrote, _ in machine.events:
        writes[src] += wrote
    source, wrote = max(writes.items(), key=lambda kv: kv[1], default=(None, 0))
    steady = [(t, tag) for t, src, _, tag in machine.events if src == source]
    steady = [(t, tag) for t, tag in steady if t >= start]
    if not wrote or not steady:
        return Cadence(frame, None, clock_hz)
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
    clock_hz = reg.NTSC_CLOCK_HZ if ntsc else reg.PAL_CLOCK_HZ
    if latch is not None and latch >= _MIN_CIA_LATCH:
        return Cadence(latch + 1, TriggerSource.CIA_TIMER, clock_hz, latch)
    frame = reg.NTSC_CYCLES_PER_FRAME if ntsc else reg.PAL_CYCLES_PER_FRAME
    source = TriggerSource.NTSC_VIDEO if ntsc else TriggerSource.PAL_VIDEO
    return Cadence(frame, source, clock_hz)


__all__ = [
    "TriggerSource",
    "Cadence",
    "playroutine_cadence",
    "cadence_from_latch",
    "EmulatorUnavailable",
]
