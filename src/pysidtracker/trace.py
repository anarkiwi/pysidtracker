"""Trace where a tune's init programs the timer/IRQ vectors, via jennings.

A relocated or IRQ-driven tune does not describe its real play routine or play
cadence in the header: it installs its own IRQ handler at the RAM vector
``$0314``/``$0315`` and latches the CIA timer ``$DC04``/``$DC05`` from *inside*
its init routine (Soundmonitor's CIA-timed cohort is the canonical example).
Reading the header's ``playAddress`` misses this entirely.

:func:`trace_init` runs the init routine in jennings (reusing the same stack-return
mechanics as :func:`pysidtracker.detect.run_init`) with a write observer over the
hardware-register and interrupt-vector addresses, optionally calls the play
address a few times, and returns an :class:`InitTrace` recording *where* the
vectors were programmed and *to what*: the CIA timer latches (the cadence), the
installed IRQ vector (the real play routine), the NMI vector, the VIC raster
compare, the set of registers touched, and the SID writes. Requires the core
``jennings`` dependency; raises
:class:`~pysidtracker.errors.EmulatorUnavailable` if it is missing, consistent
with :func:`run_init`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

from vagrant_moses import memmap as mm

from . import registers as reg
from .emu import run_to_rts, run_to_rts_until, set_iomap, sync_ram, wire_mpu
from .errors import SidParseError
from .poweron import psid_init_bank, psid_play_bank
from .registers import sid_model_for_flags
from .image import SidImage

# Addresses the write observer watches: the full SID mirror band, the VIC and
# CIA register files, the RAM interrupt vectors and the CPU vector table.
_WATCH_ADDRS = (
    *range(mm.SID_BASE, mm.SID_LAST + 1),
    *range(mm.VIC_BASE, mm.VIC_LAST + 1),
    *range(mm.CIA1_BASE, mm.CIA1_LAST + 1),
    *range(mm.CIA2_BASE, mm.CIA2_LAST + 1),
    *mm.VECTORS,
)


@dataclass
class InitTrace:
    """What a tune's init (and optional play calls) programmed.

    A vector field is ``None`` when neither of its two bytes was written.

    Attributes:
      cia1_timer_latch: ``$DC04``/``$DC05`` latch value (the play cadence).
      cia2_timer_latch: ``$DD04``/``$DD05`` latch value.
      irq_vector: value installed at the RAM IRQ vector ``$0314``/``$0315`` --
        the real play/IRQ routine address the header may hide.
      hw_irq_vector: value written to the CPU IRQ vector ``$FFFE``/``$FFFF``.
      nmi_vector: value installed at the RAM NMI vector ``$0318``/``$0319``.
      vic_raster: raster compare line if ``$D012`` was written (folding in the
        ``$D011`` bit-7 high bit when that was written too).
      registers_touched: every watched address that received a write.
      sid_writes: last value written to each SID register address touched.
      cia1_latch_rewritten: a play call's new value for the CIA #1 Timer-A
        latch, or ``None`` when no play call wrote one.
      cia2_latch_rewritten: same for CIA #2 Timer-A.
      cia1_control: last value written to CIA #1 control register A ($DC0E),
        or ``None`` if unwritten (KERNAL default: Timer-A continuous + running).
        Bit0 is START, bit3 selects one-shot mode.
      cia1_icr: last value written to CIA #1 interrupt control ($DC0D), or
        ``None`` if unwritten. Bit7 is the set/clear select; bit0 is the
        Timer-A interrupt mask.
      cia2_control: same as ``cia1_control`` for CIA #2 control A ($DD0E).
      cia2_icr: same as ``cia1_icr`` for CIA #2 interrupt control ($DD0D).
    """

    cia1_timer_latch: Optional[int] = None
    cia2_timer_latch: Optional[int] = None
    irq_vector: Optional[int] = None
    hw_irq_vector: Optional[int] = None
    nmi_vector: Optional[int] = None
    vic_raster: Optional[int] = None
    registers_touched: Set[int] = field(default_factory=set)
    sid_writes: Dict[int, int] = field(default_factory=dict)
    cia1_latch_rewritten: Optional[int] = None
    cia2_latch_rewritten: Optional[int] = None
    cia1_control: Optional[int] = None
    cia1_icr: Optional[int] = None
    cia2_control: Optional[int] = None
    cia2_icr: Optional[int] = None


def _word(writes: Dict[int, int], lo_addr: int, hi_addr: int) -> Optional[int]:
    if lo_addr not in writes and hi_addr not in writes:
        return None
    return writes.get(lo_addr, 0) | (writes.get(hi_addr, 0) << 8)


def _raster(writes: Dict[int, int]) -> Optional[int]:
    if mm.VIC_RASTER not in writes:
        return None
    return mm.raster_compare(writes.get(mm.VIC_CONTROL_1, 0), writes[mm.VIC_RASTER])


def _build_trace(
    writes: Dict[int, int],
    cia1_rewritten: Optional[int] = None,
    cia2_rewritten: Optional[int] = None,
) -> InitTrace:
    sid_writes = {a: v for a, v in writes.items() if reg.is_sid_reg(a)}
    return InitTrace(
        cia1_timer_latch=_word(writes, mm.CIA1_TIMER_A_LO, mm.CIA1_TIMER_A_HI),
        cia2_timer_latch=_word(writes, mm.CIA2_TIMER_A_LO, mm.CIA2_TIMER_A_HI),
        irq_vector=_word(writes, mm.RAM_IRQ_VECTOR_LO, mm.RAM_IRQ_VECTOR_HI),
        hw_irq_vector=_word(writes, mm.CPU_IRQ_VECTOR, mm.CPU_IRQ_VECTOR + 1),
        nmi_vector=_word(writes, mm.RAM_NMI_VECTOR_LO, mm.RAM_NMI_VECTOR_HI),
        vic_raster=_raster(writes),
        registers_touched=set(writes),
        sid_writes=sid_writes,
        cia1_latch_rewritten=cia1_rewritten,
        cia2_latch_rewritten=cia2_rewritten,
        cia1_control=writes.get(mm.CIA1_CRA),
        cia1_icr=writes.get(mm.CIA1_ICR),
        cia2_control=writes.get(mm.CIA2_CRA),
        cia2_icr=writes.get(mm.CIA2_ICR),
    )


def trace_init(
    image: SidImage,
    *,
    subtune: int = 0,
    play_calls: int = 0,
    max_cycles: int = 8_000_000,
) -> InitTrace:
    """Run ``image``'s init under a write observer and return an :class:`InitTrace`.

    Runs the init routine in jennings, capturing writes to the hardware-register and
    interrupt-vector addresses, then calls the header's play address
    ``play_calls`` times (each also observed). ``subtune`` is passed to init in
    the accumulator (the SID calling convention).

    Requires the core ``jennings`` dependency; raises
    :class:`~pysidtracker.errors.EmulatorUnavailable` if it is missing and
    :class:`~pysidtracker.errors.SidParseError` if the image has no init address.
    """
    if image.header is None:
        raise SidParseError("cannot trace init: image has no SID header")

    writes: Dict[int, int] = {}

    def _record(address, value):
        writes[address] = value & 0xFF
        return None

    init_address = image.header.init_address or image.header.real_load_address
    # The driver banks the init's ROM space before calling it (KERNAL/BASIC in).
    image.mem[1] = psid_init_bank(init_address)
    mpu, mem = wire_mpu(image.mem, sid_model=sid_model_for_flags(image.header.flags))
    mem.subscribe_to_write(_WATCH_ADDRS, _record)

    run_to_rts(mpu, mem, init_address, subtune, max_cycles)

    # Post-init CIA Timer-A latch values, to detect a mid-play reschedule.
    init_cia1 = _word(writes, mm.CIA1_TIMER_A_LO, mm.CIA1_TIMER_A_HI)
    init_cia2 = _word(writes, mm.CIA2_TIMER_A_LO, mm.CIA2_TIMER_A_HI)
    cia1_rewritten: Optional[int] = None
    cia2_rewritten: Optional[int] = None

    play_address = image.header.play_address
    if play_address:
        set_iomap(mpu, psid_play_bank(play_address))
        for _ in range(play_calls):
            run_to_rts(mpu, mem, play_address, 0, max_cycles)
            play_cia1 = _word(writes, mm.CIA1_TIMER_A_LO, mm.CIA1_TIMER_A_HI)
            play_cia2 = _word(writes, mm.CIA2_TIMER_A_LO, mm.CIA2_TIMER_A_HI)
            if play_cia1 is not None and play_cia1 != init_cia1:
                cia1_rewritten = play_cia1
            if play_cia2 is not None and play_cia2 != init_cia2:
                cia2_rewritten = play_cia2

    sync_ram(mpu)
    return _build_trace(writes, cia1_rewritten, cia2_rewritten)


def _init_address(image: SidImage, init: Optional[int]) -> int:
    if init is not None:
        return init
    if image.header is None:
        raise SidParseError("no init address: the image has no SID header")
    return image.header.init_address or image.header.real_load_address


def trace_init_argument(
    image: SidImage,
    init: Optional[int],
    entry: int,
    subtune: int = 0,
    max_cycles: int = 1_000_000,
) -> Optional[int]:
    """``A`` when ``image``'s init, called with ``A = subtune``, first reaches ``entry``.

    ``init`` defaults to the header's; ``None`` if init returns or ``max_cycles``
    run out first. E.g. the player base behind a ``TAX / LDA table,X / JMP base``.
    """
    init = _init_address(image, init)
    subject = bytearray(image.mem)
    subject[mm.CPU_PORT] = psid_init_bank(init)
    mpu, mem = wire_mpu(subject)
    if run_to_rts_until(mpu, mem, init, subtune, {entry}, max_cycles):
        return mpu.a
    return None


def init_arguments(
    image: SidImage,
    entry: int,
    init: Optional[int] = None,
    max_cycles: int = 1_000_000,
) -> Tuple[Optional[int], ...]:
    """:func:`trace_init_argument` of every 0-based subtune the header lists."""
    songs = (image.header.songs if image.header is not None else 0) or 1
    return tuple(
        trace_init_argument(image, init, entry, subtune, max_cycles)
        for subtune in range(songs)
    )
