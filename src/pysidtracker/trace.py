"""Trace where a tune's init programs the timer/IRQ vectors, via py65.

A relocated or IRQ-driven tune does not describe its real play routine or play
cadence in the header: it installs its own IRQ handler at the RAM vector
``$0314``/``$0315`` and latches the CIA timer ``$DC04``/``$DC05`` from *inside*
its init routine (Soundmonitor's CIA-timed cohort is the canonical example).
Reading the header's ``playAddress`` misses this entirely.

:func:`trace_init` runs the init routine in py65 (reusing the same stack-return
mechanics as :func:`pysidtracker.detect.run_init`) with a write observer over the
hardware-register and interrupt-vector addresses, optionally calls the play
address a few times, and returns an :class:`InitTrace` recording *where* the
vectors were programmed and *to what*: the CIA timer latches (the cadence), the
installed IRQ vector (the real play routine), the NMI vector, the VIC raster
compare, the set of registers touched, and the SID writes. Requires the core
``py65`` dependency; raises
:class:`~pysidtracker.errors.EmulatorUnavailable` if it is missing, consistent
with :func:`run_init`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from . import registers as reg
from .emu import run_to_rts, wire_mpu
from .errors import SidParseError
from .image import SidImage

# Addresses the write observer watches: the full SID mirror band, the VIC and
# CIA register files, the RAM interrupt vectors and the CPU vector table.
_WATCH_ADDRS = tuple(
    list(range(reg.SID_BASE, reg.SID_LAST + 1))
    + list(range(reg.VIC_BASE, reg.VIC_LAST + 1))
    + list(range(reg.CIA1_BASE, reg.CIA1_LAST + 1))
    + list(range(reg.CIA2_BASE, reg.CIA2_LAST + 1))
    + [
        reg.RAM_IRQ_VECTOR_LO,
        reg.RAM_IRQ_VECTOR_HI,
        reg.RAM_NMI_VECTOR_LO,
        reg.RAM_NMI_VECTOR_HI,
        reg.CPU_NMI_VECTOR,
        reg.CPU_NMI_VECTOR + 1,
        reg.CPU_IRQ_VECTOR,
        reg.CPU_IRQ_VECTOR + 1,
    ]
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
      cia1_latch_rewritten: the CIA #1 Timer-A latch was rewritten to a value
        *different* from what init left, during the observed play calls (the
        tune reschedules its own cadence -- a dynamic/variable-tempo player).
        ``None`` when unknown (no play calls) or unchanged.
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
    if reg.VIC_RASTER not in writes:
        return None
    line = writes[reg.VIC_RASTER]
    if reg.VIC_CONTROL_1 in writes:
        line |= (writes[reg.VIC_CONTROL_1] & 0x80) << 1
    return line


def _build_trace(
    writes: Dict[int, int],
    cia1_rewritten: Optional[int] = None,
    cia2_rewritten: Optional[int] = None,
) -> InitTrace:
    sid_writes = {a: v for a, v in writes.items() if reg.is_sid_reg(a)}
    return InitTrace(
        cia1_timer_latch=_word(writes, reg.CIA1_TIMER_A_LO, reg.CIA1_TIMER_A_HI),
        cia2_timer_latch=_word(writes, reg.CIA2_TIMER_A_LO, reg.CIA2_TIMER_A_HI),
        irq_vector=_word(writes, reg.RAM_IRQ_VECTOR_LO, reg.RAM_IRQ_VECTOR_HI),
        hw_irq_vector=_word(writes, reg.CPU_IRQ_VECTOR, reg.CPU_IRQ_VECTOR + 1),
        nmi_vector=_word(writes, reg.RAM_NMI_VECTOR_LO, reg.RAM_NMI_VECTOR_HI),
        vic_raster=_raster(writes),
        registers_touched=set(writes),
        sid_writes=sid_writes,
        cia1_latch_rewritten=cia1_rewritten,
        cia2_latch_rewritten=cia2_rewritten,
        cia1_control=writes.get(reg.CIA1_CRA),
        cia1_icr=writes.get(reg.CIA1_ICR),
        cia2_control=writes.get(reg.CIA2_CRA),
        cia2_icr=writes.get(reg.CIA2_ICR),
    )


def trace_init(
    image: SidImage,
    *,
    subtune: int = 0,
    play_calls: int = 0,
    max_cycles: int = 8_000_000,
) -> InitTrace:
    """Run ``image``'s init under a write observer and return an :class:`InitTrace`.

    Runs the init routine in py65, capturing writes to the hardware-register and
    interrupt-vector addresses, then calls the header's play address
    ``play_calls`` times (each also observed). ``subtune`` is passed to init in
    the accumulator (the SID calling convention).

    Requires the core ``py65`` dependency; raises
    :class:`~pysidtracker.errors.EmulatorUnavailable` if it is missing and
    :class:`~pysidtracker.errors.SidParseError` if the image has no init address.
    """
    if image.header is None:
        raise SidParseError("cannot trace init: image has no SID header")

    writes: Dict[int, int] = {}

    def _record(address, value):
        writes[address] = value & 0xFF
        return None

    mpu, mem = wire_mpu(image.mem)
    mem.subscribe_to_write(_WATCH_ADDRS, _record)

    init_address = image.header.init_address or image.header.real_load_address
    run_to_rts(mpu, mem, init_address, subtune, max_cycles)

    # Post-init CIA Timer-A latch values, to detect a mid-play reschedule.
    init_cia1 = _word(writes, reg.CIA1_TIMER_A_LO, reg.CIA1_TIMER_A_HI)
    init_cia2 = _word(writes, reg.CIA2_TIMER_A_LO, reg.CIA2_TIMER_A_HI)
    cia1_rewritten: Optional[int] = None
    cia2_rewritten: Optional[int] = None

    play_address = image.header.play_address
    if play_address:
        for _ in range(play_calls):
            run_to_rts(mpu, mem, play_address, 0, max_cycles)
            play_cia1 = _word(writes, reg.CIA1_TIMER_A_LO, reg.CIA1_TIMER_A_HI)
            play_cia2 = _word(writes, reg.CIA2_TIMER_A_LO, reg.CIA2_TIMER_A_HI)
            if play_cia1 is not None and play_cia1 != init_cia1:
                cia1_rewritten = play_cia1
            if play_cia2 is not None and play_cia2 != init_cia2:
                cia2_rewritten = play_cia2

    return _build_trace(writes, cia1_rewritten, cia2_rewritten)
