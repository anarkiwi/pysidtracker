"""Derive a tune's play-routine cadence from what its init actually programs.

*Playroutine cadence* -- the number of CPU cycles between consecutive
play-routine calls -- is a global SID concept, not a per-format constant. It is
set by whichever source triggers the play interrupt:

* a **PAL** video frame (``PAL_CYCLES_PER_FRAME`` = 19656 cycles),
* an **NTSC** video frame (``NTSC_CYCLES_PER_FRAME`` = 17095 cycles), or
* a **CIA timer** (or NMI timer) whose latch defines an arbitrary period
  (e.g. a defMON tune driving its own tempo).

The SID header's advertised speed/clock fields are only a *hint* -- packed and
IRQ-driven tunes program the real trigger from inside their init routine, and a
player can even reprogram the timer mid-tune. :func:`playroutine_cadence`
therefore *derives* the cadence by tracing what init installs (reusing
:func:`pysidtracker.trace.trace_init`), not by trusting the header.

CIA timer period
----------------
A CIA timer in continuous (reload) mode is loaded with its 16-bit latch ``L``
and counts ``L, L-1, ..., 1, 0``; on the cycle after it reaches ``0`` it
underflows (raising the interrupt) and reloads ``L``. Two consecutive
underflows are therefore ``L + 1`` cycles apart, so
``cycles_per_call = latch + 1``. This is validated against a real defMON tune:
its init latches ``0x5BF9`` (23545), which yields the documented defMON cadence
of ``23546`` cycles per call.

CIA arming
----------
A plausible latch alone does not make the timer the play trigger: a tune can
load a Timer-A latch while leaving the timer stopped or its underflow interrupt
masked, so the real trigger stays the video frame. On a real C64 the KERNAL
leaves CIA1 Timer-A running in continuous mode with its underflow IRQ enabled,
and tunes typically just reprogram the latch, so the latch is the cadence
*unless the tune disarms the timer*:

* **control register** (``$DC0E``/``$DD0E``): START bit (bit0) cleared, or
  one-shot mode (bit3 set) -- no periodic underflow interrupt;
* **interrupt control** (``$DC0D``/``$DD0D``): a write with bit7=0 and the
  Timer-A mask bit (bit0) set -- clears the Timer-A interrupt enable.

An unwritten control/ICR is the armed KERNAL default. When a plausible latch is
present but the timer is disarmed, the cadence falls through to the PAL/NTSC
video frame. This is applied symmetrically to CIA #1 (IRQ) and CIA #2 (NMI).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional, Union

from . import registers as reg
from .errors import EmulatorUnavailable
from .image import SidImage
from .trace import trace_init

# A CIA Timer-A latch below this is not a plausible play cadence (it is a
# lo-byte-only artefact, e.g. the ``$FF`` a reset writes to ``$DC04``); a real
# play period always programs the hi byte, so it is >= 256.
_MIN_CIA_LATCH = 0x100


class TriggerSource(enum.Enum):
    """What triggers each play-routine call."""

    PAL_VIDEO = "pal_video"  # PAL raster/VBI frame (19656 cycles)
    NTSC_VIDEO = "ntsc_video"  # NTSC raster/VBI frame (17095 cycles)
    CIA_TIMER = "cia_timer"  # a CIA (or NMI) timer latch period


@dataclass(frozen=True)
class Cadence:
    """The derived play-routine cadence of a tune.

    Attributes:
      cycles_per_call: CPU cycles between consecutive play calls.
      source: the :class:`TriggerSource` driving the cadence.
      clock_hz: the CPU clock (``PAL_CLOCK_HZ`` / ``NTSC_CLOCK_HZ``) for the
        resolved video standard; combine with ``cycles_per_call`` for the call
        rate in Hz (``clock_hz / cycles_per_call``).
      latch: the CIA Timer-A latch when CIA-driven, else ``None``.
      dynamic: True if the tune rewrites the timer latch to a different value
        during play (a variable-tempo player); the reported ``cycles_per_call``
        is then only the initial cadence.
    """

    cycles_per_call: int
    source: TriggerSource
    clock_hz: int
    latch: Optional[int] = None
    dynamic: bool = False

    @property
    def calls_per_second(self) -> float:
        """The play-call rate in Hz (``clock_hz / cycles_per_call``)."""
        return self.clock_hz / self.cycles_per_call


def _resolve_standard(image: SidImage, clock: Optional[str]) -> bool:
    """Return True for NTSC, False for PAL.

    Honours an explicit ``clock`` ("PAL"/"NTSC"); else reads the PSID/RSID v2+
    header ``flags`` clock bits (bits 2-3: 01=PAL, 10=NTSC) as a hint; defaults
    to PAL.
    """
    if clock is not None:
        key = clock.strip().upper()
        if key == "NTSC":
            return True
        if key == "PAL":
            return False
        raise ValueError(f"clock must be 'PAL' or 'NTSC', got {clock!r}")
    if image.header is not None:
        clock_bits = (image.header.flags >> 2) & 0x3
        if clock_bits == 0b10:  # NTSC
            return True
    return False  # PAL default (also for %00 unknown and %11 both)


def _cia_armed(control: Optional[int], icr: Optional[int]) -> bool:
    """True if a CIA Timer-A is armed as a periodic play-interrupt source.

    An unwritten control/ICR (``None``) is the armed KERNAL default. A cleared
    control START bit (bit0), one-shot mode (control bit3), or an ICR write
    clearing the Timer-A enable (bit7=0, mask bit0 set) disarms it.
    """
    if control is not None and (not control & 0x01 or control & 0x08):
        return False
    if icr is not None and not icr & 0x80 and icr & 0x01:
        return False
    return True


def _cia_latch(trace) -> tuple[Optional[int], Optional[int]]:
    """The plausible, *armed* CIA Timer-A play latch and its mid-play rewrite.

    Prefers CIA #1 over CIA #2. A latch counts only when its timer is armed
    (:func:`_cia_armed`); a disarmed latch is ignored so the cadence falls
    through to the video frame. Returns ``(latch, rewritten_latch)``.
    """
    for latch, rewritten, control, icr in (
        (
            trace.cia1_timer_latch,
            trace.cia1_latch_rewritten,
            trace.cia1_control,
            trace.cia1_icr,
        ),
        (
            trace.cia2_timer_latch,
            trace.cia2_latch_rewritten,
            trace.cia2_control,
            trace.cia2_icr,
        ),
    ):
        if latch is not None and latch >= _MIN_CIA_LATCH and _cia_armed(control, icr):
            return latch, rewritten
    return None, None


def playroutine_cadence(
    image_or_bytes: Union[SidImage, bytes, bytearray],
    *,
    clock: Optional[str] = None,
    play_calls: int = 8,
) -> Cadence:
    """Derive the play-routine :class:`Cadence` of a tune.

    Resolves the video standard (explicit ``clock``, else the header clock-bit
    hint, else PAL), then traces the tune's init (and ``play_calls`` play calls)
    to observe the real trigger. If init programs a plausible CIA Timer-A latch
    *and leaves the timer armed* (see :func:`_cia_armed`), the cadence is
    CIA-driven (``cycles_per_call = latch + 1``); a disarmed timer or no latch
    is video-timed (one PAL/NTSC frame). ``dynamic`` is set when a play call
    rewrites the latch to a different value.

    Requires the ``jennings`` emulator (a core dependency); raises
    :class:`~pysidtracker.errors.EmulatorUnavailable` if it is missing.
    """
    image = (
        image_or_bytes
        if isinstance(image_or_bytes, SidImage)
        else SidImage.from_bytes(bytes(image_or_bytes))
    )
    ntsc = _resolve_standard(image, clock)
    clock_hz = reg.NTSC_CLOCK_HZ if ntsc else reg.PAL_CLOCK_HZ

    trace = trace_init(image, play_calls=play_calls)
    latch, rewritten = _cia_latch(trace)

    if latch is not None:
        # CIA/NMI timer drives the play interrupt: the latch IS the cadence.
        return Cadence(
            cycles_per_call=latch + 1,
            source=TriggerSource.CIA_TIMER,
            clock_hz=clock_hz,
            latch=latch,
            dynamic=rewritten is not None,
        )

    frame = reg.NTSC_CYCLES_PER_FRAME if ntsc else reg.PAL_CYCLES_PER_FRAME
    source = TriggerSource.NTSC_VIDEO if ntsc else TriggerSource.PAL_VIDEO
    # TODO: a full per-call cadence schedule (video tunes that switch to a CIA
    # timer mid-play, or multi-speed raster) is a future extension; for now a
    # video-timed tune reports a single constant frame cadence.
    return Cadence(
        cycles_per_call=frame,
        source=source,
        clock_hz=clock_hz,
        latch=None,
        dynamic=False,
    )


def cadence_from_latch(latch: Optional[int], clock: Optional[str] = None) -> Cadence:
    """Derive a :class:`Cadence` from a static CIA timer latch (no-init path).

    Shares the traced path's model: a plausible latch (``>= 256``) is a
    CIA-timer cadence of ``latch + 1`` cycles per call; a missing/implausible
    latch falls back to the PAL/NTSC video frame. ``clock`` selects the video
    standard (``"PAL"``/``"NTSC"``, default PAL) for the reported ``clock_hz``.
    """
    key = (clock or "PAL").strip().upper()
    if key not in ("PAL", "NTSC"):
        raise ValueError(f"clock must be 'PAL' or 'NTSC', got {clock!r}")
    ntsc = key == "NTSC"
    clock_hz = reg.NTSC_CLOCK_HZ if ntsc else reg.PAL_CLOCK_HZ
    if latch is not None and latch >= _MIN_CIA_LATCH:
        return Cadence(
            cycles_per_call=latch + 1,
            source=TriggerSource.CIA_TIMER,
            clock_hz=clock_hz,
            latch=latch,
        )
    frame = reg.NTSC_CYCLES_PER_FRAME if ntsc else reg.PAL_CYCLES_PER_FRAME
    source = TriggerSource.NTSC_VIDEO if ntsc else TriggerSource.PAL_VIDEO
    return Cadence(cycles_per_call=frame, source=source, clock_hz=clock_hz)


# EmulatorUnavailable is surfaced by trace_init; re-exported name for callers.
__all__ = [
    "TriggerSource",
    "Cadence",
    "playroutine_cadence",
    "cadence_from_latch",
    "EmulatorUnavailable",
]
