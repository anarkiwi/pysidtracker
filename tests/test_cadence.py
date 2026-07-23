"""Tests for the derived play-routine cadence abstraction."""

import pytest

from pysidtracker import (
    Cadence,
    SidImage,
    TriggerSource,
    playroutine_cadence,
)
from pysidtracker import registers as reg
from pysidtracker.cadence import _cia_armed
from pysidtracker.testing import TuneFetchError, fetch_tune, resolve_tune
from pysidtracker.trace import trace_init

from .helpers import build_psid

jennings = pytest.importorskip("jennings")


def _lda_sta(value, addr):
    """LDA #value ; STA addr (absolute)."""
    return bytes([0xA9, value, 0x8D, addr & 0xFF, addr >> 8])


RTS = bytes([0x60])


# --- crafted images --------------------------------------------------------


def _raster_image(flags=0):
    """Init that sets only the VIC raster compare (video-timed), then RTS."""
    code = _lda_sta(0x30, reg.VIC_RASTER) + RTS
    return build_psid(code, load=0x1000, init=0x1000, play=0x1000, flags=flags)


def test_video_timed_defaults_to_pal():
    cad = playroutine_cadence(_raster_image())
    assert isinstance(cad, Cadence)
    assert cad.source is TriggerSource.PAL_VIDEO
    assert cad.cycles_per_call == reg.PAL_CYCLES_PER_FRAME
    assert cad.clock_hz == reg.PAL_CLOCK_HZ
    assert cad.latch is None
    assert cad.dynamic is False


def test_video_timed_explicit_ntsc():
    cad = playroutine_cadence(_raster_image(), clock="NTSC")
    assert cad.source is TriggerSource.NTSC_VIDEO
    assert cad.cycles_per_call == reg.NTSC_CYCLES_PER_FRAME
    assert cad.clock_hz == reg.NTSC_CLOCK_HZ


def test_video_timed_ntsc_from_header_flags():
    # flags clock bits (bits 2-3) = %10 -> NTSC hint.
    cad = playroutine_cadence(_raster_image(flags=0b10 << 2))
    assert cad.source is TriggerSource.NTSC_VIDEO
    assert cad.cycles_per_call == reg.NTSC_CYCLES_PER_FRAME


def test_bad_clock_string_raises():
    with pytest.raises(ValueError):
        playroutine_cadence(_raster_image(), clock="SECAM")


def test_cia_timer_derives_latch_plus_one():
    # Latch $5BF9 = 23545; a continuous CIA timer underflows after latch+1
    # cycles, so cycles_per_call = 23546 (the documented defMON cadence).
    code = (
        _lda_sta(0xF9, reg.CIA1_TIMER_A_LO)
        + _lda_sta(0x5B, reg.CIA1_TIMER_A_HI)
        + _lda_sta(0x00, reg.RAM_IRQ_VECTOR_LO)
        + _lda_sta(0x40, reg.RAM_IRQ_VECTOR_HI)
        + RTS
    )
    play_addr = 0x1000 + len(code)
    data = build_psid(code + RTS, load=0x1000, init=0x1000, play=play_addr)
    cad = playroutine_cadence(data)
    assert cad.source is TriggerSource.CIA_TIMER
    assert cad.latch == 23545
    assert cad.cycles_per_call == 23546
    assert cad.clock_hz == reg.PAL_CLOCK_HZ
    assert cad.dynamic is False


def test_cia_clock_hz_tracks_standard_but_latch_does_not():
    code = (
        _lda_sta(0xF9, reg.CIA1_TIMER_A_LO) + _lda_sta(0x5B, reg.CIA1_TIMER_A_HI) + RTS
    )
    data = build_psid(code + RTS, load=0x1000, init=0x1000, play=0x1000 + len(code))
    cad = playroutine_cadence(data, clock="NTSC")
    assert cad.source is TriggerSource.CIA_TIMER
    assert cad.cycles_per_call == 23546  # latch-derived, independent of video std
    assert cad.clock_hz == reg.NTSC_CLOCK_HZ


def test_dynamic_when_play_rewrites_latch():
    init = (
        _lda_sta(0xF9, reg.CIA1_TIMER_A_LO) + _lda_sta(0x5B, reg.CIA1_TIMER_A_HI) + RTS
    )
    play = (
        _lda_sta(0x2F, reg.CIA1_TIMER_A_LO) + _lda_sta(0x4D, reg.CIA1_TIMER_A_HI) + RTS
    )
    play_addr = 0x1000 + len(init)
    data = build_psid(init + play, load=0x1000, init=0x1000, play=play_addr)
    cad = playroutine_cadence(data)
    assert cad.source is TriggerSource.CIA_TIMER
    assert cad.dynamic is True


def test_lo_only_cia_write_is_not_a_cadence():
    # A lone $FF to the Timer-A lo byte (a reset artefact) must not be read as a
    # CIA cadence; the tune is still video-timed.
    code = _lda_sta(0xFF, reg.CIA1_TIMER_A_LO) + _lda_sta(0x30, reg.VIC_RASTER) + RTS
    data = build_psid(code, load=0x1000, init=0x1000, play=0x1000)
    cad = playroutine_cadence(data)
    assert cad.source is TriggerSource.PAL_VIDEO


def _cia_image(control_icr=b""):
    """Init that latches an armed CIA cadence plus optional control/ICR writes."""
    code = (
        _lda_sta(0xF9, reg.CIA1_TIMER_A_LO)
        + _lda_sta(0x5B, reg.CIA1_TIMER_A_HI)
        + control_icr
        + RTS
    )
    return build_psid(code, load=0x1000, init=0x1000, play=0x1000)


@pytest.mark.parametrize(
    "control,icr,armed",
    [
        (None, None, True),  # KERNAL default (unwritten) is armed
        (0x11, None, True),  # START set, continuous
        (0x10, None, False),  # START clear
        (0x09, None, False),  # one-shot mode (bit3) even while running
        (None, 0x81, True),  # ICR set Timer-A enable (bit7=1, bit0=1)
        (None, 0x01, False),  # ICR clear Timer-A enable (bit7=0, bit0=1)
        (None, 0x80, True),  # ICR set with no Timer-A bit leaves it armed
    ],
)
def test_cia_armed(control, icr, armed):
    assert _cia_armed(control, icr) is armed


def test_disarmed_cia_latch_falls_back_to_video():
    # A plausible latch whose timer START bit is cleared is not the trigger.
    cad = playroutine_cadence(_cia_image(_lda_sta(0x10, reg.CIA1_CRA)))
    assert cad.source is TriggerSource.PAL_VIDEO
    assert cad.latch is None


def test_icr_masked_cia_latch_falls_back_to_video():
    # A plausible latch whose Timer-A interrupt enable is cleared is not the trigger.
    cad = playroutine_cadence(_cia_image(_lda_sta(0x01, reg.CIA1_ICR)))
    assert cad.source is TriggerSource.PAL_VIDEO


def test_armed_cia_latch_is_cia_timer():
    cad = playroutine_cadence(_cia_image(_lda_sta(0x11, reg.CIA1_CRA)))
    assert cad.source is TriggerSource.CIA_TIMER
    assert cad.cycles_per_call == 23546


def test_accepts_sidimage_instance():
    img = SidImage.from_bytes(_raster_image())
    cad = playroutine_cadence(img)
    assert cad.source is TriggerSource.PAL_VIDEO


def test_calls_per_second_property():
    cad = Cadence(19656, TriggerSource.PAL_VIDEO, reg.PAL_CLOCK_HZ)
    assert cad.calls_per_second == pytest.approx(reg.PAL_CLOCK_HZ / 19656)


# --- real-tune validation (best-effort) ------------------------------------

# A real defMON (Aleksi Eeben) tune that self-installs a CIA-timer play IRQ.
_DEFMON_CIA_TUNE = "MUSICIANS/E/Eeben_Aleksi/Stella_2600_by_Starlight.sid"


def test_real_defmon_tune_is_cia_timer(tmp_path):
    """A real defMON tune derives a CIA cadence of latch+1 cycles.

    Pins the derivation to reality: the CIA latch defMON programs is the tune's
    tempo (here ~19759), and cycles_per_call == latch + 1 -- the same rule that
    yields defMON's canonical 23546 from a 23545 latch. Skips only if the HVSC
    mirror is unreachable.
    """
    try:
        path = fetch_tune(_DEFMON_CIA_TUNE, cache_dir=tmp_path)
    except TuneFetchError:
        pytest.skip("HVSC mirror unreachable")
    data = path.read_bytes()
    latch = trace_init(SidImage.from_bytes(data)).cia1_timer_latch
    cad = playroutine_cadence(data)
    assert cad.source is TriggerSource.CIA_TIMER
    assert cad.latch == latch
    assert cad.cycles_per_call == latch + 1


# Real tunes pinning the arming gate: armed CIA latch vs. video-timed non-CIA.
_ARMING_TUNES = {
    "MUSICIANS/H/Honey/8_Bit-Maerchenland_V2.sid": (
        TriggerSource.CIA_TIMER,
        16641,
    ),
    "MUSICIANS/F/Fern_Eric/Goldberg_Variations_parts_1-7.sid": (
        TriggerSource.NTSC_VIDEO,
        reg.NTSC_CYCLES_PER_FRAME,
    ),
}


@pytest.mark.parametrize("relpath,expected", list(_ARMING_TUNES.items()))
def test_real_tune_arming_gate(relpath, expected, tmp_path):
    """The CIA arming gate resolves the real trigger on two HVSC tunes.

    ``8_Bit-Maerchenland`` latches an armed CIA timer (16640 -> 16641 cycles);
    ``Goldberg_Variations`` is NTSC-video-timed and its CIA latch must not be
    read as the cadence. Skips only if the HVSC mirror is unreachable.
    """
    source, cycles = expected
    path = resolve_tune(relpath, cache_dir=tmp_path)
    if path is None:
        pytest.skip("HVSC mirror unreachable")
    cad = playroutine_cadence(path.read_bytes())
    assert cad.source is source
    assert cad.cycles_per_call == cycles


# An 8x-multispeed defMON tune whose depacker raster-syncs and whose replay uses
# NMOS illegals: init latches $0998, so the cadence is 2457 cycles (confirmed
# against a cycle-stamped sidtrace render: the CIA-IRQ delta mode is 2457).
_MULTISPEED_TUNE = "MUSICIANS/G/Goto80/Automatas.sid"
_MULTISPEED_LATCH = 0x0998


def test_multispeed_defmon_cadence_is_latch_plus_one(tmp_path):
    """A multispeed tune reports its CIA latch + 1, not the PAL video frame.

    Without the full NMOS/C64 init host, init runs to the cycle cap, the
    ``$DC04``/``$DC05`` writes are never seen and the cadence silently falls
    through to 19656 -- 8x wrong for this tune.
    """
    path = resolve_tune(_MULTISPEED_TUNE, cache_dir=tmp_path)
    if path is None:
        pytest.skip("HVSC mirror unreachable")
    data = path.read_bytes()
    assert trace_init(SidImage.from_bytes(data)).cia1_timer_latch == _MULTISPEED_LATCH
    cad = playroutine_cadence(data)
    assert cad.source is TriggerSource.CIA_TIMER
    assert cad.cycles_per_call == _MULTISPEED_LATCH + 1
