"""Tests for the static-latch cadence path."""

import pytest

from pysidtracker import PAL_CLOCK_HZ, PAL_CYCLES_PER_FRAME, cadence_from_latch
from pysidtracker.cadence import TriggerSource


def test_cia_latch():
    cad = cadence_from_latch(0x5BF9)
    assert cad.cycles_per_call == 0x5BF9 + 1
    assert cad.source is TriggerSource.CIA_TIMER
    assert cad.latch == 0x5BF9
    assert cad.clock_hz == PAL_CLOCK_HZ


def test_no_latch_falls_back_to_video():
    cad = cadence_from_latch(None)
    assert cad.cycles_per_call == PAL_CYCLES_PER_FRAME
    assert cad.source is TriggerSource.PAL_VIDEO


def test_low_latch_ignored():
    # A lo-byte-only artefact (< 0x100) is not a real play cadence.
    cad = cadence_from_latch(0x00FF)
    assert cad.source is TriggerSource.PAL_VIDEO


def test_ntsc():
    cad = cadence_from_latch(None, "NTSC")
    assert cad.source is TriggerSource.NTSC_VIDEO


def test_bad_clock():
    with pytest.raises(ValueError):
        cadence_from_latch(1000, "SECAM")
