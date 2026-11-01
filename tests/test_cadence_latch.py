"""Tests for the static-latch cadence path."""

import pytest

from pysidtracker import PAL_CLOCK_HZ, cadence_from_latch
from pysidtracker.cadence import TriggerSource


def test_cia_latch():
    cad = cadence_from_latch(0x5BF9)
    assert cad.cycles_per_call == 0x5BF9 + 1
    assert cad.source is TriggerSource.CIA_TIMER
    assert cad.latch == 0x5BF9
    assert cad.clock_hz == PAL_CLOCK_HZ


def test_bad_clock():
    with pytest.raises(ValueError):
        cadence_from_latch(1000, "SECAM")
