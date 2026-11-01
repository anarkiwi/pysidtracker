"""Tests for the audio device-resolution and timing helpers."""

import pytest

from pysidtracker import (
    PAL_CLOCK_HZ,
    PAL_CYCLES_PER_FRAME,
    frames_to_seconds,
    resolve_device,
    seconds_to_frames,
)


class _FakeSID:
    sampling_hz = 44100


def test_resolve_device_bad_model():
    with pytest.raises(ValueError, match="chip model"):
        resolve_device(_FakeSID(), model="9999")


def test_frames_to_seconds_round_trip():
    seconds = frames_to_seconds(50, PAL_CYCLES_PER_FRAME, PAL_CLOCK_HZ)
    assert seconds_to_frames(seconds, PAL_CYCLES_PER_FRAME, PAL_CLOCK_HZ) == 50
