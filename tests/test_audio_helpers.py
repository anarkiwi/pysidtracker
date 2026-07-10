"""Tests for the audio device-resolution and timing helpers."""

import pytest

from pysidtracker import (
    PAL_CLOCK_HZ,
    PAL_CYCLES_PER_FRAME,
    frames_to_seconds,
    resolve_device,
    seconds_to_frames,
)
from pysidtracker.errors import AudioUnavailable

try:
    import pyresidfp  # noqa: F401  # pylint: disable=unused-import

    HAVE_PYRESIDFP = True
except ImportError:
    HAVE_PYRESIDFP = False


class _FakeSID:
    sampling_frequency = 44100


def test_resolve_device_passthrough():
    dev = _FakeSID()
    assert resolve_device(dev, model="6581") is dev


def test_resolve_device_bad_model():
    with pytest.raises(ValueError, match="chip model"):
        resolve_device(_FakeSID(), model="9999")


@pytest.mark.skipif(HAVE_PYRESIDFP, reason="pyresidfp installed")
def test_resolve_device_default_unavailable():
    with pytest.raises(AudioUnavailable):
        resolve_device(None, model="6581")


def test_seconds_to_frames():
    frames = seconds_to_frames(1.0, PAL_CYCLES_PER_FRAME, PAL_CLOCK_HZ)
    assert frames == round(PAL_CLOCK_HZ / PAL_CYCLES_PER_FRAME)
    assert seconds_to_frames(0.0, PAL_CYCLES_PER_FRAME, PAL_CLOCK_HZ) == 1


def test_frames_to_seconds_round_trip():
    seconds = frames_to_seconds(50, PAL_CYCLES_PER_FRAME, PAL_CLOCK_HZ)
    assert seconds_to_frames(seconds, PAL_CYCLES_PER_FRAME, PAL_CLOCK_HZ) == 50
