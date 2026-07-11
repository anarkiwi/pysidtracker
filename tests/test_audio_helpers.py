"""Tests for the audio device-resolution and timing helpers."""

import sys

import pytest

from pysidtracker import (
    PAL_CLOCK_HZ,
    PAL_CYCLES_PER_FRAME,
    frames_to_seconds,
    resolve_device,
    seconds_to_frames,
)
from pysidtracker.errors import AudioUnavailable


class _FakeSID:
    sampling_frequency = 44100


def test_resolve_device_passthrough():
    dev = _FakeSID()
    assert resolve_device(dev, model="6581") is dev


def test_resolve_device_bad_model():
    with pytest.raises(ValueError, match="chip model"):
        resolve_device(_FakeSID(), model="9999")


def test_resolve_device_default_unavailable(monkeypatch):
    """Without a pyresidfp backend, the default device is unavailable.

    Simulates the missing extra by hiding ``pyresidfp`` from ``sys.modules``, so
    the fallback path is covered whether or not the extra is installed.
    """
    monkeypatch.setitem(sys.modules, "pyresidfp", None)
    with pytest.raises(AudioUnavailable):
        resolve_device(None, model="6581")


def test_seconds_to_frames():
    frames = seconds_to_frames(1.0, PAL_CYCLES_PER_FRAME, PAL_CLOCK_HZ)
    assert frames == round(PAL_CLOCK_HZ / PAL_CYCLES_PER_FRAME)
    assert seconds_to_frames(0.0, PAL_CYCLES_PER_FRAME, PAL_CLOCK_HZ) == 1


def test_frames_to_seconds_round_trip():
    seconds = frames_to_seconds(50, PAL_CYCLES_PER_FRAME, PAL_CLOCK_HZ)
    assert seconds_to_frames(seconds, PAL_CYCLES_PER_FRAME, PAL_CLOCK_HZ) == 50
