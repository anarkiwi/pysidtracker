"""Tests for the pyresidfp WAV/sample render (audio extra).

``pyresidfp`` is a dev/test dependency, so the real-backend tests always run;
the "no backend" fallback tests hide ``pyresidfp`` from ``sys.modules`` to cover
that path too -- neither branch is skipped.
"""

import sys
import wave
from datetime import timedelta

import pytest

from pysidtracker import (
    PAL_CLOCK_HZ,
    PAL_CYCLES_PER_FRAME,
    default_device,
    render_samples,
    render_wav,
)
from pysidtracker import audio as audio_mod
from pysidtracker.errors import AudioUnavailable


class FakeSID:
    """A minimal SID device: one sample per clock call."""

    def __init__(self, sampling_frequency=44100):
        self.sampling_frequency = sampling_frequency
        self.writes = []
        self.clock_seconds = []

    def write_register(self, reg, val):
        self.writes.append((reg, val))

    def clock(self, delta: timedelta):
        self.clock_seconds.append(delta.total_seconds())
        return [len(self.writes) & 0x7FFF]


def _frames():
    return [[(0, 0x11), (1, 0x22)], [(4, 0x33)]]


def test_render_samples_clocking():
    dev = FakeSID()
    samples = render_samples(
        _frames(),
        cycles_per_frame=PAL_CYCLES_PER_FRAME,
        clock_frequency=PAL_CLOCK_HZ,
        device=dev,
    )
    assert dev.writes == [(0, 0x11), (1, 0x22), (4, 0x33)]
    # 3 per-write clocks + one remainder clock per frame (2 frames) = 5 samples.
    assert len(samples) == 5
    assert len(dev.clock_seconds) == 5


def test_render_samples_bad_model():
    with pytest.raises(ValueError):
        render_samples(
            _frames(),
            model="bogus",
            cycles_per_frame=PAL_CYCLES_PER_FRAME,
            clock_frequency=PAL_CLOCK_HZ,
            device=FakeSID(),
        )


def test_render_wav(tmp_path):
    dev = FakeSID(sampling_frequency=22050)
    dst = tmp_path / "out.wav"
    path = render_wav(
        _frames(),
        dst,
        cycles_per_frame=PAL_CYCLES_PER_FRAME,
        clock_frequency=PAL_CLOCK_HZ,
        device=dev,
    )
    assert path == dst
    with wave.open(str(dst), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 22050
        assert wav.getnframes() == 5


def test_default_device_requires_pyresidfp(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyresidfp", None)
    with pytest.raises(AudioUnavailable):
        render_samples(
            _frames(),
            cycles_per_frame=PAL_CYCLES_PER_FRAME,
            clock_frequency=PAL_CLOCK_HZ,
        )


def test_default_device_public_accessor_requires_pyresidfp(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyresidfp", None)
    with pytest.raises(AudioUnavailable):
        default_device("8580", 44100, PAL_CLOCK_HZ)


def test_default_device_bad_model():
    with pytest.raises(ValueError):
        default_device("bogus")


def test_default_device_private_alias_is_public():
    # Back-compat: the old private name aliases the public accessor.
    assert audio_mod._default_device is audio_mod.default_device


def test_device_sampling_frequency_accessor():
    dev = FakeSID(sampling_frequency=48000)
    assert audio_mod.device_sampling_frequency(dev) == 48000.0


def test_default_device_returns_device_with_sampling_frequency():
    dev = default_device("8580", 44100, PAL_CLOCK_HZ)
    assert audio_mod.device_sampling_frequency(dev) > 0


def test_render_with_real_pyresidfp(tmp_path):
    dst = render_wav(
        _frames(),
        tmp_path / "real.wav",
        model="8580",
        cycles_per_frame=PAL_CYCLES_PER_FRAME,
        clock_frequency=PAL_CLOCK_HZ,
    )
    with wave.open(str(dst), "rb") as wav:
        assert wav.getnframes() > 0
