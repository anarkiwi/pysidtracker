"""Tests for the pyresidfp WAV/sample render.

``pyresidfp`` is a core dependency, so the real-backend tests always run; the
"no backend" tests hide ``pyresidfp`` from ``sys.modules`` to cover the
broken-install error path too -- neither branch is skipped.
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


class _FakePlayer:
    """A minimal MemPlayer stand-in: one changed register per frame."""

    def __init__(self, cycles_per_frame=None):
        self.calls = 0
        if cycles_per_frame is not None:
            self.cycles_per_frame = cycles_per_frame

    def play_frame(self):
        self.calls += 1
        return [(0, self.calls & 0xFF)]


def test_render_player_samples_uses_player_cadence():
    dev = FakeSID()
    player = _FakePlayer(cycles_per_frame=PAL_CYCLES_PER_FRAME)
    samples, sampling_frequency = audio_mod.render_player_samples(
        player, seconds=0.05, device=dev
    )
    assert sampling_frequency == dev.sampling_frequency
    # One frame per play_frame call, each contributing exactly one write.
    assert player.calls > 0
    assert dev.writes == [(0, (i + 1) & 0xFF) for i in range(player.calls)]
    assert len(samples) == len(dev.clock_seconds)


def test_render_player_samples_default_cadence_without_attr():
    # A player without a cycles_per_frame attribute falls back to the PAL frame.
    dev = FakeSID()
    player = _FakePlayer()
    audio_mod.render_player_samples(player, seconds=0.05, device=dev)
    assert player.calls > 0


def test_render_player_wav_writes_file(tmp_path):
    dst = tmp_path / "player.wav"
    out = audio_mod.render_player_wav(
        _FakePlayer(), dst, seconds=0.02, device=FakeSID()
    )
    assert out == dst
    with wave.open(str(dst), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getnframes() > 0
