"""Tests for the pylibresidfp WAV/sample render.

``pylibresidfp`` is a core dependency, so the real-backend tests always run; the
"no backend" tests hide ``pylibresidfp`` from ``sys.modules`` to cover the
broken-install error path too -- neither branch is skipped.
"""

import sys
import wave

import numpy as np
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
    """A minimal run-based SID device: one sample per 1000 cycles."""

    def __init__(self, sampling_hz=44100):
        self.sampling_hz = sampling_hz
        self.cycle = 0
        self.writes = []
        self.runs = []

    def run(self, writes, until):
        self.writes.extend(map(tuple, writes))
        self.runs.append(until)
        count = (until - self.cycle) // 1000
        self.cycle = until
        return np.zeros(count, dtype=np.int16)


def _frames():
    return [[(0, 0x11), (1, 0x22)], [(4, 0x33)]]


def test_render_is_deterministic():
    kwargs = dict(
        model="8580",
        cycles_per_frame=PAL_CYCLES_PER_FRAME,
        clock_frequency=PAL_CLOCK_HZ,
    )
    frames = [[(24, 15), (4, 0x21), (1, 0x30)], [(4, 0x20)]] * 3
    assert np.array_equal(
        render_samples(frames, **kwargs), render_samples(frames, **kwargs)
    )


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
    dev = FakeSID(sampling_hz=22050)
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
        assert wav.getnframes() == 2 * PAL_CYCLES_PER_FRAME // 1000


def test_default_device_requires_pylibresidfp(monkeypatch):
    monkeypatch.setitem(sys.modules, "pylibresidfp", None)
    with pytest.raises(AudioUnavailable):
        render_samples(
            _frames(),
            cycles_per_frame=PAL_CYCLES_PER_FRAME,
            clock_frequency=PAL_CLOCK_HZ,
        )


def test_default_device_bad_model():
    with pytest.raises(ValueError):
        default_device("bogus")


def test_render_with_real_pylibresidfp(tmp_path):
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


def test_render_player_wav_writes_file(tmp_path):
    dst = tmp_path / "player.wav"
    out = audio_mod.render_player_wav(
        _FakePlayer(), dst, seconds=0.02, device=FakeSID()
    )
    assert out == dst
    with wave.open(str(dst), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getnframes() > 0


def test_render_player_samples_builds_the_device_at_the_given_clock(monkeypatch):
    built = {}

    def device(model, sampling_frequency=None, clock_frequency=None):
        built.update(model=model, rate=sampling_frequency, clock=clock_frequency)
        return FakeSID()

    monkeypatch.setattr(audio_mod, "default_device", device)
    audio_mod.render_player_samples(
        _FakePlayer(), seconds=0.02, model="6581", clock_frequency=1022727.0
    )
    assert built == {"model": "6581", "rate": None, "clock": 1022727.0}
