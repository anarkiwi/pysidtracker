"""Tests for the pylibresidfp WAV/sample render.

``pylibresidfp`` is a core dependency, so the real-backend tests always run; the
"no backend" tests hide ``pylibresidfp`` from ``sys.modules`` to cover the
broken-install error path too -- neither branch is skipped.
"""

import sys
import wave

import numpy as np
import pytest
from pylibresidfp import SID

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


def test_render_samples_places_writes_at_absolute_cycles():
    dev = FakeSID()
    spacing = 43
    samples = render_samples(
        _frames(),
        cycles_per_frame=PAL_CYCLES_PER_FRAME,
        clock_frequency=PAL_CLOCK_HZ,
        write_spacing=spacing,
        device=dev,
    )
    frame = PAL_CYCLES_PER_FRAME
    assert dev.writes == [
        (0, 0, 0x11),
        (spacing, 1, 0x22),
        (frame, 4, 0x33),
    ]
    assert dev.runs == [2 * frame]
    assert len(samples) == 2 * frame // 1000


def test_render_samples_overlong_frame_extends_time():
    dev = FakeSID()
    render_samples(
        [[(0, 1)] * 5, []],
        cycles_per_frame=100,
        clock_frequency=PAL_CLOCK_HZ,
        write_spacing=40,
        device=dev,
    )
    assert dev.runs == [300]
    assert [w[0] for w in dev.writes] == [0, 40, 80, 120, 160]


def test_render_samples_batches_frames_per_run():
    dev = FakeSID()
    nframes = audio_mod.FRAMES_PER_RUN * 2 + 1
    render_samples(
        [[(0, 1)]] * nframes,
        cycles_per_frame=1000,
        clock_frequency=PAL_CLOCK_HZ,
        device=dev,
    )
    assert len(dev.runs) == 3
    assert dev.runs[-1] == nframes * 1000


def test_render_samples_matches_manual_pylibresidfp_run():
    frames = [[(0, 0x0F), (4, 0x41), (1, 0x20)], [], [(4, 0x40)], [(24, 0x0F)] * 3]
    cpf, spacing = 19656, 43
    got = render_samples(
        frames,
        model="6581",
        sampling_frequency=44100,
        cycles_per_frame=cpf,
        clock_frequency=PAL_CLOCK_HZ,
        write_spacing=spacing,
    )
    sid = SID("6581", PAL_CLOCK_HZ, 44100.0, "resample")
    writes = [
        (f * cpf + i * spacing, reg, val)
        for f, frame in enumerate(frames)
        for i, (reg, val) in enumerate(frame)
    ]
    want = sid.run(writes, len(frames) * cpf)
    assert got.dtype == np.int16
    assert np.array_equal(got, want)
    assert np.any(got)


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


def test_default_device_rates():
    dev = default_device("8580")
    assert dev.sampling_hz == audio_mod.DEFAULT_SAMPLING_HZ
    assert dev.clock_hz == PAL_CLOCK_HZ
    dev = default_device("6581", 22050, 1022727.0)
    assert (dev.sampling_hz, dev.clock_hz) == (22050.0, 1022727.0)


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


def test_default_device_public_accessor_requires_pylibresidfp(monkeypatch):
    monkeypatch.setitem(sys.modules, "pylibresidfp", None)
    with pytest.raises(AudioUnavailable):
        default_device("8580", 44100, PAL_CLOCK_HZ)


def test_default_device_bad_model():
    with pytest.raises(ValueError):
        default_device("bogus")


def test_device_sampling_frequency_accessor():
    assert audio_mod.device_sampling_frequency(FakeSID(sampling_hz=48000)) == 48000.0


def test_default_device_returns_device_with_sampling_frequency():
    dev = default_device("8580", 44100, PAL_CLOCK_HZ)
    assert audio_mod.device_sampling_frequency(dev) > 0


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


def test_render_player_samples_uses_player_cadence():
    dev = FakeSID()
    player = _FakePlayer(cycles_per_frame=PAL_CYCLES_PER_FRAME)
    samples, sampling_frequency = audio_mod.render_player_samples(
        player, seconds=0.05, device=dev
    )
    assert sampling_frequency == dev.sampling_hz
    assert player.calls > 0
    assert [w[1:] for w in dev.writes] == [
        (0, (i + 1) & 0xFF) for i in range(player.calls)
    ]
    assert dev.runs[-1] == player.calls * PAL_CYCLES_PER_FRAME
    assert len(samples) == player.calls * PAL_CYCLES_PER_FRAME // 1000


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


def test_render_player_samples_shows_progress_over_the_frames(capsys):
    player = _FakePlayer()
    audio_mod.render_player_samples(
        player, seconds=0.1, device=FakeSID(), progress=True
    )
    assert f"{player.calls}/{player.calls}" in capsys.readouterr().err
