"""Render a SID player (or write stream) through an emulated SID to WAV/samples.

The emulated SID is `pylibresidfp <https://pypi.org/project/pylibresidfp/>`_, a
core dependency. Any object exposing ``run(writes, until) -> int16 samples``
(``writes`` an ``(N, 3)`` array of absolute ``(cycle, reg, value)``) and a
``sampling_hz`` attribute may be passed as ``device`` instead.

:func:`render_player_wav` / :func:`render_player_samples` drive any
:class:`~pysidtracker.player.MemPlayer` (the shared per-frame player surface)
straight to WAV; the lower-level :func:`render_wav` / :func:`render_samples`
take a bare per-frame ``(reg, val)`` write stream. Each register write is
clocked individually at the same in-frame offset the register log uses, so
renders line up with :mod:`pysidtracker.reglog` output. This is the one WAV
render path every ``py*`` tracker shares.
"""

from __future__ import annotations

import wave
from array import array
from pathlib import Path
from itertools import islice
from typing import Iterable, Tuple

import numpy as _np
from tqdm import tqdm
from vagrant_moses.memmap import PAL

from .errors import AudioUnavailable
from .reglog import DEFAULT_WRITE_SPACING

CHIP_MODELS = ("6581", "8580")
DEFAULT_SAMPLING_HZ = 48000.0
FRAMES_PER_RUN = 256


def default_device(model: str, sampling_frequency=None, clock_frequency=None):
    """Construct the default pylibresidfp SID device for ``model``.

    ``model`` is one of :data:`CHIP_MODELS`. ``sampling_frequency`` selects the
    output rate (:data:`DEFAULT_SAMPLING_HZ` when falsy) and ``clock_frequency``
    the chip clock (the PAL system clock when falsy). Use
    :func:`device_sampling_frequency` to read back the device's rate.

    Raises :class:`~pysidtracker.errors.AudioUnavailable` if pylibresidfp is not
    installed and ``ValueError`` for an unknown model.
    """
    if model not in CHIP_MODELS:
        raise ValueError(f"chip model must be one of {CHIP_MODELS}")
    try:
        from pylibresidfp import SID
    except ImportError as exc:
        raise AudioUnavailable("pylibresidfp is required to render audio") from exc
    return SID(
        model,
        clock_hz=float(clock_frequency or PAL.clock_hz),
        sampling_hz=float(sampling_frequency or DEFAULT_SAMPLING_HZ),
        method="resample",
    )


def resolve_device(
    device=None, model: str = "6581", sampling_frequency=None, clock_frequency=None
):
    """Return a render device, validating ``model`` and building the default.

    Centralises the model-in-:data:`CHIP_MODELS` check plus default pylibresidfp
    device creation every leaf's ``audio.py`` duplicated. When ``device`` is
    given it is validated (``model``) and returned as-is; otherwise the default
    device is built at ``clock_frequency`` (which raises
    :class:`~pysidtracker.errors.AudioUnavailable` if pylibresidfp is missing).
    Raises ``ValueError`` for an unknown ``model``.
    """
    if model not in CHIP_MODELS:
        raise ValueError(f"chip model must be one of {CHIP_MODELS}")
    if device is not None:
        return device
    return default_device(model, sampling_frequency, clock_frequency)


def seconds_to_frames(
    seconds: float, cycles_per_frame: int, clock_frequency: float
) -> int:
    """Frames spanning ``seconds`` at ``cycles_per_frame`` / ``clock_frequency``.

    At least one frame is always returned (a non-looping render needs a bound).
    """
    return max(1, round(seconds / (cycles_per_frame / clock_frequency)))


def frames_to_seconds(
    frames: int, cycles_per_frame: int, clock_frequency: float
) -> float:
    """Seconds spanned by ``frames`` at ``cycles_per_frame`` / ``clock_frequency``."""
    return frames * cycles_per_frame / clock_frequency


def device_sampling_frequency(device) -> float:
    """The output sampling frequency (Hz) of a rendered ``device``."""
    return float(device.sampling_hz)


def render_samples(
    frame_iter: Iterable[Iterable[Tuple[int, int]]],
    *,
    model: str = "8580",
    sampling_frequency=None,
    cycles_per_frame: int,
    clock_frequency: float,
    write_spacing: int = DEFAULT_WRITE_SPACING,
    device=None,
):
    """Render a per-frame ``(reg, val)`` write stream on an emulated SID.

    ``frame_iter`` yields, per frame, an iterable of ``(reg, val)`` SID writes.
    Each write is applied then the SID is clocked ``write_spacing`` cycles; any
    remaining cycles of the frame are clocked after the frame's last write.
    Returns signed-16-bit mono samples as a ``numpy.ndarray`` (int16) when numpy
    is present, else a ``list``. ``device`` overrides the default pylibresidfp SID.

    Raises :class:`~pysidtracker.errors.AudioUnavailable` if the default backend
    is needed and pylibresidfp is missing, and ``ValueError`` for an unknown model.
    """
    if model not in CHIP_MODELS:
        raise ValueError(f"chip model must be one of {CHIP_MODELS}")
    if device is None:
        device = default_device(model, sampling_frequency, clock_frequency)
    frames = iter(frame_iter)
    base = int(getattr(device, "cycle", 0))
    out = []
    while chunk := list(islice(frames, FRAMES_PER_RUN)):
        rows = [_np.asarray(list(w), dtype=_np.int64).reshape(-1, 2) for w in chunk]
        count = _np.fromiter((len(r) for r in rows), _np.int64, len(rows))
        ends = base + _np.cumsum(_np.maximum(cycles_per_frame, count * write_spacing))
        starts = _np.concatenate(([base], ends[:-1]))
        slot = _np.arange(count.sum()) - _np.repeat(_np.cumsum(count) - count, count)
        cycles = _np.repeat(starts, count) + slot * write_spacing
        writes = _np.column_stack([cycles, _np.concatenate(rows)])
        out.append(device.run(writes, int(ends[-1])))
        base = int(ends[-1])
    return _np.concatenate(out).astype(_np.int16) if out else _np.empty(0, _np.int16)


def write_wav(dst, samples, sampling_frequency: float) -> None:
    """Write signed-16-bit mono ``samples`` as a WAV file."""
    if not isinstance(samples, array):
        samples = array("h", (int(s) for s in samples))
    with wave.open(str(dst), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(round(sampling_frequency))
        out.writeframes(samples.tobytes())


def render_wav(
    frame_iter: Iterable[Iterable[Tuple[int, int]]],
    dst,
    *,
    model: str = "8580",
    sampling_frequency=None,
    cycles_per_frame: int,
    clock_frequency: float,
    write_spacing: int = DEFAULT_WRITE_SPACING,
    device=None,
) -> Path:
    """Render ``frame_iter`` to a WAV file at ``dst``; return the path written.

    Keyword options are those of :func:`render_samples`.
    """
    if device is None:
        device = default_device(model, sampling_frequency, clock_frequency)
    samples = render_samples(
        frame_iter,
        model=model,
        sampling_frequency=sampling_frequency,
        cycles_per_frame=cycles_per_frame,
        clock_frequency=clock_frequency,
        write_spacing=write_spacing,
        device=device,
    )
    write_wav(dst, samples, device_sampling_frequency(device))
    return Path(dst)


def render_player_samples(
    player,
    *,
    seconds: float = 60.0,
    model: str = "8580",
    sampling_frequency=None,
    device=None,
    cycles_per_frame: "int | None" = None,
    clock_frequency: "float | None" = None,
    write_spacing: int = DEFAULT_WRITE_SPACING,
    progress: bool = False,
):
    """Render ``seconds`` of a :class:`~pysidtracker.player.MemPlayer` to samples.

    Drives ``player.play_frame()`` for as many frames as span ``seconds`` and
    renders them through :func:`render_samples`. ``cycles_per_frame`` defaults to
    the player's own play-routine cadence (its ``cycles_per_frame`` attribute)
    or the PAL video frame; ``clock_frequency`` defaults to the device's clock or
    the PAL system clock, and a default device is built at it. ``progress``
    shows a tqdm bar over the frames. Returns ``(samples, sampling_frequency)``.
    """
    device = resolve_device(device, model, sampling_frequency, clock_frequency)
    if cycles_per_frame is None:
        cycles_per_frame = getattr(player, "cycles_per_frame", None) or (
            PAL.cycles_per_frame
        )
    if clock_frequency is None:
        clock_frequency = float(getattr(device, "clock_hz", PAL.clock_hz))
    nframes = seconds_to_frames(seconds, cycles_per_frame, clock_frequency)
    frames = (
        player.play_frame()
        for _ in tqdm(range(nframes), desc="render", unit="frame", disable=not progress)
    )
    samples = render_samples(
        frames,
        model=model,
        sampling_frequency=sampling_frequency,
        cycles_per_frame=cycles_per_frame,
        clock_frequency=clock_frequency,
        write_spacing=write_spacing,
        device=device,
    )
    return samples, device_sampling_frequency(device)


def render_player_wav(player, dst, *, seconds: float = 60.0, **options) -> Path:
    """Render ``seconds`` of a :class:`~pysidtracker.player.MemPlayer` to WAV.

    Keyword options are those of :func:`render_player_samples`. Returns the path
    written. This is the shared player-to-WAV entry point for every ``py*``
    tracker -- construct the format's player and hand it here.
    """
    samples, sampling_frequency = render_player_samples(
        player, seconds=seconds, **options
    )
    write_wav(dst, samples, sampling_frequency)
    return Path(dst)
