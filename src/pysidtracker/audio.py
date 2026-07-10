"""Render a per-frame SID write stream through an emulated SID to WAV/samples.

The emulated SID is `pyresidfp <https://pypi.org/project/pyresidfp/>`_ (install
the ``audio`` extra). Any object exposing ``write_register(reg, value)``,
``clock(timedelta) -> samples`` and a ``sampling_frequency`` attribute may be
passed as ``device`` instead (e.g. a test double or a different emulator).

Each register write is clocked individually at the same in-frame offset the
register log uses, so renders line up with :mod:`pysidtracker.reglog` output.
This consolidates the per-package ``audio.py`` (pygoattracker,
pyfuturecomposer, pymusicassembler).
"""

from __future__ import annotations

import wave
from array import array
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Tuple

from .errors import AudioUnavailable
from .reglog import DEFAULT_WRITE_SPACING

try:  # numpy is optional; a stdlib ``array`` fallback is used without it.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only without numpy
    _np = None

CHIP_MODELS = ("6581", "8580")


def default_device(model: str, sampling_frequency=None, clock_frequency=None):
    """Construct the default pyresidfp SID device for ``model``.

    ``model`` is one of :data:`CHIP_MODELS`. ``sampling_frequency`` selects the
    output rate (pyresidfp's own default when falsy). ``clock_frequency`` is
    accepted for signature symmetry with the render helpers and forwarded to
    pyresidfp when it supports it; it is optional and ignored otherwise. Use
    :func:`device_sampling_frequency` to read back the device's actual rate.

    Raises :class:`~pysidtracker.errors.AudioUnavailable` if pyresidfp is not
    installed and ``ValueError`` for an unknown model.
    """
    if model not in CHIP_MODELS:
        raise ValueError(f"chip model must be one of {CHIP_MODELS}")
    try:
        from pyresidfp import SoundInterfaceDevice
        from pyresidfp.sound_interface_device import ChipModel
    except ImportError as exc:  # pragma: no cover - optional audio extra
        raise AudioUnavailable(
            "pyresidfp is required to render audio; "
            "install with: pip install pysidtracker[audio]"
        ) from exc
    chip = {"6581": ChipModel.MOS6581, "8580": ChipModel.MOS8580}[model]
    kwargs = {"model": chip}
    if sampling_frequency:
        kwargs["sampling_frequency"] = float(sampling_frequency)
    if clock_frequency:
        kwargs["clock_frequency"] = float(clock_frequency)
    try:
        return SoundInterfaceDevice(**kwargs)
    except TypeError:  # pragma: no cover - older pyresidfp without clock kwarg
        kwargs.pop("clock_frequency", None)
        return SoundInterfaceDevice(**kwargs)


# Back-compat private alias for any internal caller that imported the old name.
_default_device = default_device


def resolve_device(device=None, model: str = "6581", sampling_frequency=None):
    """Return a render device, validating ``model`` and building the default.

    Centralises the model-in-:data:`CHIP_MODELS` check plus default pyresidfp
    device creation every leaf's ``audio.py`` duplicated. When ``device`` is
    given it is validated (``model``) and returned as-is; otherwise the default
    device is built (which raises :class:`~pysidtracker.errors.AudioUnavailable`
    if pyresidfp is missing). Raises ``ValueError`` for an unknown ``model``.
    """
    if model not in CHIP_MODELS:
        raise ValueError(f"chip model must be one of {CHIP_MODELS}")
    if device is not None:
        return device
    return default_device(model, sampling_frequency)


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
    return float(device.sampling_frequency)


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
    is present, else a ``list``. ``device`` overrides the default pyresidfp SID.

    Raises :class:`~pysidtracker.errors.AudioUnavailable` if the default backend
    is needed and pyresidfp is missing, and ``ValueError`` for an unknown model.
    """
    if model not in CHIP_MODELS:
        raise ValueError(f"chip model must be one of {CHIP_MODELS}")
    if device is None:
        device = default_device(model, sampling_frequency, clock_frequency)
    write_q = write_spacing / clock_frequency
    frame_seconds = cycles_per_frame / clock_frequency
    samples = array("h")
    for writes in frame_iter:
        remainder = frame_seconds
        for reg, val in writes:
            device.write_register(reg, val)
            samples.extend(device.clock(timedelta(seconds=write_q)))
            remainder -= write_q
        if remainder > 0:
            samples.extend(device.clock(timedelta(seconds=remainder)))
    if _np is not None:
        return _np.frombuffer(samples.tobytes(), dtype=_np.int16)
    return samples.tolist()


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
