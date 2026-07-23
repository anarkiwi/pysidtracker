"""Native (pure-Python) exomizer decrunch for packed SID images.

Packed tunes ship an exomizer-crunched payload that only becomes the real C64
image after a decruncher runs. :func:`pysidtracker.detect.run_init` materialises
that by emulating the tune's *entire* init routine; this module offers the
targeted alternative: run only the exomizer decruncher (via `pydexomizer
<https://pypi.org/project/pydexomizer/>`_, a core dependency) to unpack the
image without executing the tune's own init.

:func:`native_decrunch` tries the self-extracting (``sfx``) format first, then
exomizer's ``mem`` format (auto-detecting crunch direction), and returns the
unpacked :class:`~pysidtracker.image.SidImage` on success, or ``None`` when the
image is not exomizer-packed (or cannot be decrunched natively). It never
raises for an unrecognised image, so it is safe as a first-try step before the
emulated-init fallback.
"""

from __future__ import annotations

from typing import Optional, Union

from .image import SidImage

try:  # pydexomizer is a core dependency (like jennings).
    import pydexomizer as _pdx
except ImportError:  # pragma: no cover - core dep, present in practice
    _pdx = None


def _as_image(image_or_bytes: Union[SidImage, bytes, bytearray]) -> SidImage:
    if isinstance(image_or_bytes, SidImage):
        return image_or_bytes
    return SidImage.from_bytes(bytes(image_or_bytes))


def _packed_prg(image: SidImage) -> bytes:
    """The loaded image as a PRG (2-byte LE load address + payload)."""
    load = image.load
    return bytes([load & 0xFF, load >> 8]) + bytes(image.image)


# Cap on emulated instructions for the sfx stub: a real decrunch of a SID-sized
# image converges well within this, while a non-exomizer image (wrong/looping
# entry) fails fast instead of grinding to pydexomizer's 8M default.
_SFX_MAX_STEPS = 2_000_000


def _attempt(prg: bytes, load: int):
    """Try each exomizer format; return ``(start, data)`` or ``None``."""
    if _pdx is None:  # pragma: no cover - core dep present in practice
        return None
    for call in (
        lambda: _pdx.decrunch_sfx(prg, max_steps=_SFX_MAX_STEPS),
        lambda: _pdx.decrunch_sfx(prg, entry=load, max_steps=_SFX_MAX_STEPS),
        lambda: _pdx.decrunch_mem_auto(prg),
    ):
        try:
            result = call()
        except Exception:  # pylint: disable=broad-except
            continue
        data = getattr(result, "data", None)
        if data:
            return result.start, bytes(data)
    return None


def native_decrunch(
    image_or_bytes: Union[SidImage, bytes, bytearray],
) -> Optional[SidImage]:
    """Natively decrunch an exomizer-packed image, or ``None`` if not packed.

    Runs only the exomizer decruncher (never the tune's init) on the loaded
    image. On success returns a fresh :class:`~pysidtracker.image.SidImage` with
    the decrunched bytes placed at their load address, carrying the original
    header (so ``init``/``play`` addresses remain available). Returns ``None``
    for any image that is not exomizer-packed; it never raises for that case.
    """
    image = _as_image(image_or_bytes)
    unpacked = _attempt(_packed_prg(image), image.load)
    if unpacked is None:
        return None
    start, data = unpacked
    if not data:
        return None
    return SidImage._place(  # pylint: disable=protected-access
        data, start, image.header, image.container
    )


__all__ = ["native_decrunch"]
