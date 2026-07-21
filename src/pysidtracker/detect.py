"""Detect compressed / packed / relocating playroutines in a first parse.

A ``.sid`` header advertises a load address, an init address and a subtune
count, but for many real tunes these describe a *loader* or *packer*, not the
playroutine: the actual song tables only exist after the init routine unpacks or
relocates them into memory. The header is therefore untrustworthy for locating
data.

The shared strategy (generalised from pygoattracker's ``.sid`` decompiler):

1. **Static recognition.** Ask the format's ``recognize`` callback to find its
   signature/anchor in the freshly loaded image. If it succeeds the tune is
   :attr:`PlayroutineKind.DIRECT` -- a plain direct-load image.
2. **Emulated init.** Otherwise the playroutine is packed/relocating: run its
   init routine in a 6502 emulator so the data lands where it really goes, then
   recognise again. Whether init had to *expand* memory (decompression) or only
   *move*/rewrite the existing image classifies it as
   :attr:`PlayroutineKind.PACKED` vs :attr:`PlayroutineKind.RELOCATED`.
3. If recognition still fails, :attr:`PlayroutineKind.UNKNOWN`.

Each parser supplies only the cheap ``recognize`` predicate; this module owns
the untrustworthy-header handling so every format gets it identically.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from .emu import run_to_rts, wire_mpu
from .errors import SidParseError
from .header import SidHeader
from .image import MEM_SIZE, SidImage

JMP_ABS = 0x4C

# recognize(image) -> anchor (any truthy value) if the format is found, else None.
Recognizer = Callable[[SidImage], object]


class PlayroutineKind(enum.Enum):
    """How a tune's playroutine presents its data in the loaded image."""

    DIRECT = "direct"  # data found statically, as loaded
    RELOCATED = "relocated"  # found only after init moved/rewrote the image
    PACKED = "packed"  # found only after init expanded (decompressed) memory
    UNKNOWN = "unknown"  # not recognised, even after running init


@dataclass
class Detection:
    """Outcome of :func:`detect_playroutine`.

    Attributes:
      kind: the :class:`PlayroutineKind`.
      ran_init: whether the init routine had to be emulated.
      anchor: whatever the ``recognize`` callback returned (``None`` if
        unrecognised) -- often the address/offset it found, reusable by the
        caller's full parse.
      changed_inside: bytes within the originally loaded region that init
        altered (self-modifying / in-place unpacking).
      written_outside: bytes outside the originally loaded region that init
        wrote (relocation targets / decompression output).
    """

    kind: PlayroutineKind
    ran_init: bool
    anchor: object = None
    changed_inside: int = 0
    written_outside: int = 0

    @property
    def recognised(self) -> bool:
        return self.kind is not PlayroutineKind.UNKNOWN

    @property
    def trustworthy_header(self) -> bool:
        """True when the header described the data directly (no init needed)."""
        return self.kind is PlayroutineKind.DIRECT


def detect_playroutine(
    image: SidImage,
    recognize: Recognizer,
    *,
    init: bool = True,
    subtune: int = 0,
    native: bool = False,
) -> Detection:
    """Classify how ``image`` presents its playroutine data.

    ``recognize`` inspects the (possibly unpacked) image and returns a truthy
    anchor when it finds the format, else ``None``. When static recognition
    fails and ``init`` is true, the init routine is emulated (requires the core
    ``py65`` dependency) and recognition retried.

    When ``native`` is true, an exomizer-packed image is first decrunched
    natively (:func:`pysidtracker.decrunch.native_decrunch`, no init emulated);
    if that unpacks a recognisable image it is reported as
    :attr:`PlayroutineKind.PACKED` with ``ran_init=False``. Native decrunch is
    opt-in and only a first try -- it falls back to the emulated-init path
    below when the image is not exomizer-packed or stays unrecognised, so the
    default behaviour is unchanged.

    Raises :class:`EmulatorUnavailable` if emulation is needed but ``py65`` is
    not installed.
    """
    anchor = recognize(image)
    if anchor:
        return Detection(PlayroutineKind.DIRECT, ran_init=False, anchor=anchor)

    if native:
        native_hit = _try_native_decrunch(image, recognize)
        if native_hit is not None:
            return native_hit

    if not init:
        return Detection(PlayroutineKind.UNKNOWN, ran_init=False)

    orig_load, orig_end = image.load, image.end
    snapshot = bytes(image.mem[orig_load:orig_end])
    run_init(image, subtune=subtune)
    image.end = MEM_SIZE  # init may have scattered data anywhere

    changed_inside = _count_changed(image.mem, orig_load, snapshot)
    written_outside = _count_written_outside(image.mem, orig_load, orig_end)

    anchor = recognize(image)
    if not anchor:
        kind = PlayroutineKind.UNKNOWN
    elif written_outside > changed_inside:
        kind = PlayroutineKind.PACKED
    else:
        kind = PlayroutineKind.RELOCATED
    return Detection(
        kind,
        ran_init=True,
        anchor=anchor,
        changed_inside=changed_inside,
        written_outside=written_outside,
    )


# The 6502 stack page is scratch space the init run (and our return-address
# stub) churns; it is never song data, so it is excluded from the diff metrics.
_STACK_LO = 0x0100
_STACK_HI = 0x0200


def _try_native_decrunch(image: SidImage, recognize: Recognizer):
    """Native exomizer decrunch first-try; a :class:`Detection` or ``None``.

    Returns a ``PACKED`` detection (``ran_init=False``) when native decrunch
    unpacks an image the ``recognize`` callback then accepts, else ``None`` so
    the caller falls back to the emulated-init path.
    """
    from .decrunch import native_decrunch

    unpacked = native_decrunch(image)
    if unpacked is None:
        return None
    anchor = recognize(unpacked)
    if not anchor:
        return None
    written = sum(1 for value in unpacked.mem if value)
    return Detection(
        PlayroutineKind.PACKED,
        ran_init=False,
        anchor=anchor,
        written_outside=written,
    )


def _count_changed(mem: bytearray, load: int, snapshot: bytes) -> int:
    return sum(1 for i, b in enumerate(snapshot) if mem[load + i] != b)


def _count_written_outside(mem: bytearray, load: int, end: int) -> int:
    written = 0
    for addr, value in enumerate(mem):
        if not value:
            continue
        if load <= addr < end:
            continue
        if _STACK_LO <= addr < _STACK_HI:
            continue
        written += 1
    return written


def resolve_entry_points(
    header: Optional[SidHeader],
    load: int,
    default_init: Optional[int] = None,
    default_play: Optional[int] = None,
) -> Tuple[int, int]:
    """Resolve a tune's ``(init, play)`` addresses with the SID zero fallback.

    When ``header`` is present, an ``init``/``play`` field of ``0`` means "use
    the load address" (the documented SID convention). For a bare ``.prg``
    (``header is None``) the ``default_init``/``default_play`` are used, each
    falling back to ``load`` when not given.
    """
    if header is not None:
        init = header.init_address or load
        play = header.play_address or load
    else:
        init = load if default_init is None else default_init
        play = load if default_play is None else default_play
    return init, play


def is_jmp_vector(image: SidImage, addr: int) -> bool:
    """True if ``addr`` holds a ``JMP abs`` ($4C) whose target is in the image."""
    if not image.contains(addr) or image.peek(addr) != JMP_ABS:
        return False
    target = image.peek(addr + 1) | (image.peek(addr + 2) << 8)
    return image.contains(target)


def run_init(image: SidImage, subtune: int = 0, max_cycles: int = 8_000_000) -> None:
    """Run the tune's init routine in a 6502 emulator so data lands in place.

    Mutates ``image.mem`` directly. ``subtune`` is passed to init in the
    accumulator (the SID calling convention). Requires the optional ``py65``
    dependency; raises :class:`EmulatorUnavailable` if it is missing and
    :class:`SidParseError` if the image has no init address to call.
    """
    if image.header is None:
        raise SidParseError("cannot run init: image has no SID header")
    init_address = image.header.init_address or image.header.real_load_address
    mpu, mem = wire_mpu(image.mem)
    run_to_rts(mpu, mem, init_address, subtune, max_cycles)
