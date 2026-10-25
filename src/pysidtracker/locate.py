"""Ordered layout strategies: one decode per candidate, one emulated image, a rejection log.

A format supplies :class:`Strategy` objects that yield *candidate* layouts and
decode one candidate into the format's layout (or reject it).
:func:`locate` runs them in stage order -- :data:`STATIC` (the image as loaded),
:data:`NATIVE` (exomizer decrunched natively, no init), :data:`AFTER_INIT` (init
emulated once), :data:`SYS_UNPACK` (the BASIC ``SYS`` stub chain run on the KERNAL
host until it jumps into code it unpacked), :data:`RUN_PLAYER` (init plus a few
play calls) -- caching each derived image so no stage is run twice.
The first accepted candidate wins; its strategy name is recorded as the
``variant`` facet. When nothing is accepted the result carries the full
``(strategy, candidate, reason)`` rejection log rather than only the last error.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, NamedTuple, Optional, Sequence

from .detect import (
    Detection,
    PlayroutineKind,
    _count_changed,
    _count_written_outside,
    run_init,
)
from .emu import run_to_rts, wire_mpu
from .errors import KernalHostError, SidError, SidParseError
from .facets import Entry, Facet, inferred
from .image import MEM_SIZE, SidImage
from .poweron import psid_init_bank
from .registers import sid_model_for_flags

STATIC = "static"
NATIVE = "native-decrunch"
AFTER_INIT = "after-init"
SYS_UNPACK = "sys-unpack"
RUN_PLAYER = "run-player"
STAGES = (STATIC, NATIVE, AFTER_INIT, SYS_UNPACK, RUN_PLAYER)

VARIANT_FACET = "variant"

_EMULATED = (AFTER_INIT, SYS_UNPACK, RUN_PLAYER)

# candidates(image) -> iterable of candidate layouts (any objects).
Candidates = Callable[[SidImage], Iterable[Any]]
# decode(image, candidate) -> the layout; raise SidParseError (or return None) to reject.
Decoder = Callable[[SidImage, Any], Any]


@dataclass(frozen=True)
class Strategy:
    """One way a format can find its layout.

    ``candidates`` yields the layouts to try (default: a single ``None``
    candidate, for a decoder that finds its own anchor); ``decode`` turns one into
    the format's layout or rejects it.
    """

    name: str
    stage: str
    decode: Decoder
    candidates: Optional[Candidates] = None

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(
                f"strategy {self.name}: stage {self.stage!r} not in {STAGES}"
            )

    def propose(self, image: SidImage) -> Iterable[Any]:
        """The candidates for ``image``."""
        if self.candidates is None:
            return (None,)
        return self.candidates(image)


class Rejection(NamedTuple):
    """Why one candidate of one strategy was not accepted."""

    strategy: str
    candidate: Any
    reason: str


@dataclass
class Located:
    """Outcome of :func:`locate`.

    Attributes:
      layout: the accepted layout, or ``None`` when every candidate was rejected.
      strategy: name of the winning strategy (``None`` on failure).
      stage: stage the winner ran at (``None`` on failure).
      candidate: the accepted candidate.
      image: the image the layout was decoded from (the emulated one for later
        stages), so callers need not emulate again.
      rejections: every ``(strategy, candidate, reason)`` rejected on the way.
      detection: the :class:`~pysidtracker.detect.Detection` classification.
    """

    layout: Any = None
    strategy: Optional[str] = None
    stage: Optional[str] = None
    candidate: Any = None
    image: Optional[SidImage] = None
    rejections: List[Rejection] = field(default_factory=list)
    detection: Optional[Detection] = None

    @property
    def ok(self) -> bool:
        return self.strategy is not None

    @property
    def variant(self) -> Optional[Entry]:
        """The winning strategy as a ``variant`` facet :class:`~pysidtracker.facets.Entry`."""
        if self.strategy is None:
            return None
        return Entry(self.strategy, inferred(f"layout strategy {self.stage}"))

    def describe(self) -> str:
        """The rejection log, one ``strategy[candidate]: reason`` per line."""
        return "\n".join(
            f"{r.strategy}[{r.candidate!r}]: {r.reason}" for r in self.rejections
        )

    def require(self, error: type = SidParseError) -> Any:
        """The layout, or raise ``error`` carrying the full rejection log."""
        if not self.ok:
            raise error(f"no layout found: {self.describe() or 'no strategies'}")
        return self.layout


def variant_facet(strategies: Sequence[Strategy], kind: str = "layout") -> Facet:
    """A ``variant`` :class:`~pysidtracker.facets.Facet` whose domain is the strategy names."""
    return Facet(VARIANT_FACET, kind, domain=tuple(s.name for s in strategies))


class _Stages:
    """The derived images, each built at most once."""

    def __init__(
        self, image: SidImage, subtune: int, frames: int, in_place: bool
    ) -> None:
        self.static = image
        self._subtune = subtune
        self._frames = frames
        self._in_place = in_place
        self._images: dict = {}
        self.ran_init = False
        self.init_stats = (0, 0)  # (changed_inside, written_outside)
        self.native_written = 0

    @staticmethod
    def _copy(image: SidImage) -> SidImage:
        return SidImage(
            bytearray(image.mem),
            image.load,
            image.end,
            image.header,
            image.image,
            image.container,
        )

    def _native(self) -> SidImage:
        from .decrunch import native_decrunch

        unpacked = native_decrunch(self.static)
        if unpacked is None:
            raise SidParseError("not natively decrunchable")
        self.native_written = sum(1 for value in unpacked.mem if value)
        return unpacked

    def _after_init(self) -> SidImage:
        src = self.static
        img = src if self._in_place else self._copy(src)
        snapshot = bytes(src.mem[src.load : src.end])
        orig_load, orig_end = src.load, src.end
        run_init(img, subtune=self._subtune)
        img.end = MEM_SIZE
        self.ran_init = True
        self.init_stats = (
            _count_changed(img.mem, orig_load, snapshot),
            _count_written_outside(img.mem, orig_load, orig_end),
        )
        return img

    def _sys_unpack(self) -> SidImage:
        from .kernalhost import KernalHost

        with tempfile.TemporaryDirectory() as host_dir:
            pc, code = KernalHost(host_dir).run_to_unpacked(
                self.static.to_prg(), entry=self.static.load
            )
        return SidImage._place(  # pylint: disable=protected-access
            code, pc, self.static.header, self.static.container
        )

    def _run_player(self) -> SidImage:
        img = self._copy(self.get(AFTER_INIT))
        header = img.header
        if not header.play_address:
            raise SidParseError("no play address to run (IRQ-driven tune)")
        play = header.play_address
        mpu, mem = wire_mpu(img.mem, sid_model=sid_model_for_flags(header.flags))
        img.mem[1] = psid_init_bank(play)
        for _ in range(self._frames):
            run_to_rts(mpu, mem, play, 0, 1_000_000)
        return img

    def get(self, stage: str) -> SidImage:
        """The image for ``stage``; raises ``SidParseError``/``SidError`` if unavailable."""
        if stage == STATIC:
            return self.static
        if stage not in self._images:
            build = {
                NATIVE: self._native,
                AFTER_INIT: self._after_init,
                SYS_UNPACK: self._sys_unpack,
                RUN_PLAYER: self._run_player,
            }[stage]
            self._images[stage] = build()
        return self._images[stage]


def locate(
    image: SidImage,
    strategies: Sequence[Strategy],
    *,
    init: bool = True,
    subtune: int = 0,
    frames: int = 4,
    in_place: bool = False,
    strict: bool = False,
) -> Located:
    """Run ``strategies`` in stage order (static, after-init, run-player; ties keep
    the given order) and return the first accepted layout, or the rejection log.

    Each emulated stage is built once and shared by its strategies; a stage is not
    emulated at all when an earlier one already won, or when ``init`` is false (the
    stage's strategies are then logged as skipped). ``frames`` is the number of
    play calls of the run-player stage. ``in_place`` makes the after-init stage run
    init on ``image`` itself instead of a copy; ``strict`` re-raises a stage that
    cannot be built by emulation (e.g. no SID header to run init on) instead of logging it.

    :attr:`Located.detection` classifies the result like
    :func:`~pysidtracker.detect.detect_playroutine`: static is ``DIRECT``; a later
    stage is ``PACKED`` or ``RELOCATED`` by how much init wrote outside the loaded
    region; no layout is ``UNKNOWN``.
    """
    ordered = sorted(strategies, key=lambda s: STAGES.index(s.stage))
    stages = _Stages(image, subtune, frames, in_place)
    result = Located()
    for strategy in ordered:
        if strategy.stage in _EMULATED and not init:
            result.rejections.append(Rejection(strategy.name, None, "init disabled"))
            continue
        try:
            staged = stages.get(strategy.stage)
        except SidError as exc:
            if (strict and strategy.stage in _EMULATED) or not isinstance(
                exc, (SidParseError, KernalHostError)
            ):
                raise
            result.rejections.append(Rejection(strategy.name, None, str(exc)))
            continue
        for candidate, layout, reason in _attempt(strategy, staged):
            if layout is None:
                result.rejections.append(Rejection(strategy.name, candidate, reason))
                continue
            result.layout, result.candidate, result.image = layout, candidate, staged
            result.strategy, result.stage = strategy.name, strategy.stage
            result.detection = _classify(strategy.stage, stages, candidate)
            return result
    changed, written = stages.init_stats
    result.detection = Detection(
        PlayroutineKind.UNKNOWN,
        ran_init=stages.ran_init,
        changed_inside=changed,
        written_outside=written,
    )
    return result


def _attempt(strategy: Strategy, image: SidImage):
    """Yield ``(candidate, layout, reason)`` for each candidate; layout ``None`` = rejected."""
    try:
        candidates = iter(strategy.propose(image))
        while True:
            try:
                candidate = next(candidates)
            except StopIteration:
                return
            try:
                layout = strategy.decode(image, candidate)
            except SidParseError as exc:
                yield candidate, None, str(exc)
                continue
            if layout is None:
                yield candidate, None, "decoder rejected the candidate"
                continue
            yield candidate, layout, ""
            return
    except SidParseError as exc:
        yield None, None, f"candidates: {exc}"


def _classify(stage: str, stages: _Stages, candidate: Any) -> Detection:
    if stage == STATIC:
        return Detection(PlayroutineKind.DIRECT, ran_init=False, anchor=candidate)
    if stage == NATIVE:
        return Detection(
            PlayroutineKind.PACKED,
            ran_init=False,
            anchor=candidate,
            written_outside=stages.native_written,
        )
    if stage == SYS_UNPACK:
        return Detection(PlayroutineKind.PACKED, ran_init=False, anchor=candidate)
    changed, written = stages.init_stats
    kind = PlayroutineKind.PACKED if written > changed else PlayroutineKind.RELOCATED
    return Detection(
        kind,
        ran_init=True,
        anchor=candidate,
        changed_inside=changed,
        written_outside=written,
    )


__all__ = [
    "AFTER_INIT",
    "NATIVE",
    "RUN_PLAYER",
    "STAGES",
    "STATIC",
    "SYS_UNPACK",
    "VARIANT_FACET",
    "Located",
    "Rejection",
    "Strategy",
    "locate",
    "variant_facet",
]
