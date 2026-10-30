"""Format registry for the generic ``pysidtracker`` command-line tool.

A format package registers a :class:`SidFormat` on the ``pysidtracker.formats``
entry-point group; the generic CLI (:mod:`pysidtracker.maincli`) discovers every
installed format, picks the one that recognises a given tune, and runs the
shared ``info`` / ``reglog`` / ``wav`` commands -- plus any format-specific
subcommands the format contributes -- against it.  So a dependent package plugs
its parser + player in here instead of shipping its own CLI binary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from .base import BaseSidParser
from .bytegrammar import ByteGrammar, Stream
from .cadence import cadence_info
from .errors import SidFormatError, SidParseError
from .image import SidImage
from .player import MemPlayer

ENTRY_POINT_GROUP = "pysidtracker.formats"
CANONICAL = "canonical"

# (name, author, released, load, init, play) shown by the shared info block.
Metadata = Callable[[Any], Sequence]


@dataclass(frozen=True)
class FormatCommand:
    """A format-specific subcommand contributed to the generic CLI.

    ``add_arguments(subparser)`` configures the argparse subparser, whose first
    positional argument is the tune, ``song``; ``handler`` receives the parsed
    ``args`` and the owning :class:`SidFormat`.  A name several installed
    formats contribute becomes one subcommand that runs the command of the
    format recognising ``song``, parsed with that format's own arguments.
    """

    name: str
    help: str
    add_arguments: Callable[[Any], None]
    handler: Callable[[Any, "SidFormat"], None]


@dataclass(frozen=True)
class SidFormat:
    """One registered SID format: how to recognise, play, describe and extend it.

    ``parser`` is a :class:`~pysidtracker.base.BaseSidParser` (its ``recognize``
    selects the format and its ``parse`` yields the song model; ``parse(data,
    subtune=k)`` yields the model of 0-based subtune ``k``, the song the
    container's song ``k + 1`` plays).  ``player`` builds a
    :class:`~pysidtracker.player.MemPlayer` of that model's subtune for the
    ``reglog`` / ``wav`` commands.  ``describe`` optionally yields extra ``info``
    lines; ``metadata`` optionally overrides how the shared metadata block is
    read from the model (default: the standard ``name``/``author``/``released``/
    ``load_addr``/``init_addr``/``play_addr`` attributes).  ``commands`` are
    extra subcommands (e.g. ``export``) surfaced only while this format is
    installed.  ``variant`` optionally detects the model's
    :class:`~pysidtracker.facets.Variant`; its ``as_info()`` joins the survey
    renderer's ``info``.  ``facet_sites`` optionally maps the model to
    ``{addr: facet}``, the code address each facet was detected at.  ``survey``
    optionally replaces the facet part of :meth:`info` with ``survey(model, data,
    song)``, the format-specific fields the survey renderer reports.
    ``probes`` maps a name to ``probe(model)``, a derived artifact (a rebuilt
    image, an export) whose digest :mod:`pysidtracker.snapshot` records.
    ``grammar`` and ``streams(model)`` (``(name, model, rule)`` triples) back
    the ``grammar`` command; ``canonical(model)`` is a text that is equal for
    two models exactly when they play the same (the ``equiv`` command).
    ``player_modules`` names the modules, besides ``player``'s own, whose code a
    render depends on, or ``"pkg:path/to/file"`` for a package data file it reads
    (a ``.play`` spec, say); :mod:`pysidtracker.rendercache` invalidates every
    cached render when any of them changes.
    """

    name: str
    parser: BaseSidParser
    player: Callable[[Any], MemPlayer]
    describe: Optional[Callable[[Any], Iterable[str]]] = None
    metadata: Optional[Metadata] = None
    commands: Sequence[FormatCommand] = field(default_factory=tuple)
    variant: Optional[Callable[[Any], Any]] = None
    facet_sites: Optional[Callable[[Any], Mapping[int, str]]] = None
    survey: Optional[Callable[[Any, bytes, int], Mapping[str, Any]]] = None
    probes: Mapping[str, Callable[[Any], Any]] = field(default_factory=dict)
    grammar: Optional[ByteGrammar] = None
    streams: Optional[Callable[[Any], Iterable[Stream]]] = None
    canonical: Optional[Callable[[Any], str]] = None
    player_modules: Sequence[str] = ()

    def listing(self, model: Any) -> Iterator[str]:
        """The :meth:`ByteGrammar.listing` of ``model``'s streams.

        Raises:
          SidFormatError: the format has no ``grammar`` / ``streams`` hook.
        """
        if self.grammar is None or self.streams is None:
            raise SidFormatError(f"format {self.name!r} has no byte grammar")
        return self.grammar.listing(self.streams(model))

    def canonical_text(self, model: Any) -> str:
        """``model``'s canonical text.

        Raises:
          SidFormatError: the format has no ``canonical`` hook.
        """
        if self.canonical is None:
            raise SidFormatError(f"format {self.name!r} has no canonical text")
        return self.canonical(model)

    def snapshot_probes(self) -> Dict[str, Callable[[Any], Any]]:
        """``probes``, plus the UTF-8 canonical text as :data:`CANONICAL` unless
        ``probes`` names one."""
        if self.canonical is None:
            return dict(self.probes)
        return {
            CANONICAL: lambda model: self.canonical_text(model).encode(),
            **self.probes,
        }

    def variant_info(self, model: Any) -> Dict[str, Any]:
        """``model``'s :meth:`Variant.as_info` (empty without a ``variant`` hook)."""
        return self.variant(model).as_info() if self.variant is not None else {}

    def info(self, model: Any, data: bytes, song: int) -> Dict[str, Any]:
        """The survey ``info`` of 1-based ``song`` of ``data`` (loaded as ``model``).

        ``survey(model, data, song)`` when the format supplies it, else
        :meth:`variant_info`; plus :func:`~pysidtracker.cadence.cadence_info`.
        """
        if self.survey is not None:
            fields = dict(self.survey(model, data, song))
        else:
            fields = self.variant_info(model)
        return {**fields, **cadence_info(data, song)}

    def sites(self, model: Any) -> Dict[int, str]:
        """``{addr: facet}`` of ``model``: the facet each detected code site belongs to.

        Empty without a ``facet_sites`` hook; the addresses label 6502 edges in
        whydiff / pathdiff.
        """
        return dict(self.facet_sites(model)) if self.facet_sites is not None else {}

    def model_metadata(self, model: Any) -> Sequence:
        """The ``(name, author, released, load, init, play)`` tuple for ``model``."""
        if self.metadata is not None:
            return self.metadata(model)
        return (
            model.name,
            model.author,
            model.released,
            model.load_addr,
            model.init_addr,
            model.play_addr,
        )

    def load(self, data: bytes, subtune: Optional[int] = None) -> Any:
        """The model of ``data``'s 0-based ``subtune``, by default its start song."""
        return self.parser.parse(
            data, subtune=start_subtune(data) if subtune is None else subtune
        )

    def play(self, data: bytes, subtune: Optional[int] = None) -> MemPlayer:
        """A fresh :class:`~pysidtracker.player.MemPlayer` of ``data``'s ``subtune``."""
        return self.player(self.load(data, subtune))


def start_subtune(data: bytes) -> int:
    """The 0-based subtune ``data``'s PSID/RSID header starts on (0 without one)."""
    header = SidImage.from_bytes(data).header
    return max(header.start_song, 1) - 1 if header is not None else 0


def discover_formats() -> List[SidFormat]:
    """Every :class:`SidFormat` registered on the entry-point group.

    An entry point may resolve to a :class:`SidFormat` or a zero-argument
    callable returning one.  Results are sorted by ``name`` for a stable CLI.
    """
    found: List[SidFormat] = []
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        obj = entry.load()
        fmt = obj() if not isinstance(obj, SidFormat) and callable(obj) else obj
        found.append(fmt)
    return sorted(found, key=lambda fmt: fmt.name)


def recognize_format(formats: Iterable[SidFormat], data: bytes) -> Optional[SidFormat]:
    """The first format whose parser recognises ``data`` (or ``None``).

    A parser whose ``recognize`` raises is skipped, never breaking selection.
    """
    image = SidImage.from_bytes(data)
    for fmt in formats:
        try:
            if fmt.parser.recognize(image):
                return fmt
        except Exception:  # pylint: disable=broad-except
            continue
    return None


def find_format(
    data: bytes, name: str = "auto", formats: Optional[Iterable[SidFormat]] = None
) -> SidFormat:
    """The installed format ``name`` (``auto``: the one recognising ``data``).

    ``formats`` defaults to :func:`discover_formats`; raises
    :class:`~pysidtracker.errors.SidParseError` when there is none.
    """
    found = discover_formats() if formats is None else list(formats)
    if name == "auto":
        fmt = recognize_format(found, data)
    else:
        fmt = next((f for f in found if f.name == name), None)
    if fmt is None:
        what = "recognises" if name == "auto" else f"named {name!r} plays"
        raise SidParseError(f"no installed pysidtracker format {what} this tune")
    return fmt


def render(
    name: str, data: bytes, song: int, nframes: int
) -> Tuple[np.ndarray, Dict[str, str]]:
    """A survey renderer: 1-based ``song`` of ``data`` on format ``name`` (or ``auto``).

    ``(grid, info)``, the grid ``nframes`` rows of the 25 SID registers and ``info``
    ``{"format": name}`` plus :meth:`SidFormat.info`.  A named format must also
    recognise ``data``, else :class:`~pysidtracker.errors.SidParseError`.
    """
    fmt = find_format(data, name)
    if not recognize_format([fmt], data):
        raise SidParseError(f"format {fmt.name!r} does not recognise this tune")
    model = fmt.load(data, song - 1)
    grid = fmt.player(model).render_grid(nframes)
    info = {"format": fmt.name, **fmt.info(model, data, song)}
    return np.asarray(grid, np.int16), info
