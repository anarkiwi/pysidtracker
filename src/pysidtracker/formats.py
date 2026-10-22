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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .base import BaseSidParser
from .errors import SidParseError
from .image import SidImage
from .player import MemPlayer

ENTRY_POINT_GROUP = "pysidtracker.formats"

# (name, author, released, load, init, play) shown by the shared info block.
Metadata = Callable[[Any], Sequence]


@dataclass(frozen=True)
class FormatCommand:
    """A format-specific subcommand contributed to the generic CLI.

    ``add_arguments(subparser)`` configures the argparse subparser; ``handler``
    receives the parsed ``args`` and the owning :class:`SidFormat`.
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
    installed.
    """

    name: str
    parser: BaseSidParser
    player: Callable[[Any], MemPlayer]
    describe: Optional[Callable[[Any], Iterable[str]]] = None
    metadata: Optional[Metadata] = None
    commands: Sequence[FormatCommand] = field(default_factory=tuple)

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
        raise SidParseError(
            f"no installed pysidtracker format {name!r} plays this tune"
        )
    return fmt


def render(
    name: str, data: bytes, song: int, nframes: int
) -> Tuple[np.ndarray, Dict[str, str]]:
    """A survey renderer: 1-based ``song`` of ``data`` on format ``name`` (or ``auto``).

    ``(grid, {"format": name})``, the grid ``nframes`` rows of the 25 SID registers.
    """
    fmt = find_format(data, name)
    grid = fmt.play(data, song - 1).render_grid(nframes)
    return np.asarray(grid, np.int16), {"format": fmt.name}
