"""PRG files from C64 release media, read natively by :mod:`vagrant_moses.media`.

:func:`read_d64` lists one disk image's PRGs; :func:`iter_prgs` walks any nesting of
zip/gzip, d64 (35/40/42 tracks, error block)/d71/d81, t64 and p00. The ``media
extract`` command writes every PRG of a download tree out with its index line.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from vagrant_moses.media import (
    GEOMETRIES,
    GEOMETRY_BY_SIZE,
    TYPE_PRG,
    MediaError,
    iter_prgs,
    read_disk,
)

from .errors import SidError


@dataclass(frozen=True)
class D64File:
    """One PRG from a disk image: PETSCII ``name``, ``prg`` bytes (load address first).

    ``locked`` is the directory entry's lock bit.
    """

    name: str
    prg: bytes
    locked: bool = False


def read_d64(data: bytes) -> List[D64File]:
    """Every closed PRG on the disk image ``data``, in directory order.

    The geometry follows the image size (a 35-track d64 for an unknown size); a
    broken or looping sector chain ends that file where a 1541 would.

    Raises:
      MediaError: ``data`` is shorter than the geometry, or has no directory.
    """
    geom = GEOMETRY_BY_SIZE.get(len(data), GEOMETRIES[0])
    return [
        D64File(f.name, f.data, f.locked)
        for f in read_disk(data, geom)
        if f.ftype == TYPE_PRG
    ]


def _extract(args) -> None:
    from vagrant_moses.cli import extract  # pylint: disable=import-outside-toplevel

    errors = extract(args.src, Path(args.out))
    if errors:
        raise SidError(f"{errors} unreadable media members (listed above)")


def add_media_commands(parser) -> None:
    """Add ``extract SRC... -o DIR`` (needs tqdm: ``pysidtracker[survey]``)."""
    sub = parser.add_subparsers(dest="media_command", required=True)
    ext = sub.add_parser(
        "extract", help="write every PRG in nested release media, with its index line"
    )
    ext.add_argument("src", nargs="+", help="media files or directories")
    ext.add_argument("-o", "--out", required=True, help="output directory")
    ext.set_defaults(func=_extract)


__all__ = [
    "D64File",
    "MediaError",
    "add_media_commands",
    "iter_prgs",
    "read_d64",
]
