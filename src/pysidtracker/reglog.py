"""SID register write logs -- the shared ``py*`` register-log convention.

A register log flattens a player's per-frame output to timed chip writes: one
:class:`RegWrite` per SID register write, with an absolute clock in C64 CPU
cycles. The register index ``reg`` is the SID register OFFSET (``0..$18``)
relative to ``$D400``, not the absolute address. Logs serialize to plain text,
one ``clock reg val`` triple per line (decimal, space separated, ``#`` comments
allowed), so they load directly into pandas or any line-based tooling.

This module consolidates the byte-identical ``RegWrite`` / ``read_reglog`` /
``write_reglog`` surface each format package hand-copied (pygoattracker, pyjch,
pymusicassembler, pyfuturecomposer, pydefmon), plus :func:`frame_writes`, the
shared per-frame framing loop those packages' ``iter_register_writes`` all run.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import IO, Iterable, Iterator, NamedTuple, Tuple

from .errors import SidParseError

# Cycles between consecutive writes within one frame, approximating the store
# instructions of the 6502 playroutine.
DEFAULT_WRITE_SPACING = 16

# SID register file size ($D400..$D418): 25 registers.
SID_REG_COUNT = 0x19

REGLOG_HEADER = "# pysidtracker register log: clock reg val"


class RegWrite(NamedTuple):
    """One SID register write at an absolute CPU clock (in cycles).

    ``reg`` is the register OFFSET ``0..$18`` relative to ``$D400`` (the shared
    ``py*`` register-log convention), ``val`` the byte written.
    """

    clock: int
    reg: int
    val: int


def write_reglog(writes: Iterable[RegWrite], dst, header: bool = True) -> None:
    """Write a register log to a path or text file-like object."""

    def _dump(out: IO[str]) -> None:
        if header:
            print(REGLOG_HEADER, file=out)
        for write in writes:
            print(f"{write.clock} {write.reg} {write.val}", file=out)

    if isinstance(dst, (str, Path)):
        with open(dst, "w", encoding="utf-8") as out:
            _dump(out)
        return
    _dump(dst)


def read_reglog(src) -> "list[RegWrite]":
    """Read a register log from a path or text file-like object.

    ``#`` comments and blank lines are ignored. A malformed line raises
    :class:`~pysidtracker.errors.SidParseError`.
    """
    if isinstance(src, (str, Path)):
        text = Path(src).read_text(encoding="utf-8")
    elif isinstance(src, io.IOBase) or hasattr(src, "read"):
        text = src.read()
    else:
        raise TypeError(f"cannot read a register log from {type(src).__name__}")
    writes = []
    for num, line in enumerate(text.splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 3:
            raise SidParseError(f"bad register log line {num}: {line!r}")
        try:
            writes.append(RegWrite(*(int(field) for field in fields)))
        except ValueError as exc:
            raise SidParseError(f"bad register log line {num}: {line!r}") from exc
    return writes


def frame_writes(
    per_frame_iter: Iterable[Iterable[Tuple[int, int]]],
    *,
    cycles_per_frame: int,
    write_spacing: int = DEFAULT_WRITE_SPACING,
    start_frame: int = 0,
    sid_reg_base: int = 0xD400,
    reg_count: int = SID_REG_COUNT,
) -> Iterator[RegWrite]:
    """Frame a per-frame ``(reg, val)`` write stream into :class:`RegWrite`.

    ``per_frame_iter`` is an iterable of per-frame iterables of ``(reg, val)``.
    For each frame index ``f`` (starting at ``start_frame``), each frame write
    is rebased to a SID register offset (``reg - sid_reg_base``) and masked
    (``val & 0xFF``); writes whose rebased register lands in ``0..0x18`` are
    emitted as ``RegWrite(clock=f*cycles_per_frame + offset*write_spacing,
    reg=rebased, val=masked)``, with ``offset`` incrementing per emitted write.

    Pass ``sid_reg_base=0`` when the player already yields ``0..24`` offsets.
    Raises :class:`~pysidtracker.errors.SidParseError` if
    ``write_spacing * reg_count >= cycles_per_frame`` (writes would overrun the
    frame).
    """
    if write_spacing * reg_count >= cycles_per_frame:
        raise SidParseError("write_spacing too large for one frame")
    for index, writes in enumerate(per_frame_iter):
        frame = start_frame + index
        base_clock = frame * cycles_per_frame
        offset = 0
        for reg, val in writes:
            rebased = reg - sid_reg_base
            if 0 <= rebased <= 0x18:
                yield RegWrite(base_clock + offset * write_spacing, rebased, val & 0xFF)
                offset += 1
