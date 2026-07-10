"""Masked 6502 code-fragment search with operand capture.

Every relocatable SID player hides its per-tune data behind the same trick: the
absolute addresses that would let you read a table at a fixed offset are baked
into the *player code* as instruction operands, and the player relocates as one
block, so those operands move with it. The reader must therefore locate a code
fragment by its **opcode skeleton** -- the fixed opcodes and immediates, with
the per-tune operand bytes wildcarded -- and read the captured operand (an
immediate byte, or a 16-bit little-endian address) from the match.

Each ``py*`` parser reinvented this: :mod:`pyfuturecomposer` and :mod:`pydefmon`
hand-rolled ``None``-wildcard token lists, :mod:`pymusicassembler` used ``re``
with ``(..)`` capture groups, :mod:`pyjch` walked ``prefix<operand>suffix``
idioms, and :mod:`pysoundmonitor` harvested absolute operands per opcode. This
module is the one abstraction that expresses all of them.

A :class:`CodePattern` is compiled from a compact spec of whitespace-separated
tokens:

* a two-hex literal (``A9``) matches that byte exactly,
* ``??`` matches any single byte (a wildcard),
* ``{name}`` captures one operand byte,
* ``{name:w}`` captures a two-byte little-endian word.

``find_code_all`` / ``find_code_first`` scan a :class:`~pysidtracker.SidImage`
(its ``mem`` from ``image.load`` by default) and return :class:`Match` objects
carrying the match address and the captured operands. The literal/wildcard
prefilter is numpy-accelerated when numpy is importable, with a pure-stdlib
fallback (mirroring :mod:`pysidtracker._scan`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .image import MEM_SIZE, SidImage

CodeBuffer = Union[SidImage, bytes, bytearray, memoryview]

try:  # numpy is an optional accelerator, never required.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only without numpy
    _np = None

_CAPTURE_RE = re.compile(r"^\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::(?P<kind>[bw]))?\}$")
_LITERAL_RE = re.compile(r"^[0-9A-Fa-f]{2}$")


@dataclass(frozen=True)
class Match:
    """One located code fragment.

    Attributes:
      addr: C64 address of the first pattern byte.
      captures: operand values keyed by capture name (a byte for ``{name}``,
        a little-endian word for ``{name:w}``).
      buf: the scanned buffer, so :meth:`u8`/:meth:`u16` can read operand bytes
        at an offset from :attr:`addr` (excluded from equality/hash).
    """

    addr: int
    captures: Dict[str, int] = field(default_factory=dict)
    buf: Optional[bytes] = field(default=None, compare=False, repr=False)

    def u8(self, offset: int = 0) -> int:
        """The byte ``offset`` bytes into the match."""
        return self.buf[self.addr + offset]

    def u16(self, offset: int = 0) -> int:
        """The little-endian word ``offset`` bytes into the match."""
        base = self.addr + offset
        return self.buf[base] | (self.buf[base + 1] << 8)


class CodePattern:
    """A compiled masked 6502 code-fragment pattern.

    Compile once from a token spec and reuse across images. ``literals`` holds
    the fixed ``(offset, value)`` bytes that drive the prefilter; ``captures``
    holds the ``(name, offset, width)`` operands read out of a match.
    """

    __slots__ = ("spec", "length", "literals", "captures")

    def __init__(self, spec: str):
        self.spec = spec
        literals: List[Tuple[int, int]] = []
        captures: List[Tuple[str, int, int]] = []
        offset = 0
        for token in spec.split():
            cap = _CAPTURE_RE.match(token)
            if cap is not None:
                width = 2 if cap.group("kind") == "w" else 1
                captures.append((cap.group("name"), offset, width))
                offset += width
                continue
            if token == "??":
                offset += 1
                continue
            if _LITERAL_RE.match(token):
                literals.append((offset, int(token, 16)))
                offset += 1
                continue
            raise ValueError(f"invalid code-pattern token: {token!r}")
        if offset == 0:
            raise ValueError("empty code pattern")
        self.length = offset
        self.literals: Tuple[Tuple[int, int], ...] = tuple(literals)
        self.captures: Tuple[Tuple[str, int, int], ...] = tuple(captures)

    def _read_captures(self, buf: Sequence[int], addr: int) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for name, offset, width in self.captures:
            base = addr + offset
            value = buf[base]
            if width == 2:
                value |= buf[base + 1] << 8
            out[name] = value
        return out


def _buffer_and_range(
    source: CodeBuffer, start: Optional[int], end: Optional[int]
) -> Tuple[bytes, int, int]:
    if isinstance(source, SidImage):
        buf = bytes(source.mem)
        default_lo = source.load
    else:
        buf = bytes(source)
        default_lo = 0
    lo = default_lo if start is None else start
    hi = MEM_SIZE if end is None else end
    return buf, max(0, lo), min(len(buf), hi)


def _candidate_starts(buf: bytes, pat: CodePattern, lo: int, hi: int) -> Sequence[int]:
    """Start addresses to verify, prefiltered by the pattern's first literal."""
    last = hi - pat.length
    if last < lo:
        return ()
    if not pat.literals:
        return range(lo, last + 1)
    off0, val0 = pat.literals[0]
    if _np is not None:
        return _candidate_starts_np(buf, off0, val0, lo, last)
    return _candidate_starts_py(buf, off0, val0, lo, last)


def _candidate_starts_py(buf, off0, val0, lo, last):
    starts = []
    pos = buf.find(val0, lo + off0, last + off0 + 1)
    while pos != -1:
        starts.append(pos - off0)
        pos = buf.find(val0, pos + 1, last + off0 + 1)
    return starts


def _candidate_starts_np(buf, off0, val0, lo, last):
    arr = _np.frombuffer(buf, dtype=_np.uint8)
    hits = _np.nonzero(arr[lo + off0 : last + off0 + 1] == val0)[0]
    return (hits + lo).tolist()


def _matches_literals(buf: Sequence[int], pat: CodePattern, addr: int) -> bool:
    for offset, value in pat.literals:
        if buf[addr + offset] != value:
            return False
    return True


def find_code_all(
    image: CodeBuffer,
    pattern: CodePattern | str,
    *,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> List[Match]:
    """Every match of ``pattern`` in ``image`` at or after ``start``.

    ``image`` may be a :class:`~pysidtracker.SidImage` or a raw
    ``bytes``/``bytearray``/``memoryview`` buffer. ``pattern`` may be a compiled
    :class:`CodePattern` or a spec string. Searches over ``[start, end)``
    (default: the image load address, or 0 for a raw buffer, to the top of
    memory). Matches are returned in ascending address order.
    """
    pat = pattern if isinstance(pattern, CodePattern) else CodePattern(pattern)
    buf, lo, hi = _buffer_and_range(image, start, end)
    out: List[Match] = []
    for addr in _candidate_starts(buf, pat, lo, hi):
        if _matches_literals(buf, pat, addr):
            out.append(Match(addr, pat._read_captures(buf, addr), buf))
    return out


def find_code_first(
    image: CodeBuffer,
    pattern: CodePattern | str,
    *,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> Optional[Match]:
    """First match of ``pattern`` in ``image``, or ``None``.

    Same arguments as :func:`find_code_all`; stops at the first hit.
    """
    pat = pattern if isinstance(pattern, CodePattern) else CodePattern(pattern)
    buf, lo, hi = _buffer_and_range(image, start, end)
    for addr in _candidate_starts(buf, pat, lo, hi):
        if _matches_literals(buf, pat, addr):
            return Match(addr, pat._read_captures(buf, addr), buf)
    return None
