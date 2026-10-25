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
prefilter is numpy-accelerated (mirroring :mod:`pysidtracker._scan`).

:class:`SitePattern` extends the same token syntax with alternation, optional
groups, bounded gaps and capture equality for build *sites* that are not a fixed
length (see its docstring); it compiles to a bytes regex.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .image import MEM_SIZE, SidImage

CodeBuffer = Union[SidImage, bytes, bytearray, memoryview]

import numpy as _np

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


_LABEL_RE = re.compile(r"^(?P<label>[A-Za-z_][A-Za-z0-9_]*):$")
_GAP_RE = re.compile(r"^\?\?\{(?P<lo>[0-9]+)(?:,(?P<hi>[0-9]+))?\}$")
_SPANS = ("code", "image")


@dataclass(frozen=True)
class SiteMatch(Match):
    """One located site: a :class:`Match` plus which alternatives matched.

    Attributes:
      length: number of bytes the match spans from :attr:`addr`.
      alt: label of the alternative chosen by the first ``( … | … )`` group, or
        ``None`` when the pattern has no alternation.
      alts: labels chosen by every alternation group, in pattern order (a group
        inside a non-taken optional branch contributes nothing).
    """

    length: int = 0
    alt: Optional[str] = None
    alts: Tuple[str, ...] = ()


class SitePattern:
    """A variable-length 6502 code-site pattern compiled to a bytes regex.

    Tokens are those of :class:`CodePattern` (``A9``, ``??``, ``{name}``,
    ``{name:w}``) plus:

    * ``( on: A9 01 | off: A9 00 | … )`` -- alternation. Each branch may start
      with a ``label:``; unlabelled branches are labelled by their index
      (``"0"``, ``"1"``, …). The earliest branch that matches wins.
    * ``[ … ]`` -- an optional group.
    * ``??{n}`` / ``??{lo,hi}`` -- a gap of exactly ``n`` or ``lo``..``hi`` any
      bytes (shortest first).
    * repeating a ``{name}`` / ``{name:w}`` within one branch requires the
      repeated operand to be equal to the first (capture equality). Names first
      bound inside a branch or optional group are not visible after it.

    ``first`` and ``span`` are the match policy: with ``first=True`` only the
    lowest-address match is reported; ``span="code"`` (default) scans the loaded
    code ``[load, end)`` of a :class:`~pysidtracker.SidImage`, ``"image"`` the
    whole 64 KiB image. A raw buffer is always scanned whole.
    """

    __slots__ = ("spec", "first", "span", "_regex", "_caps", "_alts", "_next")

    def __init__(self, spec: str, *, first: bool = False, span: str = "code"):
        if span not in _SPANS:
            raise ValueError(f"span must be one of {_SPANS}, not {span!r}")
        self.spec = spec
        self.first = first
        self.span = span
        self._caps: Dict[str, Tuple[int, List[str]]] = {}
        self._alts: List[Tuple[str, List[Tuple[str, str]]]] = []
        self._next = 0
        spaced = re.sub(r"([()\[\]|])", r" \1 ", spec)
        tokens = spaced.split()
        body, pos = self._seq(tokens, 0, {})
        if pos != len(tokens):
            raise ValueError(f"unbalanced {tokens[pos]!r} in site pattern")
        if not body:
            raise ValueError("empty site pattern")
        self._regex = re.compile(body.encode("latin-1"), re.DOTALL)

    def _gid(self, prefix: str) -> str:
        self._next += 1
        return f"{prefix}{self._next}"

    def _seq(self, tokens: List[str], pos: int, scope: Dict[str, str]):
        out: List[str] = []
        while pos < len(tokens) and tokens[pos] not in ("|", ")", "]"):
            token = tokens[pos]
            pos += 1
            if token == "(":
                part, pos = self._alternation(tokens, pos, scope)
            elif token == "[":
                inner, pos = self._seq(tokens, pos, dict(scope))
                if pos >= len(tokens) or tokens[pos] != "]":
                    raise ValueError("unterminated '[' in site pattern")
                pos += 1
                part = f"(?:{inner})?"
            else:
                part = self._atom(token, scope)
            out.append(part)
        return "".join(out), pos

    def _alternation(self, tokens: List[str], pos: int, scope: Dict[str, str]):
        gname = self._gid("a")
        branches: List[str] = []
        labels: List[Tuple[str, str]] = []
        while True:
            label = str(len(branches))
            if pos < len(tokens):
                lab = _LABEL_RE.match(tokens[pos])
                if lab is not None:
                    label = lab.group("label")
                    pos += 1
            body, pos = self._seq(tokens, pos, dict(scope))
            if not body:
                raise ValueError("empty alternative in site pattern")
            bname = self._gid("b")
            branches.append(f"(?P<{bname}>{body})")
            labels.append((bname, label))
            if pos >= len(tokens):
                raise ValueError("unterminated '(' in site pattern")
            closer = tokens[pos]
            pos += 1
            if closer == ")":
                break
            if closer != "|":
                raise ValueError(f"unexpected {closer!r} in alternation")
        self._alts.append((gname, labels))
        return "(?:" + "|".join(branches) + ")", pos

    def _atom(self, token: str, scope: Dict[str, str]) -> str:
        cap = _CAPTURE_RE.match(token)
        if cap is not None:
            name = cap.group("name")
            width = 2 if cap.group("kind") == "w" else 1
            known = self._caps.get(name)
            if known is not None and known[0] != width:
                raise ValueError(f"capture {name!r} used with two widths")
            if name in scope:
                return f"(?P={scope[name]})"
            gid = self._gid("g")
            scope[name] = gid
            self._caps.setdefault(name, (width, []))[1].append(gid)
            return f"(?P<{gid}>{'.' * width})"
        if token == "??":
            return "."
        gap = _GAP_RE.match(token)
        if gap is not None:
            lo = int(gap.group("lo"))
            hi = gap.group("hi")
            if hi is None:
                return f".{{{lo}}}"
            if int(hi) < lo:
                raise ValueError(f"invalid gap {token!r}")
            return f".{{{lo},{int(hi)}}}?"
        if _LITERAL_RE.match(token):
            return f"\\x{token}"
        raise ValueError(f"invalid site-pattern token: {token!r}")

    def _build(self, m: "re.Match[bytes]", buf: bytes) -> SiteMatch:
        caps: Dict[str, int] = {}
        for name, (width, gids) in self._caps.items():
            for gid in gids:
                raw = m.group(gid)
                if raw is not None:
                    caps[name] = int.from_bytes(raw[:width], "little")
                    break
        alts: List[str] = []
        for _, labels in self._alts:
            for bname, label in labels:
                if m.group(bname) is not None:
                    alts.append(label)
                    break
        return SiteMatch(
            m.start(),
            caps,
            buf,
            length=m.end() - m.start(),
            alt=alts[0] if alts else None,
            alts=tuple(alts),
        )

    def _range(self, source: CodeBuffer, start, end) -> Tuple[bytes, int, int]:
        buf = bytes(source.mem) if isinstance(source, SidImage) else bytes(source)
        lo, hi = 0, len(buf)
        if isinstance(source, SidImage) and self.span == "code":
            lo, hi = source.load, min(hi, source.end)
        if start is not None:
            lo = start
        if end is not None:
            hi = min(hi, end)
        return buf, max(0, lo), hi

    def find_all(
        self,
        image: CodeBuffer,
        *,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> List[SiteMatch]:
        """Matches in ``[start, end)`` per the pattern's ``first``/``span`` policy.

        Matches may overlap and are returned in ascending address order.
        """
        buf, pos, hi = self._range(image, start, end)
        out: List[SiteMatch] = []
        while pos <= hi:
            m = self._regex.search(buf, pos, hi)
            if m is None:
                break
            out.append(self._build(m, buf))
            if self.first:
                break
            pos = m.start() + 1
        return out

    def find_first(
        self,
        image: CodeBuffer,
        *,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> Optional[SiteMatch]:
        """The lowest-address match, or ``None``."""
        buf, pos, hi = self._range(image, start, end)
        m = self._regex.search(buf, pos, hi)
        return None if m is None else self._build(m, buf)

    def any(
        self,
        image: CodeBuffer,
        *,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> bool:
        """Whether the pattern matches anywhere in the span."""
        return self.find_first(image, start=start, end=end) is not None
