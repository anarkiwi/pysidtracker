"""The bound, instance-specialised playspec IR both backends execute.

Expressions: :class:`K`, :class:`Rd`, :class:`V`, :class:`Op` (``vol``: a read of a
``volatile`` cell, an input of the call that is never folded or shared). Statements: ``("wr", a, v)``,
``("set", n, v)``, ``("if", c, then, else)``, ``("loop", c, body, bound, line)``,
``("assert", c, what)``, ``("hook", stream_index, base)`` and
``("native", addr, ((reg, v), ...), resume)`` (run 6502 code from ``addr`` until it
reaches ``resume``; opaque to the solver), ``("call", id)`` (run ``Program.procs[id]``;
its variables are global, so arguments and results pass by name) and ``("finish",)``
(the whole call ends).

``Op("note", w, (K(clock), K(part), K(offset), index))`` (standard note mode only) is
the :func:`~pysidtracker.notes.table` entry at ``index``, which the solver rejects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple


class K(NamedTuple):
    """A constant of width ``w`` (0: untyped static integer; ``v`` None: unknown)."""

    v: Optional[int]
    w: int


class Rd(NamedTuple):
    """The byte at 16-bit address ``a``."""

    a: tuple


class V(NamedTuple):
    """A variable ``n`` of width ``w``."""

    n: str
    w: int


class Op(NamedTuple):
    """Operator ``op`` of result width ``w`` over ``a``."""

    op: str
    w: int
    a: tuple


def width(expr) -> int:
    """The bit width of ``expr`` (1 for a truth value)."""
    return 8 if isinstance(expr, Rd) else expr.w


def _signed(value: int, bits: int) -> int:
    return value - (1 << bits) if value >> (bits - 1) else value


_FOLD = {
    "add": lambda m, a, b: (a + b) & m,
    "sub": lambda m, a, b: (a - b) & m,
    "mul": lambda m, a, b: (a * b) & m,
    "and": lambda m, a, b: a & b,
    "or": lambda m, a, b: a | b,
    "xor": lambda m, a, b: a ^ b,
    "shl": lambda m, a, b: (a << b) & m,
    "shr": lambda m, a, b: a >> b,
    "not": lambda m, a: a ^ m,
    "neg": lambda m, a: -a & m,
    "zext": lambda m, a: a,
    "sext": lambda m, a: a | 0xFF00 if a & 0x80 else a,
    "lo": lambda m, a: a & 0xFF,
    "hi": lambda m, a: a >> 8,
    "word": lambda m, a, b: a | b << 8,
    "ite": lambda m, c, a, b: a if c else b,
    "eq": lambda m, a, b: int(a == b),
    "ne": lambda m, a, b: int(a != b),
    "ult": lambda m, a, b: int(a < b),
    "ule": lambda m, a, b: int(a <= b),
    "land": lambda m, a, b: a & b,
    "lor": lambda m, a, b: a | b,
    "lnot": lambda m, a: a ^ 1,
}


def evaluate(op: str, w: int, args, arg_width: int = 8) -> int:
    """``op`` of width ``w`` over integer ``args`` (``arg_width``: signed compare operands)."""
    if op in ("slt", "sle"):
        a, b = (_signed(v, arg_width) for v in args)
        return int(a < b if op == "slt" else a <= b)
    return _FOLD[op]((1 << max(w, 1)) - 1, *args)


@dataclass
class Cell:
    """A cell bound to ``place`` ``(addr,)`` or ``(lo, hi)``; ``length`` None for a scalar.

    ``notes`` is ``(part, offset)`` when the cell holds a note table: ``part`` its
    :data:`~pysidtracker.notes.PARTS` key and ``offset`` the note of index 0.
    """

    name: str
    owner: str
    typ: str
    length: Optional[int]
    place: Tuple[Optional[int], ...]
    attrs: frozenset
    line: int
    notes: Optional[Tuple[str, int]] = None

    def extent(self) -> List[int]:
        """Every byte address the declaration spans (empty while unbound)."""
        if None in self.place or (self.length is not None and self.length < 0):
            return []
        count = self.length or 1
        if len(self.place) == 2:
            return [(p + i) & 0xFFFF for p in self.place for i in range(count)]
        size = 1 if self.typ == "u8" else 2
        return [(self.place[0] + i) & 0xFFFF for i in range(count * size)]


@dataclass
class Program:
    """A lowered frame (or one routine): its statements, variables and declarations.

    ``standard`` is the clock in Hz when reads of declared note tables are lowered to
    the standard note words (None: they read memory).
    """

    name: str
    body: list
    signals: List[Tuple[str, int, int]] = field(default_factory=list)
    controls: List[Tuple[str, int, int]] = field(default_factory=list)
    cells: List[Cell] = field(default_factory=list)
    invariants: List[Tuple[str, object]] = field(default_factory=list)
    assumes: List[object] = field(default_factory=list)
    grammars: List[Tuple[str, int]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    order: List[str] = field(default_factory=list)
    inputs: Dict[str, int] = field(default_factory=dict)
    natives: list = field(default_factory=list)
    procs: Dict[str, list] = field(default_factory=dict)
    standard: Optional[int] = None


__all__ = ["Cell", "K", "Op", "Program", "Rd", "V", "evaluate", "width"]
