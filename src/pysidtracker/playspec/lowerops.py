"""Expression constructors shared by lowering: folding builders over the IR."""

from __future__ import annotations

from ..notes import table
from .ir import K, Op, evaluate, width


def mk(op: str, w: int, *args):
    """``Op(op, w, args)``, folded when every argument is a known constant."""
    if op == "note":
        return note_entry(w, args)
    if all(isinstance(a, K) and a.v is not None for a in args):
        return K(evaluate(op, w, [a.v for a in args], max(width(a) for a in args)), w)
    if op == "land":
        for a, b in (args, args[::-1]):
            if isinstance(a, K) and a.v is not None:
                return b if a.v else K(0, 1)
    if op == "lor":
        for a, b in (args, args[::-1]):
            if isinstance(a, K) and a.v is not None:
                return K(1, 1) if a.v else b
    if op == "lnot" and isinstance(args[0], Op) and args[0].op == "lnot":
        return args[0].a[0]
    return Op(op, w, tuple(args))


def note_entry(w: int, args):
    """The ``note`` operator over ``args``, its table entry once the index is constant."""
    clock, part, offset, index = args
    if all(isinstance(a, K) and a.v is not None for a in args):
        size = 1 << max(8, width(index))
        if 0 <= index.v < size:
            return K(table(clock.v, part.v, offset.v, size)[index.v], w)
    return Op("note", w, tuple(args))


def add16(base, index):
    """The 16-bit address ``base + index`` (``index`` zero-extended)."""
    return mk("add", 16, base, zext(index, 16))


def zext(expr, w: int):
    """``expr`` widened to ``w`` bits."""
    if width(expr) == w:
        return expr
    if isinstance(expr, K) and expr.w == 0:
        return K(None if expr.v is None else expr.v & ((1 << w) - 1), w)
    return mk("zext", w, expr)


def word(lo, hi):
    """The little-endian word of bytes ``lo``, ``hi``."""
    return mk("word", 16, lo, hi)


__all__ = ["add16", "mk", "note_entry", "word", "zext"]
