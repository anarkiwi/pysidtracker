"""Parse playspec source (``playspec.lark``) into a tuple AST.

Every node is ``(tag, line, *fields)``; :func:`parse` returns a :class:`Spec`
whose declarations keep source order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Dict, List, Tuple

from lark import Lark, Token, Transformer, v_args
from lark.exceptions import LarkError

from ..errors import SidError


class SpecError(SidError):
    """A playspec that does not parse, type-check or bind."""


@dataclass
class Spec:
    """A parsed spec: its name and declarations by kind, in source order."""

    name: str = ""
    layout: Dict[str, str] = field(default_factory=dict)
    consts: List[Tuple] = field(default_factory=list)
    cells: List[Tuple] = field(default_factory=list)
    procs: Dict[str, Tuple] = field(default_factory=dict)
    machines: Dict[str, Tuple] = field(default_factory=dict)
    schedule: Tuple = ()
    invariants: List[Tuple] = field(default_factory=list)
    assumes: List[Tuple] = field(default_factory=list)


def _line(meta) -> int:
    return getattr(meta, "line", 0)


def _int(text: str) -> int:
    return int(text, 0)


@v_args(meta=True)
class _ToAst(Transformer):
    # pylint: disable=missing-function-docstring,too-many-public-methods
    def _n(self, tag, meta, *fields):
        return (tag, _line(meta), *fields)

    def start(self, meta, items):
        del meta
        spec = Spec()
        for item in items:
            tag = item[0]
            if tag == "spec":
                spec.name = item[2]
            elif tag == "layout":
                for names, kind in item[2]:
                    spec.layout.update(dict.fromkeys(names, kind))
            elif tag == "const":
                spec.consts.append(item)
            elif tag == "cell":
                spec.cells.append(item)
            elif tag == "proc":
                spec.procs[item[2]] = item
            elif tag == "machine":
                spec.machines[item[2]] = item
            elif tag == "schedule":
                spec.schedule = item
            elif tag == "invariant":
                spec.invariants.append(item)
            else:
                spec.assumes.append(item)
        return spec

    def spec_decl(self, meta, c):
        return self._n("spec", meta, str(c[0]))

    def layout(self, meta, c):
        return self._n("layout", meta, c)

    def lfield(self, meta, c):
        del meta
        return [str(t) for t in c[:-1]], str(c[-1])

    def const(self, meta, c):
        return self._n("const", meta, str(c[0]), c[1])

    def cell(self, meta, c):
        name, (ctype, length), place, *attrs = c
        return self._n(
            "cell", meta, str(name), ctype, length, place, {str(a) for a in attrs}
        )

    def ctype(self, meta, c):
        del meta
        return str(c[0]), (c[1] if len(c) > 1 else None)

    def at(self, meta, c):
        return self._n("at", meta, c[0])

    def split(self, meta, c):
        return self._n("split", meta, c[0], c[1])

    def machine(self, meta, c):
        name = str(c[0])
        if len(c) > 1 and isinstance(c[1], Token):
            index, lo, hi, body = str(c[1]), c[2], c[3], c[4:]
        else:
            index, lo, hi, body = None, None, None, c[1:]
        return self._n("machine", meta, name, index, lo, hi, body)

    def signal(self, meta, c):
        return self._n("signal", meta, str(c[0]), str(c[1]), c[2])

    def control(self, meta, c):
        if isinstance(c[1], Token):
            cell, mask, states = str(c[1]), c[2], c[3:]
        else:
            cell, mask, states = None, None, c[1:]
        return self._n("control", meta, str(c[0]), cell, mask, states)

    def state(self, meta, c):
        return self._n("state", meta, str(c[0]), c[1] if len(c) > 1 else None)

    def step(self, meta, c):
        return self._n("proc", meta, "step", [], None, c[0])

    def proc(self, meta, c):
        name, *rest = c
        body = rest[-1]
        ret = rest[-2] if len(rest) >= 2 and isinstance(rest[-2], Token) else None
        params = [p for p in rest[:-1] if isinstance(p, tuple)]
        return self._n("proc", meta, str(name), params, ret and str(ret), body)

    def param(self, meta, c):
        del meta
        return str(c[0]), str(c[1])

    def schedule(self, meta, c):
        return self._n("schedule", meta, c[0])

    def sblock(self, meta, c):
        del meta
        return list(c)

    def inst(self, meta, c):
        return self._n("inst", meta, str(c[0]), c[1] if len(c) > 1 else None)

    def sif(self, meta, c):
        return self._n("sif", meta, c[0], c[1], c[2] if len(c) > 2 else [])

    def sfor(self, meta, c):
        return self._n("sfor", meta, str(c[0]), c[1], c[2])

    def irange(self, meta, c):
        return self._n("irange", meta, c[0], c[1])

    def ilist(self, meta, c):
        return self._n("ilist", meta, c[0])

    def invariant(self, meta, c):
        return self._n("invariant", meta, str(c[0]), c[1])

    def assume(self, meta, c):
        return self._n("assume", meta, c[0])

    def block(self, meta, c):
        del meta
        return list(c)

    def assign(self, meta, c):
        return self._n("assign", meta, c[0], str(c[1]), c[2])

    def let(self, meta, c):
        typ = str(c[1]) if len(c) == 3 else None
        return self._n("let", meta, str(c[0]), typ, c[-1])

    def next_let(self, meta, c):
        return self._n("next_let", meta, str(c[0]), str(c[1]))

    def next_set(self, meta, c):
        return self._n("next_set", meta, c[0], str(c[1]), str(c[2]))

    def if_(self, meta, c):
        arms = [(c[0], c[1])]
        rest = c[2:]
        while rest and isinstance(rest[0], tuple) and rest[0][0] == "elif":
            arms.append((rest[0][2], rest[0][3]))
            rest = rest[1:]
        return self._n("if", meta, arms, rest[0] if rest else [])

    def elif_(self, meta, c):
        return self._n("elif", meta, c[0], c[1])

    def while_(self, meta, c):
        return self._n("while", meta, c[0], _int(c[1]), c[2])

    def for_(self, meta, c):
        return self._n("for", meta, str(c[0]), c[1], c[2])

    def call(self, meta, c):
        return self._n("call", meta, str(c[0]), [a for a in c[1:] if a is not None])

    def return_(self, meta, c):
        return self._n("return", meta, c[0] if c else None)

    def goto(self, meta, c):
        return self._n("goto", meta, str(c[0]))

    def in_(self, meta, c):
        return self._n("in", meta, str(c[0]), c[1:])

    def arm(self, meta, c):
        del meta
        return str(c[0]), c[1]

    def stream(self, meta, c):
        return self._n("stream", meta, str(c[0]), c[1], c[2], c[3:])

    def native(self, meta, c):
        return self._n("native", meta, c[0], [a for a in c[1:-1] if a is not None], c[-1])

    def narg(self, meta, c):
        del meta
        return str(c[0]), c[1]

    def gsel(self, meta, c):
        del meta
        return str(c[0])[1:-1], c[1] if len(c) > 1 else None

    def assert_(self, meta, c):
        return self._n("assert", meta, c[0])

    def lname(self, meta, c):
        return self._n("name", meta, str(c[0]))

    def lindex(self, meta, c):
        return self._n("index", meta, str(c[0]), c[1])

    def lmem(self, meta, c):
        return self._n("mem", meta, c[0])

    def lattr(self, meta, c):
        return self._n("attr", meta, str(c[0]), str(c[1]))

    def _chain(self, tag, meta, c):
        out = c[0]
        for item in c[1:]:
            out = (tag, _line(meta), out, item)
        return out

    def lor_(self, meta, c):
        return self._chain("||", meta, c)

    def land_(self, meta, c):
        return self._chain("&&", meta, c)

    def bor_(self, meta, c):
        return self._chain("|", meta, c)

    def bxor_(self, meta, c):
        return self._chain("^", meta, c)

    def band_(self, meta, c):
        return self._chain("&", meta, c)

    def prod_(self, meta, c):
        return self._ops(meta, c)

    def _ops(self, meta, c):
        out = c[0]
        for op, rhs in zip(c[1::2], c[2::2]):
            out = (str(op), _line(meta), out, rhs)
        return out

    def shift_(self, meta, c):
        return self._ops(meta, c)

    def sum_(self, meta, c):
        return self._ops(meta, c)

    def cmp_(self, meta, c):
        return self._ops(meta, c)

    def not_(self, meta, c):
        return self._n("!", meta, c[0])

    def inv_(self, meta, c):
        return self._n("~", meta, c[0])

    def neg_(self, meta, c):
        return self._n("uminus", meta, c[0])

    def num(self, meta, c):
        return self._n("num", meta, _int(c[0]))

    def true_(self, meta, c):
        del c
        return self._n("bool", meta, True)

    def false_(self, meta, c):
        del c
        return self._n("bool", meta, False)

    def name(self, meta, c):
        return self._n("name", meta, str(c[0]))

    def index(self, meta, c):
        return self._n("index", meta, str(c[0]), c[1])

    def inst_attr(self, meta, c):
        return self._n("inst_attr", meta, str(c[0]), c[1], str(c[2]))

    def attr(self, meta, c):
        return self._n("attr", meta, str(c[0]), str(c[1]))

    def mem(self, meta, c):
        return self._n("mem", meta, c[0])

    def mem16(self, meta, c):
        return self._n("mem16", meta, c[0])

    def func(self, meta, c):
        return self._n("func", meta, str(c[0]), [a for a in c[1:] if a is not None])


@lru_cache(maxsize=None)
def _parser() -> Lark:
    text = resources.files(__package__).joinpath("playspec.lark").read_text()
    return Lark(text, parser="lalr", propagate_positions=True, maybe_placeholders=True)


def parse(text: str) -> Spec:
    """The :class:`Spec` of playspec ``text``; raises :class:`SpecError` on a syntax error."""
    try:
        tree = _parser().parse(text)
    except LarkError as exc:
        raise SpecError(f"playspec syntax: {exc}") from exc
    return _ToAst().transform(tree)


def load(path) -> Spec:
    """The :class:`Spec` in the ``.play`` file at ``path``."""
    return parse(Path(path).read_text())


__all__ = ["Spec", "SpecError", "load", "parse"]
