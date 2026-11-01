"""Parse playspec source (``playspec.lark``) into a tuple AST.

Every node is ``(tag, line, *fields)``, ``line`` a :class:`Loc`; :func:`parse`
returns a :class:`Spec` whose declarations keep source order. A spec that
``extends`` a base is derived from it here (:func:`derive`), so later stages see
one flat spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple, Union

from lark import Lark, Token, Transformer, v_args
from lark.exceptions import LarkError

from ..errors import SidError


class SpecError(SidError):
    """A playspec that does not parse, type-check or bind."""


class Loc(int):
    """A source line that also knows its spec (``origin``) and character extent."""

    def __new__(cls, line: int, origin: str = "", start: int = 0, end: int = 0):
        loc = super().__new__(cls, line)
        loc.origin, loc.start, loc.end = origin, start, end
        return loc

    def __str__(self) -> str:
        return f"{self.origin}:{int(self)}" if self.origin else str(int(self))

    def __format__(self, spec: str) -> str:
        return str(self) if not spec else int.__format__(self, spec)

    def __eq__(self, other) -> bool:
        return int(self) == int(other) and self.origin == getattr(
            other, "origin", self.origin
        )

    def __ne__(self, other) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash((int(self), self.origin))


@dataclass
class Spec:
    """A parsed spec: its name and declarations by kind, in source order.

    ``base`` names the spec it extends and ``edits`` holds its ``override``,
    ``extend``, ``patch`` and ``drop`` items until :func:`derive` applies them.
    ``sources`` is the text of every spec in the lineage by name.
    """

    name: str = ""
    base: str = ""
    layout: Dict[str, str] = field(default_factory=dict)
    consts: List[Tuple] = field(default_factory=list)
    cells: List[Tuple] = field(default_factory=list)
    procs: Dict[str, Tuple] = field(default_factory=dict)
    machines: Dict[str, Tuple] = field(default_factory=dict)
    events: List[Tuple] = field(default_factory=list)
    objects: List[Tuple] = field(default_factory=list)
    records: List[Tuple] = field(default_factory=list)
    views: List[Tuple] = field(default_factory=list)
    schedule: Tuple = ()
    invariants: List[Tuple] = field(default_factory=list)
    assumes: List[Tuple] = field(default_factory=list)
    edits: List[Tuple] = field(default_factory=list)
    source: str = ""
    sources: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Span:
    """A ``native`` statement's source extent and each argument's (``name = expr``)."""

    start: int
    end: int
    args: Tuple[Tuple[str, Tuple[int, int]], ...] = ()
    origin: str = ""


def _line(meta, origin: str = "") -> Loc:
    return Loc(
        getattr(meta, "line", 0),
        origin,
        getattr(meta, "start_pos", 0),
        getattr(meta, "end_pos", 0),
    )


OBJECTS = ("counter", "owner", "walker", "cursor", "handler")


def _int(text: str) -> int:
    return int(text, 0)


@v_args(meta=True)
class _ToAst(Transformer):
    # pylint: disable=missing-function-docstring,too-many-public-methods
    def __init__(self, origin: str = ""):
        super().__init__()
        self.origin = origin

    def _n(self, tag, meta, *fields):
        return (tag, _line(meta, self.origin), *fields)

    def start(self, meta, items):
        del meta
        spec = Spec()
        for item in items:
            tag = item[0]
            if tag == "spec":
                spec.name, spec.base = item[2], item[3] or ""
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
            elif tag == "event":
                spec.events.append(item)
            elif tag in OBJECTS:
                spec.objects.append(item)
            elif tag == "record":
                spec.records.append(item)
            elif tag == "view":
                spec.views.append(item)
            elif tag == "invariant":
                spec.invariants.append(item)
            elif tag == "assume":
                spec.assumes.append(item)
            else:
                spec.edits.append(item)
        return spec

    def spec_decl(self, meta, c):
        return self._n("spec", meta, str(c[0]), str(c[1]) if len(c) > 1 else "")

    def override(self, meta, c):
        return self._n("override", meta, c[0])

    def extend(self, meta, c):
        return self._n("extend", meta, str(c[0]), list(c[1:]))

    def patch(self, meta, c):
        return self._n("patch", meta, str(c[0]), list(c[1:]))

    def edit(self, meta, c):
        return self._n(str(c[0]), meta, str(c[1]), c[2])

    def drop(self, meta, c):
        return self._n("drop", meta, [str(n) for n in c])

    def handler(self, meta, c):
        return self._n("handler", meta, str(c[0]), c[1], c[2])

    def event(self, meta, c):
        return self._n("event", meta, str(c[0]), str(c[1]))

    def counter(self, meta, c):
        return self._n("counter", meta, str(c[0]), str(c[1]), list(c[2:]))

    def events(self, meta, c):
        del meta
        return [str(n) for n in c]

    def c_on(self, meta, c):
        if len(c) == 1:
            return self._n("c_on", meta, c[0], "reset", None)
        return self._n("c_on", meta, c[0], str(c[1]), c[2])

    def c_wrap(self, meta, c):
        return self._n("c_wrap", meta, c[0], c[1], c[2] and str(c[2]))

    def c_reload(self, meta, c):
        return self._n("c_reload", meta, c[0])

    def owner(self, meta, c):
        return self._n("owner", meta, str(c[0]), str(c[1]), list(c[2:]))

    def o_id(self, meta, c):
        return self._n("o_id", meta, c[0])

    def o_none(self, meta, c):
        return self._n("o_none", meta, c[0])

    def o_claim(self, meta, c):
        return self._n("o_on", meta, c[0], "claim", c[1])

    def o_release(self, meta, c):
        return self._n("o_on", meta, c[0], "release", c[1])

    def walker(self, meta, c):
        return self._n("walker", meta, str(c[0]), list(c[1:]))

    def w_table(self, meta, c):
        width = c[-1] and c[-1][2]
        return self._n("w_table", meta, list(c[:-1]), width)

    def w_width(self, meta, c):
        return self._n("w_width", meta, c[0])

    def w_pos(self, meta, c):
        return self._n("w_pos", meta, str(c[0]))

    def w_count(self, meta, c):
        return self._n("w_count", meta, str(c[0]))

    def w_values(self, meta, c):
        return self._n("w_values", meta, [str(n) for n in c])

    def w_step(self, meta, c):
        return self._n("w_on", meta, c[0], "step")

    def w_reset(self, meta, c):
        return self._n("w_on", meta, c[0], "reset")

    def w_start(self, meta, c):
        return self._n("w_start", meta, c[0])

    def w_row(self, meta, c):
        then = c[4][2] if c[4] else []
        return self._n("w_row", meta, c[0], c[1], c[2], c[3], then)

    def w_then(self, meta, c):
        return self._n("w_then", meta, list(c))

    def wset(self, meta, c):
        del meta
        return str(c[0]), c[1] and str(c[1]), c[2], c[3], c[4]

    def a_sweep(self, meta, c):
        return self._n("sweep", meta, list(c))

    def a_delay(self, meta, c):
        return self._n("delay", meta, list(c))

    def a_set(self, meta, c):
        return self._n("set", meta, list(c))

    def a_jump(self, meta, c):
        return self._n("jump", meta, c[0], c[1:3], c[3])

    def a_stop(self, meta, c):
        del c
        return self._n("stop", meta)

    def cursor(self, meta, c):
        grammar = [g for g in c[4:] if not isinstance(g[1], Loc)]
        clauses = [g for g in c[4:] if isinstance(g[1], Loc)]
        return self._n(
            "cursor",
            meta,
            str(c[0]),
            c[1],
            str(c[2]),
            c[3] is not None,
            grammar,
            clauses,
        )

    def k_next(self, meta, c):
        return self._n("k_on", meta, c[0], "next")

    def k_reset(self, meta, c):
        return self._n("k_on", meta, c[0], "reset")

    def k_last(self, meta, c):
        del meta, c
        return True

    def k_settle(self, meta, c):
        return self._n("k_on", meta, c[0], "settle")

    def k_run(self, meta, c):
        return self._n("k_run", meta, c[0], c[1], c[2], str(c[3]))

    def k_field(self, meta, c):
        return self._n("k_field", meta, str(c[0]), c[1], list(c[2:]))

    def kout(self, meta, c):
        del meta
        return str(c[0]), c[1], c[2]

    def record(self, meta, c):
        return self._n("record", meta, str(c[0]), c[1], c[2], list(c[3:]))

    def view(self, meta, c):
        return self._n("view", meta, str(c[0]), c[1], list(c[2:]))

    def vfield(self, meta, c):
        return self._n("vfield", meta, str(c[0]), c[1])

    def rfield(self, meta, c):
        return self._n("rfield", meta, str(c[0]), str(c[1]), c[2])

    def fire(self, meta, c):
        return self._n("fire", meta, str(c[0]), None)

    def fire_block(self, meta, c):
        return self._n("fire", meta, str(c[0]), c[1])

    def react(self, meta, c):
        args = [a for a in c[2:-1] if a is not None]
        return self._n("react", meta, str(c[0]), str(c[1]), args, c[-1] and str(c[-1]))

    def labeled(self, meta, c):
        return self._n("label", meta, str(c[0]), c[1])

    def group(self, meta, c):
        return self._n("group", meta, list(c))

    def layout(self, meta, c):
        return self._n("layout", meta, c)

    def lfield(self, meta, c):
        del meta
        return [str(t) for t in c[:-1]], str(c[-1])

    def const(self, meta, c):
        return self._n("const", meta, str(c[0]), c[1])

    def cell(self, meta, c):
        name, (ctype, length), place, *attrs = c
        notes = [a for a in attrs if isinstance(a, tuple)]
        return self._n(
            "cell",
            meta,
            str(name),
            ctype,
            length,
            place,
            {str(a) for a in attrs if isinstance(a, Token)},
            notes[0] if notes else None,
        )

    def notes(self, meta, c):
        del meta
        return str(c[0]), c[1]

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

    def eblock(self, meta, c):
        del meta
        return list(c)

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
        args = [a for a in c[1:-1] if a is not None]
        span = Span(
            meta.start_pos,
            meta.end_pos,
            tuple((r, s) for r, _e, s in args),
            self.origin,
        )
        return self._n("native", meta, c[0], [(r, e) for r, e, _s in args], c[-1], span)

    def narg(self, meta, c):
        return str(c[0]), c[1], (meta.start_pos, meta.end_pos)

    def finish(self, meta, c):
        del c
        return self._n("finish", meta)

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
    options = dict(parser="lalr", propagate_positions=True, maybe_placeholders=True)
    try:
        return Lark(text, cache=True, **options)
    except OSError:
        return Lark(text, **options)


Resolver = Union[Callable[[str], Optional[Spec]], Mapping[str, Spec]]


def parse(text: str, resolve: Optional[Resolver] = None) -> Spec:
    """The :class:`Spec` of playspec ``text``; raises :class:`SpecError` on a syntax
    or derivation error. ``resolve`` maps a base name to its :class:`Spec` for a
    spec that ``extends`` one."""
    try:
        tree = _parser().parse(text)
    except LarkError as exc:
        raise SpecError(f"playspec syntax: {exc}") from exc
    decl = next(tree.find_data("spec_decl"), None)
    spec = _ToAst(str(decl.children[0]) if decl else "").transform(tree)
    spec.source = text
    spec.sources = {spec.name: text}
    if not spec.base:
        if spec.edits:
            raise SpecError(
                f"line {spec.edits[0][1]}: {spec.edits[0][0]} needs a base (spec ... extends)"
            )
        return spec
    get = resolve.get if isinstance(resolve, Mapping) else resolve
    base = get(spec.base) if get else None
    if base is None:
        raise SpecError(f"spec {spec.name}: no base spec {spec.base!r}")
    return derive(base, spec)


def _siblings(folder: Path, seen: Tuple[str, ...]) -> Callable[[str], Spec]:
    def resolve(name: str) -> Spec:
        if name in seen:
            raise SpecError(f"spec {name} extends itself")
        path = folder / f"{name}.play"
        if not path.is_file():
            raise SpecError(f"no base spec {name!r} at {path}")
        return parse(path.read_text(), _siblings(folder, seen + (name,)))

    return resolve


def load(path, resolve: Optional[Resolver] = None) -> Spec:
    """The :class:`Spec` in the ``.play`` file at ``path``; a base it ``extends`` is
    ``NAME.play`` beside it unless ``resolve`` says otherwise."""
    path = Path(path)
    return parse(path.read_text(), resolve or _siblings(path.parent, ()))


# -- derivation ------------------------------------------------------------------
_TOP = ("const", "cell", "proc", "machine", "invariant")


class _Deriver:
    """Applies a derived spec's declarations and edits to a copy of its base."""

    def __init__(self, base: Spec, own: Spec):
        self.errors: List[str] = []
        self.top = {
            "const": {d[2]: d for d in base.consts},
            "cell": {d[2]: d for d in base.cells},
            "proc": dict(base.procs),
            "machine": dict(base.machines),
            "invariant": {d[2]: d for d in base.invariants},
        }
        self.own = own

    def error(self, line, message: str) -> None:
        self.errors.append(f"line {line}: {message}")

    def put(self, table: Dict[str, Tuple], item, replace: bool, where: str) -> None:
        """Add ``item`` to ``table`` (``replace``: in its base's place)."""
        name, tag = item[2], item[0]
        old = table.get(name)
        if replace and old is None:
            self.error(item[1], f"override of {tag} {name}: {where} has none")
        elif replace and old[0] != tag:
            self.error(item[1], f"override of {tag} {name}: {where} has a {old[0]}")
        elif not replace and old is not None:
            self.error(
                item[1],
                f"{tag} {name} is already in {where}: say override to replace it",
            )
        else:
            table[name] = item

    def drop(self, tables, node, where: str) -> None:
        """Remove each name in a ``drop`` from whichever of ``tables`` holds it."""
        for name in node[2]:
            hits = [t for t in tables if name in t]
            if len(hits) != 1:
                self.error(
                    node[1], f"drop {name}: {where} has {len(hits)} of that name"
                )
            else:
                del hits[0][name]

    def patch(self, table: Dict[str, Tuple], node, where: str) -> None:
        """Apply a ``patch`` node's edits to the proc it names in ``table``."""
        proc = table.get(node[2])
        if proc is None or proc[0] != "proc":
            self.error(node[1], f"patch {node[2]}: {where} has no such proc")
            return
        body = proc[5]
        for op, line, label, stmts in node[3]:
            body, hits = _edit(body, op, label, ("group", line, stmts))
            if hits != 1:
                self.error(
                    line,
                    f"{op} {label}: proc {node[2]} has {hits} statements so labelled",
                )
        table[node[2]] = proc[:5] + (body,)

    def extend(self, node) -> None:
        """Merge an ``extend machine`` node's members into the base machine."""
        machine = self.top["machine"].get(node[2])
        if machine is None:
            self.error(node[1], f"extend machine {node[2]}: no such machine")
            return
        where = f"machine {node[2]}"
        members = {m[2]: m for m in machine[6]}
        for item in node[3]:
            tag = item[0]
            if tag == "override":
                self.put(members, item[2], True, where)
            elif tag == "patch":
                self.patch(members, item, where)
            elif tag == "drop":
                self.drop([members], item, where)
            else:
                self.put(members, item, False, where)
        self.top["machine"][node[2]] = machine[:6] + (list(members.values()),)

    def apply(self, base: Spec) -> Spec:
        """The derived spec."""
        own = self.own
        layout = dict(base.layout)
        for name, kind in own.layout.items():
            if layout.setdefault(name, kind) != kind:
                self.errors.append(
                    f"layout {name}: {kind} here, {layout[name]} in base"
                )
        for item in [
            *own.consts,
            *own.cells,
            *own.procs.values(),
            *own.machines.values(),
            *own.invariants,
        ]:
            self.put(self.top[item[0]], item, False, f"base {base.name}")
        declared = {
            "event": [*base.events, *own.events],
            "object": [*base.objects, *own.objects],
            "record": [*base.records, *own.records],
            "view": [*base.views, *own.views],
        }
        for kind, items in declared.items():
            seen: Dict[str, Tuple] = {}
            for item in items:
                if seen.setdefault(item[2], item) is not item:
                    self.error(item[1], f"{item[0]} {item[2]} is already in base")
        schedule = base.schedule
        if own.schedule:
            if schedule:
                self.error(own.schedule[1], "the base has a schedule: say override")
            schedule = own.schedule
        for node in own.edits:
            tag = node[0]
            if tag == "override" and node[2][0] == "schedule":
                if not schedule:
                    self.error(node[1], "override of schedule: the base has none")
                schedule = node[2]
            elif tag == "override" and node[2][0] in _TOP and node[2][2] != "step":
                self.put(self.top[node[2][0]], node[2], True, f"base {base.name}")
            elif tag == "override":
                self.error(node[1], f"override of {node[2][0]} outside extend machine")
            elif tag == "extend":
                self.extend(node)
            elif tag == "patch":
                self.patch(self.top["proc"], node, f"base {base.name}")
            else:
                self.drop([self.top[k] for k in _TOP], node, f"base {base.name}")
        if self.errors:
            raise SpecError("\n".join(self.errors))
        return Spec(
            name=own.name,
            base=base.name,
            layout=layout,
            consts=list(self.top["const"].values()),
            cells=list(self.top["cell"].values()),
            procs=self.top["proc"],
            machines=self.top["machine"],
            events=declared["event"],
            objects=declared["object"],
            records=declared["record"],
            views=declared["view"],
            schedule=schedule,
            invariants=list(self.top["invariant"].values()),
            assumes=[*base.assumes, *own.assumes],
            source=own.source,
            sources={**base.sources, **own.sources},
        )


def _edit(stmts: list, op: str, label: str, group: tuple) -> Tuple[list, int]:
    """``stmts`` with ``group`` put ``before``/``after``/in place of (``replace``)
    the statement labelled ``label``, at any depth, and how many such labels."""
    out: list = []
    hits = 0
    for stmt in stmts:
        if stmt[0] == "label" and stmt[2] == label:
            hits += 1
            if op == "replace":
                out.append(stmt[:3] + (group,))
            else:
                out += [group, stmt] if op == "before" else [stmt, group]
            continue
        stmt, found = _descend(stmt, op, label, group)
        hits += found
        out.append(stmt)
    return out, hits


def _descend(stmt: tuple, op: str, label: str, group: tuple) -> Tuple[tuple, int]:
    """``stmt`` with :func:`_edit` applied to the statement lists inside it."""
    tag = stmt[0]

    def sub(body):
        return _edit(body, op, label, group)

    if tag in ("group", "while", "for", "label"):
        body = [stmt[-1]] if tag == "label" else stmt[-1]
        new, hits = sub(body)
        if tag == "label":
            new = new[0] if len(new) == 1 else ("group", stmt[1], new)
        return stmt[:-1] + (new,), hits
    if tag in ("if", "in"):
        arms, hits = [], 0
        for key, body in stmt[3 if tag == "in" else 2]:
            new, found = sub(body)
            arms.append((key, new))
            hits += found
        if tag == "in":
            return stmt[:3] + (arms,), hits
        orelse, found = sub(stmt[3])
        return stmt[:2] + (arms, orelse), hits + found
    return stmt, 0


def derive(base: Spec, own: Spec) -> Spec:
    """``own`` (a spec that extends ``base``) applied to ``base``: declarations it
    adds, ``override``s replacing the base's in place, ``extend machine`` merging
    members, ``patch`` edits at labelled statements and ``drop``s. Raises
    :class:`SpecError` listing every edit that does not apply exactly."""
    if own.name in base.sources:
        raise SpecError(f"spec {own.name} extends itself")
    return _Deriver(base, own).apply(base)


__all__ = ["Loc", "Span", "Spec", "SpecError", "derive", "load", "parse"]
