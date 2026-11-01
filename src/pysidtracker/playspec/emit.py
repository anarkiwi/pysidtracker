"""Write a lowered :class:`~.ir.Program` back as playspec source.

:func:`declared` gives a program the zero initialisations its variables need to be
``let``-declared in scope; :func:`emit` writes such a program as a spec whose
elaboration is the program again, up to variable names (:func:`isomorphic`).
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

from dataclasses import replace

from ..errors import SidError
from . import emitobj
from .ir import K, Obj, Op, Program, Rd, V, width
from .lower import mk

TYPES = {1: "bool", 8: "u8", 16: "u16"}
INFIX = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "and": "&",
    "or": "|",
    "xor": "^",
    "shl": "<<",
    "shr": ">>",
    "eq": "==",
    "ne": "!=",
    "ult": "<",
    "ule": "<=",
    "land": "&&",
    "lor": "||",
}
CALLS = {
    "lo": "lo",
    "hi": "hi",
    "word": "word",
    "sext": "sx",
    "ite": "ite",
    "own": "owns",
}
CALLS.update(slt="slt", sle="sle")
REG_NAMES = ("freq_lo", "freq_hi", "pw_lo", "pw_hi", "ctrl", "ad", "sr")
SONG_NAMES = ("fc_lo", "fc_hi", "res_filt", "mode_vol")
RESERVED = frozenset(
    "spec extends layout const cell machine proc schedule invariant assume override"
    " extend patch drop signal control step if elif else while bound for in return"
    " finish goto stream grammar assert native resumes let next mem mem16 true false"
    " lo hi word u8 u16 bool ptr sx zp neg slt sle sgt sge bit ind rol ror ite carry"
    " borrow len member old cpu L sid owns event counter owner walker cursor record fire"
    " react".split()
)


class EmitError(SidError):
    """A program with a construct the emitter cannot state as playspec source."""


def normal(e):
    """``e`` rebuilt bottom-up with :func:`~.lower.mk`, as lowering builds it."""
    if isinstance(e, (K, V)):
        return e
    if isinstance(e, Rd):
        return Rd(normal(e.a))
    if e.op == "vol":
        return Op("vol", 8, (normal(e.a[0]),))
    return mk(e.op, e.w, *(normal(a) for a in e.a))


def _exprs(s) -> tuple:
    tag = s[0]
    if tag == "wr":
        return (s[1], s[2])
    if tag == "set":
        return (s[2],)
    if tag in ("if", "loop", "assert"):
        return (s[1],)
    if tag == "obj":
        return s[4]
    return ()


def _arms(s) -> tuple:
    if s[0] == "if":
        return (s[2], s[3])
    return (s[2],) if s[0] in ("loop", "fire") else ()


def _rearm(s, arms) -> tuple:
    if s[0] == "fire":
        return ("fire", s[1], arms[0])
    return ("if", s[1], *arms) if s[0] == "if" else ("loop", s[1], arms[0], *s[3:])


def _names(e, out: set) -> set:
    if isinstance(e, V):
        out.add(e.n)
    elif isinstance(e, Rd):
        _names(e.a, out)
    elif isinstance(e, Op):
        for a in e.a:
            _names(a, out)
    return out


def _occurrences(stmts, path=(), out=None) -> Dict[str, List[tuple]]:
    """Each variable's statement locations: ``((index, arm), ..., (index, None))``."""
    out = {} if out is None else out
    for i, s in enumerate(stmts):
        here = path + ((i, None),)
        found = set()
        for e in _exprs(s):
            _names(e, found)
        if s[0] == "set":
            found.add(s[1])
        if s[0] == "obj" and s[3]:
            found.add(s[3])
        for name in found:
            out.setdefault(name, []).append(here)
        for a, arm in enumerate(_arms(s)):
            _occurrences(arm, path + ((i, a),), out)
    return out


def _block(stmts, path):
    for i, a in path:
        stmts = _arms(stmts[i])[a]
    return stmts


def _home(stmts, places: List[tuple]) -> Tuple[tuple, int]:
    """``(block path, index)``: the innermost block outside any loop body holding every
    place, and the index of its first statement holding one."""
    common: list = []
    for level in zip(*places):
        if len(set(level)) != 1 or level[0][1] is None:
            break
        common.append(level[0])
    for depth in range(len(common)):
        if _block(stmts, common[:depth])[common[depth][0]][0] == "loop":
            common = common[:depth]
            break
    depth = len(common)
    return tuple(common), min(p[depth][0] for p in places)


def declared(body, keep: Iterable[str] = ()) -> list:
    """``body`` with ``v := 0`` before the first statement of each variable's home block
    that does not itself assign ``v`` a value not reading ``v`` (``keep``: variables
    defined outside the body)."""
    keep = frozenset(keep)
    inserts: Dict[tuple, Dict[int, list]] = {}
    for name, places in _occurrences(body).items():
        if name in keep:
            continue
        block, index = _home(body, places)
        first = _block(body, block)[index]
        if (
            first[0] == "set"
            and first[1] == name
            and name not in _names(first[2], set())
        ):
            if (block + ((index, None),)) in places:
                continue
        w = next(width(v) for v in _vars(body, name))
        inserts.setdefault(block, {}).setdefault(index, []).append(
            ("set", name, K(0, w))
        )
    return _insert(body, (), inserts)


def _vars(body, name):
    for s in _walk(body):
        if s[0] == "set" and s[1] == name:
            yield s[2]
        if s[0] == "obj" and s[3] == name:
            yield K(0, 1)
        for e in _exprs(s):
            stack = [e]
            while stack:
                n = stack.pop()
                if isinstance(n, V) and n.n == name:
                    yield n
                stack.extend(
                    n.a if isinstance(n, Op) else (n.a,) if isinstance(n, Rd) else ()
                )


def _walk(stmts):
    for s in stmts:
        yield s
        for arm in _arms(s):
            yield from _walk(arm)


def _insert(stmts, path, inserts) -> list:
    out = []
    for i, s in enumerate(stmts):
        out += sorted(inserts.get(path, {}).get(i, []), key=lambda t: t[1])
        if _arms(s):
            arms = [
                _insert(arm, path + ((i, a),), inserts)
                for a, arm in enumerate(_arms(s))
            ]
            s = _rearm(s, arms)
        out.append(s)
    return out


def _stem(name: str, reg: Dict[int, str]) -> str:
    """A readable base name for IR variable ``name``."""
    found = re.fullmatch(r"(o|h|last|seen)(\d*)_(\d+)", name)
    if found:
        kind = {"o": "out", "h": "wrote"}.get(found.group(1), found.group(1))
        k = found.group(2) if found.group(2) not in ("", "1") else ""
        return f"{kind}{k}_{reg.get(int(found.group(3)), 'native')}"
    if name.startswith("g_"):
        return "cond"
    if name.startswith("cpu_"):
        return name
    stem = re.sub(r"_\d+$", "", name)
    stem = re.sub(r"^(l|sig|a|r)_", "", stem)
    return stem or "v"


def _ident(text: str) -> str:
    text = re.sub(r"\W", "", text.replace("[", "").replace("]", "").replace(".", "_"))
    return text if text and not text[0].isdigit() else f"v{text}"


class _Writer:
    """Source text of one program."""

    def __init__(self, program: Program, base: int):
        self.base = base
        self.cells = [c for c in program.cells if c.typ == "u8" and len(c.place) == 1]
        self.used: Dict[str, object] = {}
        self.consts: Dict[str, int] = {}
        reg = {r: f"v{r // 7}_{REG_NAMES[r % 7]}" for r in range(21)}
        reg.update((21 + i, n) for i, n in enumerate(SONG_NAMES))
        self.reg = reg
        self.vars: Dict[str, str] = {}
        self.taken = set(RESERVED)
        self.lets: Dict[int, None] = {}
        self.widths: Dict[str, int] = {}
        self.objects = program.objects
        self.named: Dict[str, str] = {}

    def declarations(self) -> List[str]:
        """The cells and declarations of the objects the body reacts."""
        out: List[str] = []
        for oid, name in self.named.items():
            obj = self.objects[oid]
            cells = {}
            for role, loc in obj.locs.items():
                cells[role] = self.fresh(f"{name}_{role}")
                out.append(emitobj.cell(cells[role], loc))
            out += emitobj.declare(obj, cells, name) + [""]
        return out

    def fresh(self, stem: str) -> str:
        """``stem``, suffixed until unused."""
        stem, n = _ident(stem), 1
        name = stem
        while name in self.taken:
            n += 1
            name = f"{stem}_{n}"
        self.taken.add(name)
        return name

    def var(self, name: str) -> str:
        """The source name of IR variable ``name``."""
        if name.startswith("cpu_"):
            return f"cpu.{name[4:]}"
        if name not in self.vars:
            cell = re.fullmatch(r"m_([0-9a-f]{4})", name)
            stem = self.cell_name(int(cell.group(1), 16)) if cell else None
            self.vars[name] = self.fresh(stem or _stem(name, self.reg))
        return self.vars[name]

    def cell_name(self, addr: int) -> Optional[str]:
        """The declared name of the scalar cell at ``addr``, if any."""
        for c in self.cells:
            if c.length is None and c.place[0] == addr:
                return f"{c.owner}_{c.name}" if c.owner else c.name
        return None

    def cell(self, addr: int, scalar: bool, volatile: bool, write: bool):
        """The cell a read or store at ``addr`` names, or None."""
        for c in self.cells:
            if (c.length is None) != scalar or c.place[0] != addr:
                continue
            if (
                ("volatile" in c.attrs) != volatile
                or write
                and c.attrs & {"ro", "volatile"}
            ):
                continue
            key = (c.owner, c.name)
            if key not in self.used:
                self.used[key] = (
                    self.fresh(f"{c.owner}_{c.name}" if c.owner else c.name),
                    c,
                )
            return self.used[key][0]
        return None

    def place(self, a, volatile: bool = False, write: bool = False) -> Optional[str]:
        """``name`` or ``name[index]`` for the address ``a``, None where no cell fits."""
        if isinstance(a, K) and self.base <= a.v < self.base + 25 and not volatile:
            name = self.reg[a.v - self.base].upper()
            self.consts[name] = a.v - self.base
            self.used.setdefault(("", "sid"), ("sid", None))
            return f"sid[{name}]"
        if isinstance(a, K):
            return self.cell(a.v, True, volatile, write)
        if isinstance(a, Op) and a.op == "add" and isinstance(a.a[0], K):
            index = a.a[1]
            if isinstance(index, Op) and index.op == "zext" and width(index.a[0]) == 8:
                index = index.a[0]
            name = self.cell(a.a[0].v, False, volatile, write)
            if name is not None:
                return f"{name}[{self.ex(index)}]"
        return None

    def ex(self, e, typed: bool = True) -> str:
        """Source of ``e``; an integer constant is bare where ``typed`` context gives
        it its width."""
        if isinstance(e, K):
            if e.w == 1:
                return "true" if e.v else "false"
            text = f"{e.v:#x}" if e.v > 9 else str(e.v)
            return text if typed or e.w == 8 else f"u16({text})"
        if isinstance(e, V):
            self.widths[e.n] = e.w
            return self.var(e.n)
        if isinstance(e, Rd):
            return self.place(e.a) or f"mem[{self.ex(e.a)}]"
        if e.op == "vol":
            name = self.place(e.a[0], volatile=True)
            if name is None:
                raise EmitError("a volatile read outside a declared cell")
            return name
        if e.op == "note":
            return self.note(e)
        args = e.a
        if e.op in INFIX:
            return f"({self.ex(args[0])} {INFIX[e.op]} {self.ex(args[1])})"
        if e.op == "lnot":
            return f"!{self.ex(args[0])}"
        if e.op in ("not", "neg"):
            return f"{'~' if e.op == 'not' else '-'}{self.ex(args[0])}"
        if e.op == "zext":
            return f"u{e.w}({self.ex(args[0])})"
        if e.op == "ite":
            both = all(isinstance(a, K) for a in args[1:])
            parts = [self.ex(args[0])] + [self.ex(a, not both) for a in args[1:]]
            return f"ite({', '.join(parts)})"
        if e.op in CALLS:
            return f"{CALLS[e.op]}({', '.join(self.ex(a) for a in args)})"
        raise EmitError(f"no source form for {e.op}")

    def note(self, e) -> str:
        """A read of the note table cell the ``note`` operator stands for."""
        _clock, part, offset, index = e.a
        for c in self.cells:
            if c.notes is not None and c.length is not None:
                parts = {"lo": 0, "hi": 1, "word": 2}
                if parts.get(c.notes[0], -1) == part.v and c.notes[1] == offset.v:
                    key = (c.owner, c.name)
                    if key not in self.used:
                        self.used[key] = (
                            self.fresh(f"{c.owner}_{c.name}" if c.owner else c.name),
                            c,
                        )
                    return f"{self.used[key][0]}[{self.ex(index)}]"
        raise EmitError("a standard note read without its declared table")

    def block(self, stmts, depth: int) -> List[str]:
        """Source lines of ``stmts``."""
        pad = "  " * depth
        out: List[str] = []
        for s in stmts:
            tag = s[0]
            if tag == "set":
                rhs = self.ex(s[2])
                if id(s) in self.lets:
                    w = width(s[2])
                    out.append(f"{pad}let {self.var(s[1])}: {TYPES[w]} = {rhs};")
                else:
                    out.append(f"{pad}{self.var(s[1])} := {rhs};")
            elif tag == "wr":
                target = self.place(s[1], write=True) or f"mem[{self.ex(s[1])}]"
                out.append(f"{pad}{target} := {self.ex(s[2])};")
            elif tag == "if":
                out += self.branch(s, depth, "if")
            elif tag == "loop":
                out.append(f"{pad}while {self.ex(s[1])} bound {s[3]} {{")
                out += self.block(s[2], depth + 1) + [f"{pad}}}"]
            elif tag == "assert":
                out.append(f"{pad}assert {self.ex(s[1])};")
            elif tag == "finish":
                out.append(f"{pad}finish;")
            elif tag == "fire":
                out.append(f"{pad}fire {s[1]} {{")
                out += self.block(s[2], depth + 1) + [f"{pad}}}"]
            elif tag == "obj":
                name = self.named.setdefault(s[1], self.fresh(s[1]))
                ops = ", ".join(self.ex(o) for o in s[4])
                flag = f" -> {self.var(s[3])}" if s[3] else ""
                out.append(f"{pad}react {name} on {s[2]}({ops}){flag};")
            else:
                raise EmitError(f"no source form for a {tag} statement")
        return out

    def branch(self, s, depth: int, keyword: str) -> List[str]:
        """An ``if`` with ``elif`` for an else arm that is a lone ``if``."""
        pad = "  " * depth
        out = [f"{pad}{keyword} {self.ex(s[1])} {{"] + self.block(s[2], depth + 1)
        other = s[3]
        if len(other) == 1 and other[0][0] == "if" and id(other[0]) not in self.lets:
            tail = self.branch(other[0], depth, "elif")
            return out + [f"{pad}}} {tail[0].lstrip()}"] + tail[1:]
        if other:
            out += [f"{pad}}} else {{"] + self.block(other, depth + 1)
        return out + [f"{pad}}}"]


def _lets(body) -> Dict[int, None]:
    """The ``set`` statements that declare their variable (see :func:`declared`)."""
    out: Dict[int, None] = {}
    for name, places in _occurrences(body).items():
        block, index = _home(body, places)
        first = _block(body, block)[index]
        if (
            first[0] == "set"
            and first[1] == name
            and name not in _names(first[2], set())
        ):
            out[id(first)] = None
    return out


def emit(program: Program, name: Optional[str] = None, base: int = 0xD400) -> str:
    """``program`` (a :func:`declared` body, no procs) as playspec source."""
    if program.procs or program.controls:
        raise EmitError("only a flat program without hidden controls is emitted")
    name = _ident(name or program.name)
    body = program.body
    writer = _Writer(program, base)
    writer.lets = _lets(body)
    writer.taken.add(name)
    lines = writer.block(body, 2)
    objects = writer.declarations()
    head = [f"spec {name};", ""]
    head += [
        f"const {k} = {v};"
        for k, v in sorted(writer.consts.items(), key=lambda kv: kv[1])
    ]
    head += [""] if writer.consts else []
    for cname, cell in sorted(
        writer.used.values(), key=lambda t: (t[1] is not None, t[0])
    ):
        if cell is None:
            head.append(f"cell sid: u8[25] @ {base:#x} out;")
            continue
        typ = "u8" if cell.length is None else f"u8[{cell.length}]"
        attrs = [
            a
            for a in ("ro", "out", "operand", "scratch", "volatile")
            if a in cell.attrs
        ]
        if cell.notes is not None:
            attrs.append(f"notes({cell.notes[0]}, {cell.notes[1]})")
        head.append(
            f"cell {cname}: {typ} @ {cell.place[0]:#06x}{''.join(' ' + a for a in attrs)};"
        )
    head += objects
    return "\n".join(
        head
        + ["", f"machine {name} {{", "  step {"]
        + lines
        + ["  }", "}", "", f"schedule {{ {name}; }}", ""]
    )


def _alike(x: Obj, y: Obj) -> bool:
    """Whether two objects are the same up to name, owner, source line and grammar."""
    blank = {"name": "", "owner": "", "line": 0, "grammar": None}
    return replace(x, **blank) == replace(y, **blank)


def isomorphic(a, b, objects=None) -> Optional[str]:
    """None if statement lists ``a`` and ``b`` are equal up to a one-to-one renaming of
    variables and objects, assert messages and loop lines; else the first difference.
    ``objects`` is the pair of tables of the objects the statements react."""
    names: Dict[str, str] = {}
    back: Dict[str, str] = {}
    units: Dict[str, str] = {}

    def same_name(x: str, y: str) -> bool:
        if names.setdefault(x, y) != y or back.setdefault(y, x) != x:
            return False
        return True

    def same(x, y) -> bool:
        if type(x) is not type(y):
            return False
        if isinstance(x, V):
            return x.w == y.w and same_name(x.n, y.n)
        if isinstance(x, K):
            return x == y
        if isinstance(x, Rd):
            return same(x.a, y.a)
        return (
            x.op == y.op
            and x.w == y.w
            and len(x.a) == len(y.a)
            and all(map(same, x.a, y.a))
        )

    def stmts(x, y, where: str) -> Optional[str]:
        if len(x) != len(y):
            return f"{where}: {len(x)} statements against {len(y)}"
        for i, (s, t) in enumerate(zip(x, y)):
            here = f"{where}.{i}"
            if s[0] != t[0]:
                return f"{here}: {s[0]} against {t[0]}"
            if s[0] == "set" and not same_name(s[1], t[1]):
                return f"{here}: assigns {s[1]} against {t[1]}"
            if s[0] == "fire" and s[1] != t[1]:
                return f"{here}: fires {s[1]} against {t[1]}"
            if s[0] == "obj":
                same_flag = (s[3] is None) == (t[3] is None) and (
                    s[3] is None or same_name(s[3], t[3])
                )
                if units.setdefault(s[1], t[1]) != t[1] or not same_flag:
                    return f"{here}: reacts {s[1]} against {t[1]}"
                if (s[2], s[5:], len(s[4])) != (t[2], t[5:], len(t[4])):
                    return f"{here}: {s[1:3]} against {t[1:3]}"
            if not all(map(same, _exprs(s), _exprs(t))):
                return f"{here}: {s} against {t}"
            if s[0] == "loop" and s[3] != t[3]:
                return f"{here}: bound {s[3]} against {t[3]}"
            for arm, (p, q) in enumerate(zip(_arms(s), _arms(t))):
                found = stmts(p, q, f"{here}/{arm}")
                if found:
                    return found
        return None

    found = stmts(a, b, "body")
    if found or objects is None:
        return found
    for mine, theirs in units.items():
        if not _alike(objects[0][mine], objects[1][theirs]):
            return f"object {mine} against {theirs}"
    return None


__all__ = ["EmitError", "declared", "emit", "isomorphic", "normal"]
