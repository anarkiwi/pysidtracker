"""Carry chains: byte-wise arithmetic linked by ``carry``/``borrow`` that one wider operation
states once. :func:`find` follows each carry through ``let``s, proc parameters and
``state``/signals, joins the links into chains, classifies them and proves each unit's byte
lanes equal to one 16-bit add or subtract with z3.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Iterator, List, Optional, Set, Tuple

import z3

from .syntax import Loc, Spec

CARRIES = ("carry", "borrow")
STATEMENTS = ("let", "assign", "if", "call", "while", "for", "return", "match", "fire")
_SUM = ("+", "-")
_COMPOUND = {"+=": "+", "-=": "-", "&=": "&", "|=": "|", "^=": "^"}


def _tagged(node) -> bool:
    return isinstance(node, tuple) and bool(node) and isinstance(node[0], str)


def src(node) -> str:
    """A compact source text of an expression."""
    if not _tagged(node):
        return "?"
    tag = node[0]
    if tag in ("name", "num", "bool"):
        return str(node[2])
    if tag == "index":
        return f"{node[2]}[{src(node[3])}]"
    if tag == "attr":
        return f"{node[2]}.{node[3]}"
    if tag == "func":
        return f"{node[2]}({', '.join(src(a) for a in node[3])})"
    if tag == "!":
        return f"!{src(node[2])}"
    if len(node) == 4 and _tagged(node[2]) and _tagged(node[3]):
        return f"({src(node[2])} {tag} {src(node[3])})"
    return f"{tag}({', '.join(src(n) for n in node[2:] if _tagged(n))})"


def _root(text: str) -> str:
    return text.split("[")[0].split(".")[0]


@dataclasses.dataclass
class Link:
    """One ``carry``/``borrow``: its operands ``(base, operand)``, the lane its sum lands
    in and its uses ``(unit, line, kind, destination)``."""

    ident: int
    unit: str
    line: int
    kind: str
    operands: Tuple[str, str]
    args: tuple = ()
    dest: Optional[str] = None
    dest_line: int = 0
    uses: List[Tuple[str, int, str, Optional[str]]] = dataclasses.field(
        default_factory=list
    )


@dataclasses.dataclass
class Assign:
    """An assignment or ``let``: its target and the terms of its top-level sum, as text
    and as syntax (an updated target first)."""

    unit: str
    line: int
    target: str
    terms: List[str]
    nodes: list


class _Finder:
    """Two passes over every proc and handler: links, their uses and the assignments."""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, spec: Spec):
        self.units: Dict[str, tuple] = {}
        self.procs: Dict[str, str] = {}
        self.types: Dict[str, str] = {s[2]: s[3] for s in spec.states}
        self.types.update((c[2], c[3]) for c in spec.cells)
        self.shared = set(self.types)
        for machine in spec.machines.values():
            for item in machine[6]:
                if item[0] in ("own", "signal", "cell"):
                    self.shared.add(item[2])
                    self.types[item[2]] = item[3]
                elif item[0] in ("proc", "handler"):
                    self.units[f"{machine[2]}.{item[2]}"] = item
        self.units.update((f"proc {n}", p) for n, p in spec.procs.items())
        for unit, item in self.units.items():
            if item[0] == "proc":
                self.procs[item[2]] = unit
        self.links: Dict[int, Link] = {}
        self.parent: Dict[int, int] = {}
        self.carried: Dict[str, Set[int]] = {}
        self.params: Dict[Tuple[str, str], Set[int]] = {}
        self.assigns: Dict[Tuple[str, int, str], Assign] = {}
        self.unit = ""

    def find(self, ident: int) -> int:
        """The chain ``ident`` belongs to."""
        while self.parent[ident] != ident:
            self.parent[ident] = self.parent[self.parent[ident]]
            ident = self.parent[ident]
        return ident

    def run(self) -> None:
        """Two passes, so the second sees carries passed in parameters and state."""
        for _ in range(2):
            for unit, item in self.units.items():
                self.unit = unit
                params = item[3] if item[0] == "proc" else []
                env = {p: set(self.params.get((unit, p), ())) for p, _t in params}
                self.block(item[-1], env, {})

    def link(self, node) -> int:
        """The link of the ``carry``/``borrow`` call ``node``."""
        ident = id(node)
        if ident not in self.links:
            ops = (src(node[3][0]), src(node[3][1]))
            self.links[ident] = Link(
                ident, self.unit, int(node[1]), node[2], ops, tuple(node[3][:2])
            )
            self.parent[ident] = ident
        return ident

    def carry_of(self, node, env) -> Set[int]:
        """The links a truth value ``node`` carries."""
        if not _tagged(node):
            return set()
        if node[0] == "func" and node[2] in CARRIES:
            return {self.link(node)}
        if node[0] == "!":
            return self.carry_of(node[2], env)
        if node[0] == "name":
            return set(env.get(node[2], self.carried.get(node[2], ())))
        return set()

    def use(self, ids: Set[int], kind: str, line, dest: Optional[str]) -> None:
        """Records a use of the carries ``ids``."""
        for ident in ids:
            self.links[ident].uses.append((self.unit, int(line or 0), kind, dest))

    def expr(self, node, env, alias, dest: Optional[str]) -> Set[int]:
        """Scans ``node``: links, carry-in, arithmetic and condition uses; returns the
        carries its value adds in."""
        if isinstance(node, list):
            return set().union(set(), *(self.expr(n, env, alias, dest) for n in node))
        if not _tagged(node):
            return set()
        tag, line = node[0], node[1]
        if tag == "func" and node[2] in CARRIES:
            ident = self.link(node)
            self.links[ident].operands = tuple(
                alias.get(src(a), src(a)) for a in node[3][:2]
            )
            if len(node[3]) > 2:
                cin = self.carry_of(node[3][2], env)
                self.use(cin, "carry-in", line, dest)
                for other in cin:
                    self.parent[self.find(ident)] = self.find(other)
            self.expr(node[3], env, alias, dest)
            return set()
        if tag == "func" and (
            node[2] in ("u8", "u16")
            or node[2] == "ite"
            and _tagged(node[3][1])
            and node[3][1][0] == "num"
        ):
            ids = self.carry_of(node[3][0], env)
            if ids:
                self.use(ids, "lane", line, dest)
                self.expr(node[3], env, alias, dest)
                return ids
        if tag == "func" and node[2] == "ite":
            self.use(self.carry_of(node[3][0], env), "condition", line, None)
        if tag == "func" and node[2] in self.procs:
            self.call(node[2], node[3], env, line)
        return set().union(
            set(),
            *(
                self.expr(i, env, alias, dest)
                for i in node[2:]
                if not isinstance(i, str)
            ),
        )

    def call(self, proc: str, args, env, line) -> None:
        """Binds carries passed to ``proc``'s parameters."""
        unit = self.procs[proc]
        for (param, _t), arg in zip(self.units[unit][3], args):
            ids = self.carry_of(arg, env)
            if ids:
                self.use(ids, "call", line, f"{unit}({param})")
                self.params.setdefault((unit, param), set()).update(ids)

    def block(self, stmts, env, alias) -> None:
        """Walks a statement list in order."""
        for stmt in stmts if isinstance(stmts, list) else [stmts]:
            if _tagged(stmt):
                self.stmt(stmt, env, alias)

    def stmt(self, node, env, alias) -> None:
        """One statement."""
        tag, line = node[0], node[1]
        if tag in ("let", "assign"):
            self.assign(node, env, alias)
        elif tag == "if":
            for cond, arm in node[2]:
                self.use(self.carry_of(cond, env), "condition", line, None)
                self.expr(cond, env, alias, None)
                self.block(arm, dict(env), dict(alias))
            self.block(node[3], dict(env), dict(alias))
        elif tag == "call" and node[2] in self.procs:
            self.call(node[2], node[3], env, line)
            self.expr(node[3], env, alias, None)
        else:
            for item in node[2:]:
                if isinstance(item, list) and any(
                    _tagged(i) and i[0] in STATEMENTS for i in item
                ):
                    self.block(item, dict(env), dict(alias))
                elif not isinstance(item, str):
                    self.expr(item, env, alias, None)

    def assign(self, node, env, alias) -> None:
        """A ``let`` or assignment: binds a carry, records an escape and the sum's terms."""
        let = node[0] == "let"
        target = node[2] if let else src(node[2])
        name = node[2] if let else _root(target)
        op, rhs = (":=", node[4]) if let else (node[3], node[4])
        ids = self.carry_of(rhs, env) if op == ":=" else set()
        adds = self.expr(rhs, env, alias, alias.get(target, target))
        if ids and not adds:
            if not let and name in self.shared:
                self.use(ids, "escape", node[1], target)
                self.carried.setdefault(name, set()).update(ids)
            env[name] = ids
        else:
            env.pop(name, None)
        nodes = ([node[2]] if op in ("+=", "-=") else []) + _term_nodes(rhs)
        self.assigns[(self.unit, int(node[1]), target)] = Assign(
            self.unit,
            int(node[1]),
            alias.get(target, target),
            [alias.get(src(t), src(t)) for t in nodes],
            nodes,
        )
        if let and _tagged(rhs) and rhs[0] in ("name", "index", "attr"):
            alias[name] = src(rhs)


def _term_nodes(node) -> list:
    """The operands of a top-level sum or difference, left to right."""
    if _tagged(node) and node[0] in _SUM:
        return _term_nodes(node[2]) + _term_nodes(node[3])
    return [node]


def _is_carry_term(node) -> bool:
    """``u8(c)``, ``u16(c)`` or ``ite(c, k, k')`` of a carry, ``!`` or a truth name."""
    if not (_tagged(node) and node[0] == "func" and node[3]):
        return False
    if node[2] == "ite" and not (_tagged(node[3][1]) and node[3][1][0] == "num"):
        return False
    if node[2] not in ("u8", "u16", "ite"):
        return False
    arg = node[3][0]
    return _tagged(arg) and (
        arg[0] in ("!", "name", "bool") or arg[0] == "func" and arg[2] in CARRIES
    )


def _pairs(finder: _Finder, links: List[Link]) -> List[Tuple[str, str]]:
    """``(low, high)`` destinations of the chain's width steps; sets each link's lane."""
    for link in links:
        lanes = [
            (abs(a.line - link.line), a.line, a.target)
            for a in finder.assigns.values()
            if a.unit == link.unit and set(link.operands) <= set(a.terms)
        ]
        if lanes:
            _d, link.dest_line, link.dest = min(lanes)
    return sorted(
        {
            (link.dest, dest)
            for link in links
            for _u, _n, kind, dest in link.uses
            if kind == "lane" and link.dest and dest and dest != link.dest
        }
    )


def find(spec: Spec) -> List[dict]:
    """Every carry chain of ``spec``: units, lines, links, kind, ends, byte pairs, writes of
    each paired variable, and per unit the proof of the fused form."""
    finder = _Finder(spec)
    finder.run()
    groups: Dict[int, List[Link]] = {}
    for ident, link in finder.links.items():
        groups.setdefault(finder.find(ident), []).append(link)
    writes: Dict[str, int] = {}
    for a in finder.assigns.values():
        writes[_root(a.target)] = writes.get(_root(a.target), 0) + 1
    out = []
    for links in groups.values():
        links.sort(key=lambda l: (l.unit, l.line))
        uses = [u for l in links for u in l.uses]
        pairs = _pairs(finder, links)
        nary = any(
            k == "lane" and d == l.dest for l in links for _u, _n, k, d in l.uses
        )
        kinds = {k for _u, _n, k, _d in uses}
        roots = sorted({_root(x) for p in pairs for x in p})
        out.append(
            {
                "units": sorted({l.unit for l in links} | {u for u, *_ in uses}),
                "lines": sorted({l.line for l in links} | {n for _u, n, *_ in uses}),
                "links": [f"{l.kind}({', '.join(l.operands)})" for l in links],
                "kind": "+".join(
                    k for k, on in (("width", pairs), ("n-ary", nary)) if on
                )
                or "flag",
                "ends": sorted(kinds & {"condition", "escape", "call"}) or ["internal"],
                "escapes": sorted({d for _u, _n, k, d in uses if k == "escape"}),
                "pairs": [list(p) for p in pairs],
                "writes": {r: writes.get(r, 0) for r in roots},
                "proofs": {
                    unit: _prove(finder, [l for l in links if l.unit == unit])
                    for unit in sorted({l.unit for l in links})
                },
            }
        )
        out[-1]["saved"] = sum(p["saved"] for p in out[-1]["proofs"].values())
    return sorted(out, key=lambda c: (not c["escapes"], c["units"], c["lines"]))


class _Sym:
    """z3 semantics of straight-line statements over the inputs they read."""

    def __init__(self, types: Dict[str, str]):
        self.types, self.env, self.inputs = types, {}, {}

    def read(self, text: str):
        """The value of ``text`` now: assigned, or a fresh input of its declared type."""
        if text not in self.env:
            typ = self.types.get(_root(text), "u8")
            fresh = (
                z3.Bool(text)
                if typ == "bool"
                else z3.BitVec(text, 16 if typ == "u16" and "." not in text else 8)
            )
            self.inputs[text] = self.env[text] = fresh
        return self.env[text]

    def val(self, node):
        """``node`` as a z3 bit-vector or boolean."""
        tag = node[0]
        if tag == "num":
            return z3.BitVecVal(node[2], 8 if node[2] < 256 else 16)
        if tag == "bool":
            return z3.BoolVal(node[2])
        if tag in ("name", "index", "attr"):
            return self.read(src(node))
        if tag == "!":
            return z3.Not(self.val(node[2]))
        if tag == "func":
            return _builtin(node[2], [self.val(a) for a in node[3]])
        if tag not in _BINARY and tag not in _LOGIC or len(node) != 4:
            raise ValueError(f"syntax {tag}")
        a, b = _widen(self.val(node[2]), self.val(node[3]))
        if tag in _LOGIC:
            return _LOGIC[tag](a, b)
        return _BINARY[tag](a, b)

    def run(self, stmts) -> bool:
        """Executes ``stmts``; False if one is outside the straight-line subset."""
        for node in stmts:
            if node[0] not in ("let", "assign"):
                return False
            rhs = node[4]
            if node[0] == "assign" and node[3] != ":=":
                rhs = (_COMPOUND[node[3]], node[1], node[2], rhs)
            value = self.val(rhs)
            text = node[2] if node[0] == "let" else src(node[2])
            if node[0] == "assign":
                old = self.read(text)
                if z3.is_bv(value) and z3.is_bv(old):
                    value = _resize(value, old.size())
            self.env[text] = value
        return True


_BINARY = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "&": lambda a, b: a & b,
    "|": lambda a, b: a | b,
    "^": lambda a, b: a ^ b,
    "<<": lambda a, b: a << b,
    ">>": z3.LShR,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": z3.ULT,
    "<=": z3.ULE,
    ">": z3.UGT,
    ">=": z3.UGE,
}
_LOGIC = {"&&": z3.And, "||": z3.Or}


def _resize(value, width: int):
    if value.size() > width:
        return z3.Extract(width - 1, 0, value)
    return z3.ZeroExt(width - value.size(), value) if value.size() < width else value


def _widen(a, b):
    if z3.is_bv(a) and z3.is_bv(b):
        w = max(a.size(), b.size())
        return _resize(a, w), _resize(b, w)
    return a, b


def _bit(flag, width: int):
    return z3.If(flag, z3.BitVecVal(1, width), z3.BitVecVal(0, width))


def _builtin(name: str, args):
    """``u8``, ``u16``, ``ite``, ``carry``, ``borrow``."""
    if name in ("u8", "u16"):
        width = 8 if name == "u8" else 16
        return _bit(args[0], width) if z3.is_bool(args[0]) else _resize(args[0], width)
    if name == "ite":
        a, b = _widen(args[1], args[2])
        return z3.If(args[0], a, b)
    if name in CARRIES:
        size = args[0].size()
        a, b = (_resize(x, size + 1) for x in args[:2])
        cin = _bit(args[2], size + 1) if len(args) > 2 else z3.BitVecVal(0, size + 1)
        if name == "carry":
            return z3.UGT(a + b + cin, (1 << size) - 1)
        return z3.ULT(a, b + cin)
    raise ValueError(name)


def _span(finder: _Finder, links: List[Link]) -> Optional[Tuple[list, list]]:
    """The statements of one block from the unit's first link or lane to its last, and the
    ``let``s just before them."""
    lines = _lines(links) | {l.dest_line for l in links if l.dest}
    for block in _blocks(finder.units[links[0].unit][-1]):
        at = [i for i, s in enumerate(block) if int(s[1]) in lines]
        if at and {int(block[i][1]) for i in at} >= lines:
            first = at[0]
            while first and block[first - 1][0] == "let":
                first -= 1
            return block[first : at[0]], block[at[0] : at[-1] + 1]
    return None


def _blocks(node) -> Iterator[list]:
    if isinstance(node, list):
        if node and all(_tagged(s) and isinstance(s[1], Loc) for s in node):
            yield node
        for item in node:
            yield from _blocks(item)
    elif isinstance(node, tuple):
        for item in node[1:] if _tagged(node) else node:
            yield from _blocks(item)


def _prove(finder: _Finder, links: List[Link]) -> dict:
    """``status`` and AST nodes ``saved``: a unit's two lanes against one 16-bit add
    (``carry``) or subtract (``borrow``) of the byte pairs, with no carry in, a constant one
    or an input flag; ``proved: <form>``, ``refuted`` or why not tried."""
    pairs = _pairs(finder, links)
    if len(pairs) != 1 or len({l.kind for l in links}) != 1:
        return {"status": "not tried: not one byte pair", "saved": 0}
    span = _span(finder, links)
    if span is None:
        return {"status": "not tried: lanes in different blocks", "saved": 0}
    lo, hi = pairs[0]
    low = next(l for l in links if l.dest == lo)
    high = next((l for l in links if l.dest == hi), None)
    if high is not None:
        hi_terms = list(high.args)
    else:
        at = {n for l in links for u, n, k, d in l.uses if k == "lane" and d == hi}
        lanes = [
            a
            for a in finder.assigns.values()
            if a.unit == low.unit and a.target == hi and a.line in at
        ]
        if not lanes:
            return {"status": "not tried: high lane not assigned", "saved": 0}
        hi_terms = [t for t in lanes[0].nodes if not _is_carry_term(t)][:3]
        if not 1 <= len(hi_terms) <= 2:
            return {"status": "not tried: high lane operands", "saved": 0}
    item = finder.units[low.unit]
    sym = _Sym({**finder.types, **dict(item[3] if item[0] == "proc" else [])})
    try:
        if not sym.run(span[0]):
            return {"status": "not tried: not straight-line", "saved": 0}
        values = [sym.val(t) for t in (*low.args, *hi_terms)]
        if not all(z3.is_bv(v) for v in values):
            return {"status": "not tried: operands", "saved": 0}
        base_lo, op_lo, base_hi, *op_hi = (_resize(v, 8) for v in values)
        if not sym.run(span[1]) or not all(z3.is_bv(sym.read(t)) for t in (lo, hi)):
            return {"status": "not tried: not straight-line", "saved": 0}
    except (KeyError, ValueError, IndexError, z3.Z3Exception) as exc:
        return {"status": f"not tried: outside the subset ({exc})", "saved": 0}
    before = z3.Concat(base_hi, base_lo)
    operand = z3.Concat(op_hi[0] if op_hi else z3.BitVecVal(0, 8), op_lo)
    after = z3.Concat(_resize(sym.read(hi), 8), _resize(sym.read(lo), 8))
    sign = "+" if low.kind == "carry" else "-"
    want = before + operand if sign == "+" else before - operand
    forms = [(want + k, f" {'+' if k > 0 else '-'} 1" if k else "") for k in (0, 1, -1)]
    for name, flag in sym.inputs.items():
        if z3.is_bool(flag):
            forms.append((want + _bit(flag, 16), f" + {name}"))
            forms.append((want - _bit(z3.Not(flag), 16), f" - !{name}"))
    for value, extra in forms:
        solver = z3.Solver()
        solver.add(after != value)
        if solver.check() == z3.unsat:
            dest = f"{hi}:{lo}"
            base = f"{src(hi_terms[0])}:{low.operands[0]}"
            other = (
                f"{src(hi_terms[1]) if len(hi_terms) > 1 else '0'}:{low.operands[1]}"
            )
            fused = 3 + 2 * (dest != base) + 3 * bool(extra)
            used = _lines(links) | {l.dest_line for l in links}
            nodes = sum(_nodes(st) for st in span[1] if int(st[1]) in used)
            return {
                "status": f"proved: {dest} = {base} {sign} {other}{extra}",
                "saved": max(nodes - fused, 0),
            }
    return {"status": "refuted", "saved": 0}


def _nodes(node) -> int:
    if isinstance(node, (tuple, list)):
        return int(_tagged(node)) + sum(_nodes(n) for n in node)
    return 0


def _lines(links: List[Link]) -> Set[int]:
    return {l.line for l in links} | {
        n for l in links for u, n, k, _d in l.uses if u == l.unit and k == "lane"
    }


__all__ = ["Assign", "CARRIES", "Link", "find", "src"]
