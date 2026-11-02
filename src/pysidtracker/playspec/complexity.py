"""Where a playspec's complexity is, to triage what to simplify.

:func:`analyse` measures the spec as written (units, clones, layout uses, state coupling,
addressing a binding could take over); :func:`cover` counts per statement the frames and tunes
that ran it; :func:`report` joins them, :func:`delta` compares two reports.
"""

from __future__ import annotations

import dataclasses
import re
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from . import chains as carry_chains
from .ir import K
from .lower import Lowerer
from .syntax import Loc, Spec, load

BRANCHES = ("sif", "while", "for", "sfor", "w_row", "urow", "&&", "||")
NESTS = ("if", "sif", "while", "for", "sfor", "w_row", "urow")
DECLS = ("consts", "cells", "states", "records", "views", "tables", "outputs", "events")
MACHINE_DECLS = ("own", "signal", "cell", "record")
PROBE = "cov_"


def _tagged(node) -> bool:
    return isinstance(node, tuple) and bool(node) and isinstance(node[0], str)


def _items(node) -> Iterable:
    return node[1:] if _tagged(node) else node


def _lines(node) -> Iterator[int]:
    if isinstance(node, Loc):
        yield int(node)
    elif isinstance(node, (tuple, list)):
        for item in node:
            yield from _lines(item)


def _is_wset(node) -> bool:
    """An object row's ``then``/``set`` item ``(name, sub, value, if, when)``."""
    return _tagged(node) and len(node) == 5 and node[1] is None


def _layout(node) -> Optional[str]:
    if _tagged(node) and node[0] in ("attr", "attr_index") and node[2] == "L":
        return node[3]
    return None


def _guard_fields(node) -> Tuple[int, ...]:
    """The field indices of ``node`` that hold a condition."""
    if node[0] in ("sif", "while"):
        return (2,)
    if node[0] == "w_row":
        return (4,)
    if _is_wset(node):
        return tuple(i for i in (3, 4) if node[i] is not None)
    return ()


def _is_ite(node) -> bool:
    return node[0] == "func" and node[2] == "ite"


def _decisions(node) -> int:
    if node[0] == "if":
        return len(node[2])
    if _is_ite(node):
        return 1
    return len(_guard_fields(node)) if _is_wset(node) else int(node[0] in BRANCHES)


def key(line: Loc, tag: str) -> str:
    """The name of a statement: ``origin:line:tag``."""
    return f"{line.origin}:{int(line)}:{tag}"


@dataclasses.dataclass
class Unit:
    """A unit's AST nodes, decisions, nesting depth and statement keys."""

    name: str
    line: int
    nodes: int = 0
    decisions: int = 0
    depth: int = 0
    statements: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Analysis:
    """:func:`analyse`'s measures; ``text`` the spec's source lines."""

    spec: str
    units: Dict[str, Unit]
    clones: List[dict]
    layout: Dict[str, dict]
    coupling: Dict[str, dict]
    addressing: List[dict]
    text: List[str]
    chains: List[dict] = dataclasses.field(default_factory=list)

    def totals(self) -> dict:
        """Whole-spec counts."""
        units = self.units.values()
        return {
            "units": len(self.units),
            "nodes": sum(u.nodes for u in units),
            "decisions": sum(u.decisions for u in units),
            "statements": sum(len(u.statements) for u in units),
            "clone_savings": sum(c["savings"] for c in self.clones if not c["nested"]),
            "layout_fields": len(self.layout),
            "flag_guards": sum(
                f["guard"] for f in self.layout.values() if f["kind"] == "flag"
            ),
            "multi_writer_state": sum(
                len(c["writers"]) > 1 for c in self.coupling.values()
            ),
            "addressing": len(self.addressing),
            "chains": len(self.chains),
            "chains_proved": sum(
                p["status"].startswith("proved")
                for c in self.chains
                for p in c["proofs"].values()
            ),
            "chain_savings": sum(c["saved"] for c in self.chains),
            "escaping_chains": sum(bool(c["escapes"]) for c in self.chains),
        }

    def excerpt(self, line: int) -> str:
        """Source line ``line``, stripped and cut."""
        return self.text[line - 1].strip()[:72] if 0 < line <= len(self.text) else ""


def _units(spec: Spec) -> Iterator[Tuple[str, object]]:
    yield "decls", [getattr(spec, f) for f in DECLS]
    for obj in spec.objects:
        yield f"object {obj[2]}", obj
    for name, proc in spec.procs.items():
        yield f"proc {name}", proc
    for machine in spec.machines.values():
        decls = [i for i in machine[6] if i[0] in MACHINE_DECLS]
        if decls:
            yield f"{machine[2]} decls", decls
        for item in machine[6]:
            if item[0] not in MACHINE_DECLS:
                yield f"{machine[2]}.{item[2]}", item
    if spec.schedule:
        yield "schedule", spec.schedule


class _Walk:
    """One pass over the units: measures, hash-consed subtrees and name uses."""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, spec: Spec, shape: bool):
        self.spec, self.shape = spec, shape
        owned = [i for m in spec.machines.values() for i in m[6]]
        decls = [
            *spec.states,
            *spec.cells,
            *(i for i in owned if i[0] in MACHINE_DECLS),
        ]
        self.names = {d[2] for d in decls if d[0] != "record"}
        self.wide = {d[2] for d in decls if d[0] == "own" and d[3] == "u16"}
        self.ids: Dict[tuple, int] = {}
        self.occurs: Dict[int, list] = defaultdict(list)
        self.layout: Dict[str, Counter] = defaultdict(Counter)
        self.where: Dict[str, set] = defaultdict(set)
        self.reads: Dict[str, set] = defaultdict(set)
        self.writes: Dict[str, set] = defaultdict(set)
        self.addressing: List[dict] = []
        self.unit: Optional[Unit] = None
        self.serial = 0
        self.up: Dict[int, int] = {}

    def cons(self, node, context: str, parent: int, depth: int) -> Tuple[tuple, int]:
        """The hash-cons key and AST size of ``node``, recording it as an occurrence."""
        if isinstance(node, Loc) or node is None:
            return ("-",), 0
        if isinstance(node, str):
            return ("$" if self.shape and node in self.names else node,), 0
        if isinstance(node, (set, frozenset, dict)):
            return (repr(sorted(node)),), 0
        if not isinstance(node, (tuple, list)):
            return (node,), 0
        self.serial += 1
        me, tagged = self.serial, _tagged(node)
        self.up[me] = parent
        contexts = [context] * len(node)
        if tagged:
            self.uses(node, context)
            self.unit.nodes += 1
            self.unit.decisions += _decisions(node)
            if node[0] in NESTS:
                depth += 1
                self.unit.depth = max(self.unit.depth, depth)
            for i in _guard_fields(node):
                contexts[i] = "guard"
        if tagged and node[0] == "if":
            parts = [self.cons(item, context, me, depth) for item in node[:2]]
            for cond, block in node[2]:
                parts.append(self.cons(cond, "guard", me, depth))
                parts.append(self.cons(block, context, me, depth))
            parts.append(self.cons(node[3], context, me, depth))
        elif tagged and _is_ite(node):
            args = node[3]
            parts = [self.cons(item, context, me, depth) for item in node[:3]]
            parts.append(self.cons(args[0], "guard", me, depth))
            parts += [self.cons(a, context, me, depth) for a in args[1:]]
        else:
            parts = [self.cons(n, c, me, depth) for n, c in zip(node, contexts)]
        size = int(tagged) + sum(s for _k, s in parts)
        cid = self.ids.setdefault((tagged, tuple(k for k, _s in parts)), len(self.ids))
        lines = list(_lines(node))
        if lines:
            self.occurs[cid].append((self.unit.name, min(lines), size, parent, me))
        return (cid,), size

    def uses(self, node, context: str) -> None:
        """The layout, state and addressing uses ``node`` makes itself."""
        tag, unit = node[0], self.unit.name
        field = _layout(node)
        if field is not None:
            self.layout[field][context] += 1
            self.where[field].add(unit)
        if tag in ("name", "attr") and node[2] in self.names:
            self.reads[node[2]].add(unit)
        elif tag == "assign":
            lval = node[2]
            while _tagged(lval) and not isinstance(lval[2], str):
                lval = lval[2]
            self.writes[lval[2]].add(unit)
        elif _is_wset(node):
            self.writes[node[0]].add(unit)
        elif tag in ("w_pos", "w_count", "w_values"):
            for name in node[2] if isinstance(node[2], list) else [node[2]]:
                self.writes[name].add(unit)
                self.reads[name].add(unit)
        elif tag == "owner":
            self.writes[node[3]].add(unit)
        self.addresses(node, context)

    def addresses(self, node, context: str) -> None:
        """Addressing a binding could take over: relocation outside cell placement,
        split pointer tables, word assembly in code, tables placed at u16 state."""
        tag, line = node[0], node[1]
        kinds = self.spec.layout
        if tag == "+" and context != "decl":
            if any(kinds.get(_layout(a) or "") == "addr" for a in node[2:]):
                self.found("relocation", line, _text(node))
        elif tag == "split" and context == "decl":
            self.found("split pointer table", line, "")
        elif tag == "func" and node[2] == "word" and context != "decl":
            self.found("word assembly", line, "word(...)")
        elif tag in ("table", "view", "w_table"):
            at = node[2] if tag == "w_table" else node[3:5]
            for name in sorted(set(_names(at)) & self.wide):
                self.found("pointer state", line, name)

    def found(self, kind: str, line, what: str) -> None:
        """Record one addressing finding."""
        self.addressing.append(
            {"kind": kind, "unit": self.unit.name, "line": int(line or 0), "what": what}
        )


def _names(node) -> Iterator[str]:
    if _tagged(node) and node[0] == "name":
        yield node[2]
    elif isinstance(node, (tuple, list)):
        for item in _items(node):
            yield from _names(item)


def _text(node) -> str:
    if _tagged(node):
        if node[0] in ("name", "num"):
            return str(node[2])
        if _layout(node):
            return f"L.{node[3]}"
        if len(node) == 4 and _tagged(node[2]) and _tagged(node[3]):
            return f"{_text(node[2])} {node[0]} {_text(node[3])}"
    return "..."


def _statements(node) -> Iterator[str]:
    """The key of every statement in ``node``: what :func:`probing` counts."""
    if isinstance(node, list):
        for item in node:
            if (
                _tagged(item)
                and isinstance(item[1], Loc)
                and hasattr(Lowerer, f"st_{item[0]}")
            ):
                yield key(item[1], item[0])
    if isinstance(node, (tuple, list)):
        for item in _items(node):
            yield from _statements(item)


def _clones(occurs: Dict[int, list], up: Dict[int, int], min_nodes: int) -> List[dict]:
    """Maximal repeated subtrees of at least ``min_nodes``: a group is left out when all its
    occurrences sit in occurrences of one other group of the same count, and is ``nested``
    when all sit in occurrences of other groups kept (its savings are then counted there).
    """
    groups = {c: o for c, o in occurs.items() if len(o) > 1}
    owner = {occ[4]: c for c, o in groups.items() for occ in o}

    def outer(serial: int) -> Iterator[int]:
        while serial in up:
            serial = up[serial]
            if serial in owner:
                yield owner[serial]

    kept = {}
    for cid, occ in groups.items():
        first = {next(outer(o[4]), None) for o in occ}
        if (
            len(first) == 1
            and None not in first
            and len(groups[first.pop()]) == len(occ)
        ):
            continue
        if occ[0][2] >= min_nodes:
            kept[cid] = occ
    out = []
    for cid, occ in kept.items():
        size = occ[0][2]
        out.append(
            {
                "size": size,
                "count": len(occ),
                "savings": (len(occ) - 1) * size,
                "nested": all(
                    any(g in kept and g != cid for g in outer(o[4])) for o in occ
                ),
                "at": [(u, line) for u, line, *_ in occ],
            }
        )
    return sorted(out, key=lambda c: (-c["savings"], c["at"]))


def analyse(spec: Spec, shape: bool = False, min_nodes: int = 8) -> Analysis:
    """The static measures of ``spec``; ``shape`` also matches clones that differ only in
    the state, signal and cell names they use."""
    walk = _Walk(spec, shape)
    units: Dict[str, Unit] = {}
    for name, node in _units(spec):
        lines = list(_lines(node))
        walk.unit = units[name] = Unit(name, min(lines) if lines else 0)
        walk.cons(node, "decl" if name.endswith("decls") else "code", 0, 0)
        walk.unit.statements = sorted(set(_statements(node)))
    return Analysis(
        spec=spec.name,
        units=units,
        clones=_clones(walk.occurs, walk.up, min_nodes),
        layout={
            f: {
                "kind": spec.layout.get(f, "?"),
                "guard": c["guard"],
                "code": c["code"],
                "decl": c["decl"],
                "units": sorted(walk.where[f]),
            }
            for f, c in sorted(walk.layout.items())
        },
        coupling={
            n: {
                "writers": sorted(walk.writes.get(n, ())),
                "readers": sorted(walk.reads.get(n, ())),
            }
            for n in sorted(walk.names)
        },
        addressing=walk.addressing,
        text=spec.sources.get(spec.name, spec.source).splitlines(),
        chains=carry_chains.find(spec),
    )


_PROBES: Dict[str, str] = {}


def _probe(low: Lowerer, node, out: list) -> list:
    name = PROBE + re.sub(r"\W", "_", key(node[1], node[0]))
    _PROBES[name] = key(node[1], node[0])
    if not any(c[0] == name for c in low.controls):
        low.controls.append((name, 1, 0))
    return [("set", name, K(1, 1))] + out


@contextmanager
def probing():
    """Elaborate with every statement first setting its probe control to 1."""
    old, Lowerer.probe = Lowerer.probe, _probe
    try:
        yield
    finally:
        Lowerer.probe = old


def _flags(bound, layout: Dict[str, str]) -> Dict[str, int]:
    values = bound[1] if isinstance(bound, tuple) else bound
    return {k: int(bool(values.get(k))) for k, v in layout.items() if v == "flag"}


def cover_tune(path: str, spec_path: str, frames: int) -> dict:
    """The tune's format, its play entry's flag values and the frames that ran each
    statement of the spec at ``spec_path``, on the Python backend; a tune of another spec
    is ``skipped``."""
    # pylint: disable=import-outside-toplevel,protected-access
    from ..formats import find_format

    data = Path(path).read_bytes()
    fmt = find_format(data)
    model = fmt.load(data, None)
    alt = load(spec_path)
    binding = fmt.spec(model)
    if binding.spec.name != alt.name:
        return {"format": fmt.name, "skipped": binding.spec.name}
    seen: List[dict] = []

    def bind(view, entry):
        bound = binding.bind(view, entry)
        seen.append(_flags(bound, alt.layout))
        return bound

    use = dataclasses.replace(
        fmt, spec=lambda m: dataclasses.replace(binding, spec=alt, bind=bind)
    )
    hits: Counter = Counter()
    with probing():
        player = use.spec_player(model, native=False)
        for _ in range(frames):
            player.play_frame()
            for ident, slot in player._slot.items():
                name = ident.rpartition(":")[2]
                if player._state[slot] and name in _PROBES:
                    player._state[slot] = 0
                    hits[_PROBES[name]] += 1
    return {"format": fmt.name, "flags": seen[0] if seen else {}, "hits": dict(hits)}


def cover(spec_path: str, tunes: List[str], frames: int, jobs: Optional[int] = None):
    """``{tune: cover_tune result, or {"error": text}}`` over ``tunes``."""
    # pylint: disable=import-outside-toplevel
    from tqdm import tqdm

    from ..jobs import imap

    tasks = [(t, spec_path, frames) for t in tunes]
    out = {}
    for index, value, error in tqdm(
        imap(cover_tune, tasks, jobs), total=len(tasks), desc="tunes", unit="tune"
    ):
        out[tunes[index]] = value if error is None else {"error": repr(error)[:200]}
    return out


def report(analysis: Analysis, corpus: Optional[dict] = None, rare: int = 3) -> dict:
    """The measures as JSON-ready tables; with ``corpus`` (from :func:`cover`) per unit the
    statements no tune ran, those at most ``rare`` tunes reached and the frames run, and
    per flag the tunes it is true in."""
    out = {
        "spec": analysis.spec,
        "totals": analysis.totals(),
        "units": [dataclasses.asdict(u) for u in analysis.units.values()],
        "clones": analysis.clones,
        "layout": analysis.layout,
        "coupling": analysis.coupling,
        "addressing": analysis.addressing,
        "chains": analysis.chains,
    }
    if corpus is None:
        return out
    played = {t: r for t, r in corpus.items() if "hits" in r}
    reach: Dict[str, set] = defaultdict(set)
    frames: Counter = Counter()
    for tune, result in played.items():
        for stmt, count in result["hits"].items():
            reach[stmt].add(tune)
            frames[stmt] += count
    for unit in out["units"]:
        stmts = unit["statements"]
        unit["unreached"] = [s for s in stmts if not reach[s]]
        unit["rare"] = {s: sorted(reach[s]) for s in stmts if 0 < len(reach[s]) <= rare}
        unit["frames"] = sum(frames[s] for s in stmts)
    flags = sorted({f for r in played.values() for f in r["flags"]})
    out["corpus"] = {
        "tunes": len(played),
        "skipped": sum("skipped" in r for r in corpus.values()),
        "errors": {t: r["error"] for t, r in corpus.items() if "error" in r},
        "flags": {f: sum(r["flags"].get(f, 0) for r in played.values()) for f in flags},
        "combinations": len(
            {
                tuple(sorted(f for f, v in r["flags"].items() if v))
                for r in played.values()
            }
        ),
    }
    return out


def delta(old: dict, new: dict) -> dict:
    """``(old, new)`` of each total and unit measure that differs between two reports."""
    totals = {
        k: (old["totals"].get(k), v)
        for k, v in new["totals"].items()
        if old["totals"].get(k) != v
    }
    before = {u["name"]: u for u in old["units"]}
    after = {u["name"]: u for u in new["units"]}
    units = {}
    for name in sorted(set(before) | set(after)):
        a, b = before.get(name, {}), after.get(name, {})
        diff = {
            k: (a.get(k), b.get(k))
            for k in ("nodes", "decisions", "depth")
            if a.get(k) != b.get(k)
        }
        if diff:
            units[name] = diff
    return {"totals": totals, "units": units}


def render(analysis: Analysis, table: dict, top: int = 15) -> Iterator[str]:
    """``table`` (from :func:`report`) as text, ``top`` rows per section."""
    yield f"spec {table['spec']}: " + ", ".join(
        f"{k} {v}" for k, v in table["totals"].items()
    )
    corpus = table.get("corpus")
    if corpus:
        yield (
            f"corpus: {corpus['tunes']} tunes ({corpus['skipped']} of other specs, "
            f"{len(corpus['errors'])} errors), {corpus['combinations']} flag combinations"
        )
    yield ""
    yield "units by decisions"
    head = f"{'unit':32s} {'nodes':>6s} {'dec':>5s} {'depth':>5s}"
    yield head + (" stmts unrun  rare    frames" if corpus else "")
    for u in sorted(table["units"], key=lambda u: (-u["decisions"], u["name"]))[:top]:
        row = (
            f"{u['name'][:32]:32s} {u['nodes']:6d} {u['decisions']:5d} {u['depth']:5d}"
        )
        if corpus:
            row += (
                f" {len(u['statements']):5d} {len(u['unreached']):5d}"
                f" {len(u['rare']):5d} {u['frames']:9d}"
            )
        yield row
    yield ""
    yield "clones by nodes saved"
    for c in table["clones"][:top]:
        at = ", ".join(f"{u}:{line}" for u, line in c["at"][:4])
        more = f" +{c['count'] - 4}" if c["count"] > 4 else ""
        yield f"{c['savings']:5d} = {c['count']} x {c['size']:4d}  {at}{more}"
        yield f"        {analysis.excerpt(c['at'][0][1])}"
    yield ""
    yield "layout fields by uses in code"
    ranked = sorted(
        table["layout"].items(), key=lambda kv: (-kv[1]["guard"] - kv[1]["code"], kv[0])
    )
    flags = corpus["flags"] if corpus else {}
    for name, f in ranked[:top]:
        seen = f" true in {flags[name]}" if name in flags else ""
        yield (
            f"{name:24s} {f['kind']:5s} guard {f['guard']:3d} code {f['code']:3d} "
            f"decl {f['decl']:3d} units {len(f['units']):3d}{seen}"
        )
    if corpus:
        fixed = [n for n, k in flags.items() if k in (0, corpus["tunes"])]
        yield f"flags constant over the corpus: {', '.join(fixed) or 'none'}"
    yield ""
    yield "state written by several units"
    shared = [(n, c) for n, c in table["coupling"].items() if len(c["writers"]) > 1]
    for name, c in sorted(shared, key=lambda s: (-len(s[1]["writers"]), s[0]))[:top]:
        yield f"{name:16s} {', '.join(c['writers'])}"
    yield ""
    yield "addressing"
    for kind, n in Counter(a["kind"] for a in table["addressing"]).most_common():
        lines = [str(a["line"]) for a in table["addressing"] if a["kind"] == kind]
        yield f"{kind:20s} {n:3d}  lines {', '.join(lines[:12])}"
    yield ""
    yield "carry chains: escaping first, then by nodes saved"
    ranked = sorted(
        table.get("chains", ()),
        key=lambda c: (not c["escapes"], -c["saved"], c["lines"]),
    )
    for c in ranked[:top]:
        ends = ", ".join(c["ends"]) + (
            f" via {', '.join(c['escapes'])}" if c["escapes"] else ""
        )
        pairs = " ".join(f"{lo}/{hi}" for lo, hi in c["pairs"]) or "-"
        yield (
            f"{c['kind']:12s} {ends:24s} saves {c['saved']:3d}  "
            f"lines {','.join(map(str, c['lines'][:6]))}  pairs {pairs}"
        )
        for unit, proof in c["proofs"].items():
            if proof["status"].startswith(("proved", "refuted")):
                yield f"        {unit}: {proof['status']}"


__all__ = [
    "Analysis",
    "Unit",
    "analyse",
    "cover",
    "cover_tune",
    "delta",
    "key",
    "probing",
    "render",
    "report",
]
