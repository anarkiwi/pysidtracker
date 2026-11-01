"""Whole-call proof of an :func:`~.outorder.reorder` result, composed from local steps.

:func:`prove` checks each pass's structural side conditions and discharges every
obligation the forwarding passes record, each a small query over a fresh state at one
program point; docs/playspec.md, "Proving a reordered call", states why they compose.
"""

from __future__ import annotations

import resource
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import z3
from tqdm import tqdm

from . import outorder as O
from . import smt
from .emit import declared, normal
from .ir import Program, V, width


class _Env(dict):
    """Variables of a fresh state: a free symbol for each name on first use."""

    def __init__(self, widths: Dict[str, int]):
        super().__init__()
        self.widths = widths

    def __missing__(self, name):
        w = self.widths.get(name, 8)
        value = z3.Bool(f"v_{name}") if w == 1 else z3.BitVec(f"v_{name}", w)
        self[name] = value
        return value


def var_widths(*bodies) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for body in bodies:
        for s in O.statements(body):
            if s[0] == "set":
                out.setdefault(s[1], width(s[2]))
            for e in O.stmt_exprs(s):
                for n in O.walk(e):
                    if isinstance(n, V):
                        out[n.n] = n.w
    return out


@dataclass
class Discharged:
    """The local obligations of one pass: how many, the first refuted, the unknown."""

    total: int = 0
    failed: Optional[str] = None
    unknown: List[str] = field(default_factory=list)


class _Checker:
    """One solver over fresh states, with the premises every point shares."""

    def __init__(self, widths, ranges, fixed, timeout_ms: int):
        self.widths, self.ranges, self.fixed = widths, ranges, fixed
        self.solver = z3.Solver()
        self.solver.set("timeout", timeout_ms)
        self.executor = smt.Executor()

    def state(self) -> smt.State:
        """A fresh state: free memory and variables."""
        return smt.State(z3.Array("M", smt.BV16, smt.BV8), _Env(self.widths))

    def copy(self, st: smt.State) -> smt.State:
        """``st`` with its own variable map."""
        env = _Env(self.widths)
        env.update(st.env)
        return smt.State(st.mem, env)

    def facts(self, env: dict, st: smt.State) -> list:
        """The facts ``env`` claims of ``st``."""
        ev = self.executor.expr
        return [
            (st.env[key] if isinstance(key, str) else ev(key, st)) == ev(fact.expr, st)
            for key, fact in env.items()
        ]

    def premises(self, st: smt.State) -> list:
        """The fixed bytes, and the value ranges of the variables ``st`` has used."""
        out = [z3.Select(st.mem, a) == v for a, v in self.fixed.items()]
        for name, value in list(st.env.items()):
            if name in self.ranges and not z3.is_bool(value):
                lo, hi = self.ranges[name]
                out += [z3.UGE(value, lo), z3.ULE(value, hi)]
        return out

    def valid(self, premises: list, claims: list) -> Optional[bool]:
        """Whether ``premises`` imply ``claims`` (None: the solver gave up)."""
        if not claims:
            return True
        answer = self.solver.check(*premises, z3.Not(z3.And(*claims)))
        return None if answer == z3.unknown else answer == z3.unsat

    def obligation(self, item) -> Optional[bool]:
        """Discharge one recorded obligation: the claimed facts not carried unchanged
        (the frame rule), from the facts their names depend on."""
        kind, st, ev = item[0], self.state(), self.executor.expr
        extra: list = []
        if kind == "cond":
            pre, post, used = item[1], {}, _names(item[2], item[3])
            claims = [ev(item[2], st) == ev(item[3], st)]
        elif kind in ("entry", "join"):
            pre, post = item[1], _changed(item[4], item[1], frozenset(), False)
            used = _names(item[2]) if item[2] is not None else set()
            if item[2] is not None:
                extra = [ev(item[2], st) == ev(item[3], st)]
            claims = self.facts(post, st)
        else:
            _k, pre, old, new, post = item
            sets = {t[1] for t in (old, new) if t[0] == "set"}
            post = _changed(post, pre, sets, "wr" in (old[0], new[0]))
            used = _names(*O.stmt_exprs(old), *O.stmt_exprs(new)) | sets
            a, b = self.copy(st), self.copy(st)
            self.executor.run([old], a)
            self.executor.run([new], b)
            claims = [a.mem == b.mem]
            claims += [
                a.env[n] == b.env[n]
                for n in set(a.env) | set(b.env)
                if not a.env[n].eq(b.env[n])
            ]
            claims += self.facts(post, a)
            if (
                old[0] == "set"
                and old[1] in self.ranges
                and not z3.is_bool(a.env[old[1]])
            ):
                lo, hi = self.ranges[old[1]]
                claims += [z3.UGE(a.env[old[1]], lo), z3.ULE(a.env[old[1]], hi)]
        if not claims:
            return True
        for key, fact in post.items():
            used |= fact.uses | ({key} if isinstance(key, str) else _names(key))
        facts = self.facts(_relevant(pre, used, post), st) + extra
        return self.valid(facts + self.premises(st), claims)


def _names(*exprs) -> set:
    return set().union(*(O.names(e) for e in exprs)) if exprs else set()


def _changed(post: dict, pre: dict, sets, stores: bool) -> dict:
    """The facts of ``post`` not carried unchanged from ``pre`` past a statement that
    assigns ``sets`` and stores to memory if ``stores``."""
    return {
        key: fact
        for key, fact in post.items()
        if pre.get(key) is not fact
        or (key in sets if isinstance(key, str) else False)
        or fact.uses & sets
        or (stores and fact.spans)
    }


def _relevant(facts: dict, used: set, keys) -> dict:
    """The facts on ``keys`` and those whose names ``used`` reaches, closed under the
    names facts use."""
    out: dict = {key: facts[key] for key in keys if key in facts}
    for fact in out.values():
        used |= fact.uses
    grown = True
    while grown:
        grown = False
        for key, fact in facts.items():
            names = {key} if isinstance(key, str) else _names(key)
            if key not in out and names & used:
                out[key] = fact
                used |= fact.uses | names
                grown = True
    return out


def discharge(
    proof: list,
    widths: Dict[str, int],
    ranges: Dict[str, tuple],
    fixed: Dict[int, int],
    *,
    timeout_ms: int,
    what: str = "obligations",
    progress: bool = False,
) -> Discharged:
    """Check every obligation in ``proof`` under the shared premises."""
    checker = _Checker(widths, ranges, fixed, timeout_ms)
    out = Discharged(len(proof))
    for index, item in enumerate(
        tqdm(proof, desc=what, unit="ob", disable=not progress)
    ):
        answer = checker.obligation(item)
        if answer is None:
            out.unknown.append(f"{what} {index}: {item[0]}")
        elif not answer:
            out.failed = f"{what} {index}: {item[0]} {item[2:4]}"
            return out
    return out


def flushed_last(placed: list, slots) -> Optional[str]:
    """None if no statement after a slot's flush, on any path, assigns its output or
    flag; else the slot's flag."""
    effects = O.Effects()
    owned = {O.flag_var(s).n: {O.out_var(s).n, O.flag_var(s).n} for s in slots}

    def run(stmts, after: frozenset) -> Optional[str]:
        later = [after]
        for s in reversed(stmts):
            later.append(
                frozenset() if s[0] == "finish" else later[-1] | effects(s).sets
            )
        later.reverse()
        for i, s in enumerate(stmts):
            if s[0] == "if" and isinstance(s[1], V) and s[1].n in owned and not s[3]:
                if owned[s[1].n] & later[i + 1]:
                    return s[1].n
                continue
            rest = later[i + 1] | (effects(s).sets if s[0] == "loop" else frozenset())
            for arm in O.branches(s):
                found = run(arm, rest)
                if found:
                    return found
        return None

    return run(placed, frozenset())


@dataclass
class Proof:
    """A whole-call proof: the structural checks, the local passes, time and memory."""

    holds: bool
    checks: Dict[str, Optional[str]]
    passes: Dict[str, Discharged]
    seconds: float
    peak_mb: int

    @property
    def open(self) -> List[str]:
        """What stays unproved."""
        out = [f"{k}: {v}" for k, v in self.checks.items() if v is not None]
        for name, d in self.passes.items():
            out += [f"{name}: {d.failed}"] if d.failed else []
            out += d.unknown
        return out


def prove(
    program: Program,
    order,
    *,
    memory: Optional[bytes] = None,
    observed=(),
    timeout_ms: int = 10_000,
    progress: bool = False,
) -> Proof:
    """Reorder ``program`` and prove the whole call of the result equal to the original
    on the kept writes, memory outside the SID and promoted cells, and controls."""
    start = time.monotonic()
    stages: dict = {}
    result = O.reorder(program, order, memory=memory, observed=observed, proof=stages)
    end = frozenset(n for n, _w, _d in program.controls)
    final = declared(
        O.rewrite(
            O.eliminate(stages["forwarded"], end, end)[0],
            normal,
            lambda s: [] if s[0] == "hook" else None,
        ),
        [*end, *program.inputs],
    )
    checks = {
        "flush order": None if result.paths == {result.slots} else "out of order",
        "flushed after the last store": flushed_last(stages["placed"], result.slots),
        "result": None if final == result.program.body else "differs from the passes",
    }
    sizes = var_widths(stages["nested"], stages["placed"], stages["forwarded"])
    passes = {
        name: discharge(
            stages[name],
            sizes,
            ranges,
            fixed,
            timeout_ms=timeout_ms,
            what=name,
            progress=progress,
        )
        for name, ranges, fixed in (
            ("fix", {}, result.fixed),
            ("forward", result.ranges, {}),
        )
    }
    holds = all(v is None for v in checks.values()) and not any(
        d.failed or d.unknown for d in passes.values()
    )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    return Proof(holds, checks, passes, time.monotonic() - start, peak)


__all__ = ["Discharged", "Proof", "discharge", "flushed_last", "prove", "var_widths"]
