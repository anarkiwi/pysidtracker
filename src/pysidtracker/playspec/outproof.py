"""Whole-call proof of an :func:`~.outorder.reorder` result, composed from local steps.

:func:`prove` checks each pass's structural side conditions and discharges every
obligation the forwarding passes record, each a small query over a fresh state at one
program point; docs/playspec.md, "Proving a reordered call", states why they compose.
"""

from __future__ import annotations

import os
import resource
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import z3
from tqdm import tqdm

from . import outorder as O
from . import smt
from ..jobs import imap
from .emit import declared
from .ir import Op, Program, Rd, V, width


class _Env(dict):
    """Variables of a fresh state: a free symbol for each name on first use."""

    def __init__(self, widths: Dict[str, int], symbols: dict):
        super().__init__()
        self.widths, self.symbols = widths, symbols

    def __missing__(self, name):
        value = self.symbols.get(name)
        if value is None:
            w = self.widths.get(name, 8)
            value = z3.Bool(f"v_{name}") if w == 1 else z3.BitVec(f"v_{name}", w)
            self.symbols[name] = value
        self[name] = value
        return value


def var_widths(*bodies) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for body in bodies:
        for s in O.statements(body):
            if s[0] == "set":
                out.setdefault(s[1], width(s[2]))
            if s[0] == "obj" and s[3]:
                out.setdefault(s[3], 1)
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
    unchecked: int = 0


THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


class _Evaluator(smt.Executor):
    """An executor that builds the term of a pure expression once per values of the
    memory and variables it reads, however many states and obligations ask."""

    def __init__(self):
        super().__init__()
        self.memo: dict = {}
        self.reads: dict = {}

    def info(self, e):
        """``(e, variable names, reads memory, pure)``: pure if no volatile read or
        ``old``, so its term needs no path condition."""
        found = self.reads.get(id(e))
        if found is None:
            nodes = list(O.walk(e))
            found = self.reads[id(e)] = (
                e,
                tuple(sorted({n.n for n in nodes if isinstance(n, V)})),
                any(isinstance(n, Rd) for n in nodes),
                not any(isinstance(n, Op) and n.op in ("vol", "old") for n in nodes),
            )
        return found

    def lazy(self, op, args, st):
        if not all(self.info(a)[3] for a in args):
            return super().lazy(op, args, st)
        first = self.expr(args[0], st)
        first = first if z3.is_bool(first) else first != 0
        rest = [self.expr(a, st) for a in args[1:]]
        if op == "ite":
            return z3.If(first, *rest)
        return (z3.And if op == "land" else z3.Or)(first, *rest)

    def expr(self, e, st):
        _e, names, memory, pure = self.info(e)
        if not pure:
            return super().expr(e, st)
        env = st.env
        key = (
            id(e),
            st.mem.get_id() if memory else 0,
            *(env[n].get_id() for n in names),
        )
        term = self.memo.get(key)
        if term is None:
            term = self.memo[key] = super().expr(e, st)
        return term


class _Checker:
    """Fresh states, and a fresh solver for each query: a solver reused across queries
    slows with every one it answers."""

    TACTIC = z3.Then("simplify", "propagate-values", "solve-eqs", "simplify", "smt")

    def __init__(self, widths, ranges, fixed, timeout_ms: int):
        self.widths, self.ranges, self.fixed = widths, ranges, fixed
        self.timeout_ms = timeout_ms
        self.executor = _Evaluator()
        self.symbols: dict = {}
        self.bounds: dict = {}
        self.memory = z3.Array("M", smt.BV16, smt.BV8)
        self.pinned = [z3.Select(self.memory, a) == v for a, v in fixed.items()]

    def state(self) -> smt.State:
        """A fresh state: free memory and variables."""
        return smt.State(self.memory, _Env(self.widths, self.symbols))

    def copy(self, st: smt.State) -> smt.State:
        """``st`` with its own variable map."""
        env = _Env(self.widths, self.symbols)
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
        out = list(self.pinned)
        for name, value in st.env.items():
            if name in self.ranges and not z3.is_bool(value):
                bound = self.bounds.get(name)
                if bound is None:
                    lo, hi = self.ranges[name]
                    bound = self.bounds[name] = [z3.UGE(value, lo), z3.ULE(value, hi)]
                out += bound
        return out

    def valid(self, premises: list, claims: list) -> Optional[bool]:
        """Whether ``premises`` imply ``claims`` (None: the solver gave up)."""
        if not claims:
            return True
        solver = self.TACTIC.solver()
        solver.set("timeout", self.timeout_ms)
        solver.add(*premises, z3.Not(z3.And(*claims)))
        answer = solver.check()
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
            sets |= {t[3] for t in (old, new) if t[0] == "obj" and t[3]}
            post = _changed(post, pre, sets, bool({"wr", "obj"} & {old[0], new[0]}))
            used = _names(*O.stmt_exprs(old), *O.stmt_exprs(new)) | sets
            for name in used:
                _ = st.env[name]
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
    """The facts on ``keys``, the known conditions, and the facts whose names ``used``
    reaches, closed under the names facts use."""
    out: dict = {key: facts[key] for key in keys if key in facts}
    out.update((key, fact) for key, fact in facts.items() if not isinstance(key, str))
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


def _chunk(items, base, widths, ranges, fixed, timeout_ms, until):
    """``(index, answer)`` of each obligation of ``items`` (numbered from ``base``) not
    proved, then how many ``until`` left unchecked."""
    checker = _Checker(widths, ranges, fixed, timeout_ms)
    out = []
    for at, item in enumerate(items):
        if until is not None and time.monotonic() > until:
            return out, len(items) - at
        answer = checker.obligation(item)
        if answer is not True:
            out.append((base + at, answer))
    return out, 0


def _blocks(count: int, jobs: int):
    """``[lo, hi)`` blocks of ``count`` obligations: one if ``jobs`` is 1, else several
    per job so workers that finish early take another."""
    size = count if jobs == 1 else max(1, -(-count // (jobs * 4)))
    return [(lo, min(lo + size, count)) for lo in range(0, count, size)]


def discharge(
    proof: list,
    widths: Dict[str, int],
    ranges: Dict[str, tuple],
    fixed: Dict[int, int],
    *,
    timeout_ms: int,
    what: str = "obligations",
    progress: bool = False,
    until: Optional[float] = None,
    jobs: int = 1,
) -> Discharged:
    """Check every obligation in ``proof`` under the shared premises over ``jobs``
    processes, those left when the monotonic clock passes ``until`` counted unchecked.
    """
    out, found = Discharged(len(proof)), []
    blocks = _blocks(len(proof), jobs)
    tasks = [
        (proof[lo:hi], lo, widths, ranges, fixed, timeout_ms, until)
        for lo, hi in blocks
    ]
    if jobs > 1:
        for var in THREAD_VARS:
            os.environ.setdefault(var, "2")
    with tqdm(total=len(proof), desc=what, unit="ob", disable=not progress) as bar:
        results = (
            ((i, *_chunk(*t)) for i, t in enumerate(tasks))
            if jobs == 1
            else (
                (i, *value) if error is None else _raise(error)
                for i, value, error in imap(_chunk, tasks, jobs)
            )
        )
        for index, answers, unchecked in results:
            bar.update(blocks[index][1] - blocks[index][0])
            out.unchecked += unchecked
            found += answers
            if any(answer is False for _i, answer in answers):
                break
    for at, answer in sorted(found):
        if answer is None:
            out.unknown.append(f"{what} {at}: {proof[at][0]}")
        elif out.failed is None:
            out.failed = f"{what} {at}: {proof[at][0]} {proof[at][2:4]}"
    return out


def _raise(error):
    raise error


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
            out += (
                [f"{name}: {d.unchecked} obligations unchecked"] if d.unchecked else []
            )
            out += d.unknown
        return out


def prove(
    program: Program,
    order,
    *,
    memory: Optional[bytes] = None,
    observed=(),
    timeout_ms: int = 10_000,
    budget_s: Optional[float] = None,
    progress: bool = False,
    jobs: int = 1,
) -> Proof:
    """Reorder ``program`` and prove the whole call of the result equal to the original
    on the kept writes, memory outside the SID and promoted cells, and controls, within
    ``budget_s`` seconds if set (obligations left then stay open), over ``jobs``
    processes."""
    start = time.monotonic()
    until = None if budget_s is None else start + budget_s
    stages: dict = {}
    result = O.reorder(program, order, memory=memory, observed=observed, proof=stages)
    end = frozenset(n for n, _w, _d in program.controls)
    final = declared(
        O.rewrite(
            O.eliminate(stages["forwarded"], end, end)[0],
            O.tidy,
            lambda s: [] if s[0] == "hook" else None,
        ),
        [*end, *program.inputs],
    )
    live = O.live_slots(result.slots, result.native)
    checks = {
        "flush order": None if result.paths == {live} else "out of order",
        "flushed after the last store": flushed_last(stages["placed"], live),
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
            until=until,
            jobs=jobs,
        )
        for name, ranges, fixed in (
            ("fix", {}, result.fixed),
            ("forward", result.ranges, {}),
        )
    }
    holds = all(v is None for v in checks.values()) and not any(
        d.failed or d.unknown or d.unchecked for d in passes.values()
    )
    peak = max(
        resource.getrusage(who).ru_maxrss // 1024
        for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
    )
    return Proof(holds, checks, passes, time.monotonic() - start, peak)


__all__ = ["Discharged", "Proof", "discharge", "flushed_last", "prove", "var_widths"]
