"""The playspec IR as a z3 transition relation, and bounded property checking.

Memory is a z3 array from 16-bit addresses to bytes; statements execute guarded
(each assignment and store is an if-then-else on its path condition), so one
frame is one term per memory byte and variable, with no path enumeration.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import z3
from tqdm import tqdm

from ..errors import SidError
from .ir import K, Program, Rd, V

BV16 = z3.BitVecSort(16)
BV8 = z3.BitVecSort(8)


def volatile_inputs():
    """The stream of values the volatile reads of a call return, in order."""
    return z3.Array("volatile_input", BV16, BV8)


def volatile_log():
    """``(count, addresses)`` of volatile reads made so far: none, over all addresses."""
    return z3.BitVecVal(0, 16), z3.K(BV16, z3.BitVecVal(0, 16))


def symbolic_memory(name: str = "mem0"):
    """A fully unconstrained 64 KiB memory."""
    return z3.Array(name, BV16, BV8)


def concrete_memory(data: bytes, symbolic: Iterable[int] = (), prefix: str = "m"):
    """``data`` (64 KiB) as a z3 array, the addresses in ``symbolic`` left free."""
    common = Counter(data).most_common(1)[0][0]
    mem = z3.K(BV16, z3.BitVecVal(common, 8))
    for addr, byte in enumerate(data):
        if byte != common:
            mem = z3.Store(mem, addr, byte)
    for addr in symbolic:
        mem = z3.Store(mem, addr, z3.BitVec(f"{prefix}_{addr:04x}", 8))
    return mem


def select(mem, addr, memo: Optional[dict] = None):
    """``Select(mem, addr)`` with every store and if-then-else of ``mem`` rewritten away.

    The result reads only base arrays, so the solver needs no array reasoning;
    ``memo`` shares rewritten reads across calls.
    """
    memo = {} if memo is None else memo
    stack = [(mem, addr)]
    while stack:
        arr, idx = stack[-1]
        key = (arr.get_id(), idx.get_id())
        if key in memo:
            stack.pop()
            continue
        if z3.is_store(arr):
            base, at, val = arr.arg(0), arr.arg(1), arr.arg(2)
            if z3.is_bv_value(at) and z3.is_bv_value(idx):
                inner = (base.get_id(), idx.get_id())
                if inner not in memo:
                    stack.append((base, idx))
                    continue
                memo[key] = val if at.as_long() == idx.as_long() else memo[inner]
            else:
                inner = (base.get_id(), idx.get_id())
                if inner not in memo:
                    stack.append((base, idx))
                    continue
                memo[key] = z3.If(at == idx, val, memo[inner])
        elif z3.is_app_of(arr, z3.Z3_OP_ITE):
            a, b = arr.arg(1), arr.arg(2)
            pending = [
                (x, idx) for x in (a, b) if (x.get_id(), idx.get_id()) not in memo
            ]
            if pending:
                stack.extend(pending)
                continue
            memo[key] = z3.If(
                arr.arg(0),
                memo[(a.get_id(), idx.get_id())],
                memo[(b.get_id(), idx.get_id())],
            )
        elif z3.is_const_array(arr):
            memo[key] = arr.arg(0)
        else:
            memo[key] = z3.Select(arr, idx)
        stack.pop()
    return memo[(mem.get_id(), addr.get_id())]


@dataclass
class State:
    """Memory, variables and the proof obligations a run accumulates."""

    mem: object
    env: Dict[str, object] = field(default_factory=dict)
    obligations: List[Tuple[object, str]] = field(default_factory=list)
    old: object = None
    writes: List[object] = field(default_factory=list)
    done: object = None
    reads: object = field(default_factory=lambda: volatile_log()[0])
    read_log: object = field(default_factory=lambda: volatile_log()[1])
    read_sites: int = 0


def _bool(term):
    return term if z3.is_bool(term) else term != 0


class Executor:
    """Guarded symbolic execution of lowered statements."""

    def __init__(self, procs: Optional[Dict[str, list]] = None, quiet: bool = False):
        self.procs = procs or {}
        self.quiet = quiet
        self.guard = None

    def volatile(self, addr, st: State):
        """A volatile read: the next value of the input stream, logged with its
        address under the path condition it executes under."""
        if self.quiet:
            return z3.Select(st.mem, addr)
        value = z3.Select(volatile_inputs(), st.reads)
        logged = z3.Store(st.read_log, st.reads, addr)
        counted = st.reads + 1
        if self.guard is not None:
            logged = z3.If(self.guard, logged, st.read_log)
            counted = z3.If(self.guard, counted, st.reads)
        st.read_log, st.reads = logged, counted
        st.read_sites += 1
        return value

    def lazy(self, op, args, st: State):
        """``land``/``lor``/``ite`` evaluate an operand only on the paths that reach it."""
        first = self.expr(args[0], st)
        saved = self.guard
        test = _bool(first)
        arms = {
            "land": [(test, 1)],
            "lor": [(z3.Not(test), 1)],
            "ite": [(test, 1), (z3.Not(test), 2)],
        }[op]
        rest = {}
        for cond, index in arms:
            self.guard = _and(saved, cond)
            rest[index] = self.expr(args[index], st)
        self.guard = saved
        if op == "ite":
            return z3.If(test, rest[1], rest[2])
        return (z3.And if op == "land" else z3.Or)(test, rest[1])

    def expr(self, e, st: State):
        """The z3 term of ``e`` in ``st``."""
        # pylint: disable=too-many-return-statements,too-many-branches
        if isinstance(e, K):
            if e.v is None or e.w == 0:
                raise ValueError("cannot execute an unbound or untyped constant")
            return z3.BoolVal(bool(e.v)) if e.w == 1 else z3.BitVecVal(e.v, e.w)
        if isinstance(e, Rd):
            return z3.Select(st.mem, self.expr(e.a, st))
        if isinstance(e, V):
            return st.env[e.n]
        op, args = e.op, e.a
        if op == "note":
            raise SidError("standard note mode is not supported by the solver")
        if op == "vol":
            return self.volatile(self.expr(args[0], st), st)
        if op in ("land", "lor", "ite"):
            return self.lazy(op, args, st)
        if op == "old":
            inner = State(st.old if st.old is not None else st.mem, st.env)
            return self.expr(args[0], inner)
        vals = [self.expr(a, st) for a in args]
        if op == "lnot":
            return z3.Not(vals[0])
        if op == "zext":
            src = vals[0]
            if z3.is_bool(src):
                return z3.If(src, z3.BitVecVal(1, e.w), z3.BitVecVal(0, e.w))
            return z3.ZeroExt(e.w - src.size(), src)
        if op == "sext":
            return z3.SignExt(e.w - vals[0].size(), vals[0])
        if op == "lo":
            return z3.Extract(7, 0, vals[0])
        if op == "hi":
            return z3.Extract(15, 8, vals[0])
        if op == "word":
            return z3.Concat(vals[1], vals[0])
        if op == "not":
            return ~vals[0]
        if op == "neg":
            return -vals[0]
        a, b = vals
        return _BINARY[op](a, b)

    def run(self, stmts, st: State, guard=None) -> State:
        """Execute ``stmts`` under path condition ``guard`` (None: always)."""
        for s in stmts:
            tag = s[0]
            if st.done is not None:  # a ``finish`` ended the call on some paths
                guard = _and(guard, z3.Not(st.done))
            self.guard = guard
            if tag == "finish":
                now = z3.BoolVal(True) if guard is None else guard
                st.done = now if st.done is None else z3.Or(st.done, now)
                continue
            if tag == "wr":
                addr, val = self.expr(s[1], st), self.expr(s[2], st)
                if guard is not None:
                    val = z3.If(guard, val, z3.Select(st.mem, addr))
                st.mem = z3.Store(st.mem, addr, val)
                st.writes.append(addr)
            elif tag == "set":
                val = self.expr(s[2], st)
                if guard is not None and s[1] in st.env:
                    val = z3.If(guard, val, st.env[s[1]])
                st.env[s[1]] = val
            elif tag == "if":
                self.branch(s, st, guard)
            elif tag == "loop":
                self.loop(s, st, guard)
            elif tag == "call":
                self.run(self.procs[s[1]], st, guard)
            elif tag == "fire":
                self.run(s[2], st, guard)
            elif tag == "obj":
                self.react(s, st, guard)
            elif tag == "native":
                raise SidError("native 6502 code is opaque to the solver")
            elif tag == "assert":
                cond = self.expr(s[1], st)
                st.obligations.append((_and(guard, z3.Not(cond)), s[2]))
        return st

    def react(self, s, st: State, guard) -> None:
        """An object reaction: every byte it writes and its wrap flag are uninterpreted
        functions of its operands and the bytes it may read, the same for every call."""
        _tag, name, event, out, ops, reads, writes = s
        if (0, 0xFFFF) in reads:
            raise SidError(f"object {name} reads an unbounded table")
        args = [self.expr(o, st) for o in ops]
        for lo, hi in sorted(reads):
            args += [z3.Select(st.mem, z3.BitVecVal(a, 16)) for a in range(lo, hi + 1)]
        sorts = [a.sort() for a in args]
        key = f"{name}:{event}"
        for addr in writes:
            value = z3.Function(f"{key}:{addr}", *sorts, BV8)(*args)
            at = z3.BitVecVal(addr, 16)
            if guard is not None:
                value = z3.If(guard, value, z3.Select(st.mem, at))
            st.mem = z3.Store(st.mem, at, value)
            st.writes.append(at)
        if out:
            flag = z3.Function(f"{key}:wrap", *sorts, z3.BoolSort())(*args)
            if guard is not None and out in st.env:
                flag = z3.If(guard, flag, st.env[out])
            st.env[out] = flag

    def branch(self, s, st: State, guard) -> None:
        """An ``if``: a static condition picks one arm, else both run guarded."""
        cond = z3.simplify(self.expr(s[1], st))
        if z3.is_true(cond):
            self.run(s[2], st, guard)
        elif z3.is_false(cond):
            self.run(s[3], st, guard)
        else:
            self.run(s[2], st, _and(guard, cond))
            self.run(s[3], st, _and(guard, z3.Not(cond)))

    def loop(self, s, st: State, guard) -> None:
        """A bounded loop, unrolled; running past the bound is an obligation."""
        _tag, cond, body, bound, line = s
        for _ in range(bound):
            self.guard = guard
            live = z3.simplify(_and(guard, self.expr(cond, st)))
            if z3.is_false(live):
                return
            self.run(body, st, live)
            guard = live
        self.quiet, quiet = True, self.quiet
        live = _and(guard, self.expr(cond, st))
        self.quiet = quiet
        st.obligations.append(
            (live, f"loop at line {line} runs past its bound {bound}")
        )


def _and(guard, cond):
    return cond if guard is None else z3.And(guard, cond)


_BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "and": lambda a, b: a & b,
    "or": lambda a, b: a | b,
    "xor": lambda a, b: a ^ b,
    "shl": lambda a, b: a << b,
    "shr": z3.LShR,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "ult": z3.ULT,
    "ule": z3.ULE,
    "slt": lambda a, b: a < b,
    "sle": lambda a, b: a <= b,
    "own": lambda a, b: a == b,
}


def initial_env(program: Program, prefix: str = "") -> Dict[str, object]:
    """Free z3 variables for the hidden control states and entry registers."""
    env = {n: z3.BitVec(f"{prefix}{n}", w) for n, w, _d in program.controls}
    env.update(
        (n, z3.Bool(n) if w == 1 else z3.BitVec(n, w))
        for n, w in program.inputs.items()
    )
    return env


def step(program: Program, mem, env: Optional[dict] = None) -> State:
    """One frame of ``program`` from memory ``mem`` (and hidden controls ``env``)."""
    st = State(mem, {**initial_env(program), **(env or {})}, old=mem)
    return Executor(program.procs).run(program.body, st)


def transition(program: Program):
    """``(pre, post, relation)``: memories and z3 formula ``post == frame(pre)``."""
    pre = symbolic_memory("pre")
    post = symbolic_memory("post")
    env0 = initial_env(program, "pre_")
    st = step(program, pre, env0)
    eqs = [post == st.mem]
    eqs += [z3.BitVec(f"post_{n}", w) == st.env[n] for n, w, _d in program.controls]
    return pre, post, z3.And(eqs)


def check_property(expr, mem, old=None, env=None):
    """The z3 truth of a lowered ``expr`` on ``mem`` (``old``: frame-start memory)."""
    st = State(mem, dict(env or {}), old=old)
    return Executor(quiet=True).expr(expr, st)


@dataclass
class BmcResult:
    """A bounded check: ``holds``, else the failing ``frame``, ``what`` and ``model``."""

    holds: bool
    frames: int
    frame: Optional[int] = None
    what: Optional[str] = None
    model: Dict[str, object] = field(default_factory=dict)


def bmc(
    program: Program,
    frames: int,
    mem=None,
    *,
    timeout_ms: int = 0,
    progress: bool = False,
) -> BmcResult:
    """Check invariants, asserts and loop bounds of ``frames`` calls from ``mem``.

    ``mem`` defaults to fully symbolic memory constrained by the spec's assumes.
    """
    mem0 = mem if mem is not None else symbolic_memory()
    solver = z3.Solver()
    if timeout_ms:
        solver.set("timeout", timeout_ms)
    env = initial_env(program, "init_")
    solver.add(*[_bool(check_property(a, mem0, env=env)) for a in program.assumes])
    current = mem0
    for frame in tqdm(range(frames), desc="bmc", unit="frame", disable=not progress):
        st = step(program, current, env)
        goals = [(cond, what) for cond, what in st.obligations]
        goals += [
            (
                z3.Not(check_property(inv, st.mem, current, st.env)),
                f"invariant {name}",
            )
            for name, inv in program.invariants
        ]
        for cond, what in goals:
            if solver.check(cond) == z3.sat:
                return BmcResult(
                    False, frames, frame, what, _model(program, solver.model(), mem0)
                )
        current, env = st.mem, st.env
    return BmcResult(True, frames)


def _model(program: Program, model, mem0) -> Dict[str, object]:
    """The initial bytes of every cell a counterexample sets, by cell name."""
    out = {}
    for cell in program.cells:
        values = [
            model.eval(z3.Select(mem0, a), model_completion=True).as_long()
            for a in cell.extent()[:16]
        ]
        if values:
            key = f"{cell.owner}.{cell.name}" if cell.owner else cell.name
            out[key] = values if len(values) > 1 else values[0]
    return out


__all__ = [
    "BmcResult",
    "Executor",
    "State",
    "bmc",
    "check_property",
    "concrete_memory",
    "initial_env",
    "step",
    "symbolic_memory",
    "select",
    "transition",
]
