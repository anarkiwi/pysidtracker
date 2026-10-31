"""Reschedule a lowered playspec so its SID register writes come out in a canonical order.

:func:`reorder` inlines, promotes frame-local cells, instruments SID stores as output
variables, places each slot's flush, forwards values and removes dead code; :func:`check`
proves the result with the solver. See docs/playspec.md, "Output order".
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import z3
from tqdm import tqdm

from ..errors import SidError
from . import smt
from .ir import K, Op, Program, Rd, V, width
from .lower import mk

SID = 0xD400
REGS = 25
VOICE_REG = {
    "freq_lo": 0,
    "freq_hi": 1,
    "pw_lo": 2,
    "pw_hi": 3,
    "ctrl": 4,
    "ad": 5,
    "sr": 6,
}
ORDER = (
    "fc_lo,fc_hi,res_filt,mode_vol,voice(ad,sr,ctrl:2,ctrl,freq_lo,freq_hi,pw_lo,pw_hi)"
)
SONG_REG = {"fc_lo": 21, "fc_hi": 22, "res_filt": 23, "mode_vol": 24}
FULL = (0, 0xFFFF)
TRUE, FALSE = K(1, 1), K(0, 1)
Slot = Tuple[int, int]


class OrderError(SidError):
    """An order that is malformed, or that a program cannot be rescheduled to."""


def _split(text: str) -> List[str]:
    out, depth, cur = [], 0, ""
    for ch in text.replace(" ", ""):
        depth += (ch == "(") - (ch == ")")
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return out + [cur] if cur else out


def _token(token: str, voice: Optional[int]) -> Slot:
    name, _, k = token.partition(":")
    table = SONG_REG if voice is None else VOICE_REG
    if name not in table or k not in ("", "1", "2"):
        raise OrderError(f"no register {token!r}")
    return (table[name] if voice is None else 7 * voice + table[name], int(k or 1))


def parse_order(text: str) -> Tuple[Slot, ...]:
    """The ``(register, k)`` slots of an order: song register names, ``voice(...)``
    (each voice's registers in turn) and ``interleave(...)`` (register by register);
    ``name`` is a register's last write in a call, ``name:2`` the one before."""
    slots: List[Slot] = []
    for item in _split(text):
        group = re.fullmatch(r"(voice|interleave)\((.*)\)", item)
        if group is None:
            slots.append(_token(item, None))
            continue
        tokens = group.group(2).split(",")
        pairs = [(v, t) for v in range(3) for t in tokens]
        if group.group(1) == "interleave":
            pairs = [(v, t) for t in tokens for v in range(3)]
        slots += [_token(t, v) for v, t in pairs]
    at = {s: i for i, s in enumerate(slots)}
    if len(at) != len(slots) or {r for r, k in slots if k == 1} != set(range(REGS)):
        raise OrderError(f"{text!r} must keep every register's last write once")
    if any(at[(r, 1)] < i for (r, k), i in at.items() if k == 2):
        raise OrderError(f"{text!r} puts a register's last write first")
    return tuple(slots)


def kept(writes: Sequence[Tuple[int, int]], slots: Sequence[Slot]) -> List[Tuple]:
    """The ``(register, value)`` writes of one call that ``slots`` keep, in their order."""
    regs = np.fromiter((r for r, _ in writes), np.int64, len(writes))
    out = []
    for reg, k in slots:
        at = np.flatnonzero(regs == reg)
        if len(at) >= k:
            out.append(tuple(writes[at[-k]]))
    return out


# ---- expressions ---------------------------------------------------------------------


def _subs(e) -> tuple:
    if isinstance(e, Rd):
        return (e.a,)
    return e.a if isinstance(e, Op) else ()


def walk(e):
    """Every node of expression ``e``."""
    stack = [e]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(_subs(node))


def names(e) -> FrozenSet[str]:
    """The variables ``e`` reads."""
    return frozenset(n.n for n in walk(e) if isinstance(n, V))


def volatile(e) -> bool:
    """Whether ``e`` reads a volatile input (never moved, merged or dropped)."""
    return any(isinstance(n, Op) and n.op == "vol" for n in walk(e))


RANGES: ContextVar[Dict[str, Tuple[int, int]]] = ContextVar("RANGES", default={})


def span(e) -> Tuple[int, int]:
    """An interval holding every value of ``e`` (a variable's from :data:`RANGES`)."""
    if isinstance(e, K):
        return FULL if e.v is None else (e.v, e.v)
    top = (1 << max(width(e), 1)) - 1
    if isinstance(e, V):
        return RANGES.get().get(e.n, (0, top))
    if isinstance(e, Op) and e.op == "and":
        return (0, min(span(a)[1] for a in e.a))
    if isinstance(e, Rd) or e.op not in ("add", "word", "zext"):
        return (0, top)
    parts = [span(a) for a in e.a]
    scale = (1, 256) if e.op == "word" else (1, 1)
    lo = sum(p[0] * s for p, s in zip(parts, scale))
    hi = sum(p[1] * s for p, s in zip(parts, scale))
    return (lo, hi) if hi <= top else (0, top)


def ranges(body, unknown: Iterable[str]) -> Dict[str, Tuple[int, int]]:
    """An interval holding every value each variable ``body`` assigns takes (``unknown``:
    variables that also hold values from outside, left out)."""
    unknown = frozenset(unknown)
    assigns = [
        (s[1], s[2]) for s in statements(body) if s[0] == "set" and s[1] not in unknown
    ]
    table: Dict[str, Tuple[int, int]] = {}
    for _ in range(8):
        token = RANGES.set(table)
        new: Dict[str, Tuple[int, int]] = {}
        for name, value in assigns:
            lo, hi = span(value)
            old = new.get(name, (lo, hi))
            new[name] = (min(lo, old[0]), max(hi, old[1]))
        RANGES.reset(token)
        if new == table:
            break
        table = new
    return table


def reads(e) -> List[Tuple[int, int]]:
    """The address intervals of the memory reads in ``e``."""
    return [span(n.a) for n in walk(e) if isinstance(n, Rd)]


def rebuild(e, leaf):
    """``e`` with each node ``leaf`` maps (None: kept) replaced, then folded."""
    if isinstance(e, K):
        return e
    got = leaf(e)
    if got is not None:
        return got
    if isinstance(e, V):
        return e
    if isinstance(e, Rd):
        return Rd(rebuild(e.a, leaf))
    args = tuple(rebuild(a, leaf) for a in e.a)
    return fold(e.op, e.w, args) if args != e.a else e


def fold(op: str, w: int, args: tuple):
    """``Op(op, w, args)`` with constants and trivial branches folded."""
    if op == "ite":
        cond, a, b = args
        if isinstance(cond, K) and cond.v is not None:
            return a if cond.v else b
        if a == b:
            return a
        if (a, b) == (TRUE, FALSE):
            return cond
        if (a, b) == (FALSE, TRUE):
            return fold("lnot", 1, (cond,))
        if isinstance(b, Op) and b.op == "ite" and b.a[0] == cond:
            return fold("ite", w, (cond, a, b.a[2]))
    if op in ("land", "lor") and args[0] == args[1]:
        return args[0]
    return mk(op, w, *args)


def _stmt_exprs(s) -> tuple:
    tag = s[0]
    if tag == "wr":
        return (s[1], s[2])
    if tag in ("set", "hook"):
        return (s[2],)
    if tag in ("if", "loop", "assert"):
        return (s[1],)
    if tag == "native":
        return (s[1], s[3], *(v for _r, v in s[2]))
    return ()


def _map_stmt(s, f):
    """Statement ``s`` with its own expressions (not its arms') mapped by ``f``."""
    tag = s[0]
    if tag == "wr":
        return ("wr", f(s[1]), f(s[2]))
    if tag in ("set", "hook"):
        return (tag, s[1], f(s[2]))
    if tag in ("if", "loop", "assert"):
        return (tag, f(s[1]), *s[2:])
    return s


def _arms(s) -> tuple:
    return (s[2], s[3]) if s[0] == "if" else (s[2],) if s[0] == "loop" else ()


def _with_arms(s, arms) -> tuple:
    return ("if", s[1], *arms) if s[0] == "if" else ("loop", s[1], *arms, s[3], s[4])


def statements(stmts) -> Iterable[tuple]:
    """Every statement in ``stmts``, nested ones included."""
    for s in stmts:
        yield s
        for arm in _arms(s):
            yield from statements(arm)


def inline(stmts, procs: Dict[str, list]) -> list:
    """``stmts`` with each ``call`` replaced by the proc's body (procs share variables)."""
    out = []
    for s in stmts:
        if s[0] == "call":
            out += inline(procs[s[1]], procs)
        elif _arms(s):
            out.append(_with_arms(s, [inline(a, procs) for a in _arms(s)]))
        else:
            out.append(s)
    return out


def _known(stmts, flag: str) -> Optional[int]:
    """The constant ``flag`` holds at the end of ``stmts`` whatever the path, None if
    unknown or unassigned."""
    for s in reversed(stmts):
        if s[0] == "set" and s[1] == flag:
            return s[2].v if isinstance(s[2], K) else None
        if s[0] == "if":
            a, b = _known(s[2], flag), _known(s[3], flag)
            if a is not None and a == b:
                return a
            if (
                a is not None
                or b is not None
                or any(t[0] == "set" and t[1] == flag for t in statements(s[2] + s[3]))
            ):
                return None
        elif s[0] == "loop" and any(
            t[0] == "set" and t[1] == flag for t in statements(s[2])
        ):
            return None
    return None


def _skipped(arm, cond) -> bool:
    """Whether an ``if`` on flag test ``cond`` after ``arm`` is never taken."""
    test, flip = (
        (cond.a[0], 1) if isinstance(cond, Op) and cond.op == "lnot" else (cond, 0)
    )
    if not isinstance(test, V) or width(test) != 1:
        return False
    value = _known(arm, test.n)
    return value is not None and bool(value) == bool(flip)


def nest(stmts) -> list:
    """``stmts`` with each ``if`` on a flag that a preceding ``if`` leaves false on one
    arm moved into its other arm (a lowered ``return``'s guard), recursively."""
    out: list = []
    for s in stmts:
        if _arms(s):
            s = _with_arms(s, [nest(a) for a in _arms(s)])
        prev = out[-1] if out else None
        if s[0] == "if" and not s[3] and prev is not None and prev[0] == "if":
            skip = [_skipped(arm, s[1]) for arm in prev[2:]]
            if any(skip):
                arms = [
                    arm if gone else nest(arm + [s])
                    for arm, gone in zip(prev[2:], skip)
                ]
                out[-1] = ("if", prev[1], *arms)
                continue
        out.append(s)
    return out


def rewrite(stmts, expr, stmt=None) -> list:
    """``stmts`` with every expression mapped by ``expr``, then every statement by
    ``stmt`` (a list of statements, or None: kept)."""
    out = []
    for s in stmts:
        s = _map_stmt(s, expr)
        if _arms(s):
            s = _with_arms(s, [rewrite(a, expr, stmt) for a in _arms(s)])
        got = stmt(s) if stmt else None
        out += [s] if got is None else got
    return out


# ---- effects -------------------------------------------------------------------------


@dataclass(frozen=True)
class Effect:
    """What a statement may do: variables assigned, address intervals stored, finishes."""

    sets: FrozenSet[str] = frozenset()
    stores: Tuple[Tuple[int, int], ...] = ()
    finishes: bool = False

    def __or__(self, other: "Effect") -> "Effect":
        return Effect(
            self.sets | other.sets,
            self.stores + other.stores,
            self.finishes or other.finishes,
        )


class Effects:
    """Memoised :class:`Effect` of statements (each kept alive while memoised)."""

    def __init__(self):
        self.memo: Dict[int, tuple] = {}

    def __call__(self, s) -> Effect:
        hit = self.memo.get(id(s))
        if hit is not None:
            return hit[1]
        eff = {
            "set": lambda: Effect(frozenset((s[1],))),
            "wr": lambda: Effect(stores=(span(s[1]),)),
            "finish": lambda: Effect(finishes=True),
        }.get(s[0], Effect)()
        for arm in _arms(s):
            eff = eff | self.seq(arm)
        self.memo[id(s)] = (s, eff)
        return eff

    def seq(self, stmts) -> Effect:
        """The effect of a statement sequence."""
        eff = Effect()
        for s in stmts:
            eff = eff | self(s)
        return eff


def _overlap(spans, intervals) -> bool:
    return any(a <= d and c <= b for a, b in spans for c, d in intervals)


# ---- promotion -----------------------------------------------------------------------


def fix(body, program: Program, memory: Optional[bytes]) -> Tuple[list, Dict[int, int]]:
    """``(body, fixed)``: ``body`` with each fixed-address read of a ``ro`` or ``operand``
    cell's byte that no bounded store reaches, once those are fixed, replaced by its
    byte in ``memory`` (none without), ``fixed`` those bytes."""
    if memory is None:
        return body, {}
    table = np.zeros(0x10000, bool)
    for cell in program.cells:
        if cell.attrs & {"ro", "operand"}:
            table[cell.extent()] = True
    while True:
        fixed: Dict[int, int] = {}

        def leaf(e):
            if isinstance(e, Rd) and isinstance(e.a, K) and table[e.a.v]:
                fixed[e.a.v] = memory[e.a.v]
                return K(memory[e.a.v], 8)
            return None

        out = rewrite(body, lambda e: rebuild(e, leaf))
        hit = table.copy()
        for lo, hi in (span(s[1]) for s in statements(out) if s[0] == "wr"):
            if (lo, hi) != FULL:
                hit[lo : hi + 1] = False
        if (hit == table).all():
            return out, fixed
        table = hit


def cell_var(addr: int) -> V:
    """The variable a promoted cell lives in."""
    return V(f"m_{addr:04x}", 8)


def frame_local(body, keep: np.ndarray) -> List[int]:
    """Constant-address cells ``body`` stores that no call reads before storing, outside
    ``keep`` (a mask over the address space) and any bounded non-constant access."""
    livein = keep.copy()
    stored = np.zeros(0x10000, bool)

    def run(stmts, written):
        for s in stmts:
            for lo, hi in (r for e in _stmt_exprs(s) for r in reads(e)):
                if lo == hi:
                    livein[lo] |= not written[lo]
                elif (lo, hi) != FULL:
                    livein[lo : hi + 1] = True
            tag = s[0]
            if tag == "wr":
                lo, hi = span(s[1])
                if (lo, hi) == FULL:
                    raise OrderError("a store through an unbounded pointer")
                if lo == hi:
                    written[lo] = stored[lo] = True
                livein[lo : hi + 1] |= lo != hi
            elif tag == "if":
                written = run(s[2], written.copy()) & run(s[3], written.copy())
            elif tag == "loop":
                run(s[2], written.copy())
            elif tag == "native":
                raise OrderError("native code is opaque to rescheduling")
        return written

    run(body, np.zeros(0x10000, bool))
    return np.flatnonzero(stored & ~livein).tolist()


def promote(body, cells: Iterable[int]) -> list:
    """``body`` with each of ``cells`` held in a variable instead of memory."""
    cells = frozenset(cells)

    def at(addr) -> Optional[int]:
        lo, hi = span(addr)
        return lo if lo == hi and lo in cells else None

    def leaf(e):
        if isinstance(e, Rd) and at(e.a) is not None:
            return cell_var(at(e.a))
        return None

    def stmt(s):
        if s[0] == "wr" and at(s[1]) is not None:
            return [("set", cell_var(at(s[1])).n, s[2])]
        return None

    return rewrite(body, lambda e: rebuild(e, leaf), stmt)


# ---- instrumentation and placement ---------------------------------------------------


def out_var(slot: Slot) -> V:
    """The output variable of ``slot``."""
    return V(f"o{slot[1]}_{slot[0]}", 8)


def flag_var(slot: Slot) -> V:
    """The written flag of ``slot``."""
    return V(f"h{slot[1]}_{slot[0]}", 1)


def staging(name: str) -> bool:
    """Whether ``name`` is an output variable or flag."""
    return re.fullmatch(r"[oh][12]_\d+", name) is not None


def instrument(body, slots: Sequence[Slot], base: int = SID) -> list:
    """``body`` with SID stores and reads through the slots' output variables (the flags
    cleared first) and no SID store left."""
    second = {r for r, k in slots if k == 2}
    band = [(base, base + REGS - 1)]

    def leaf(e):
        if not isinstance(e, Rd):
            return None
        lo, hi = span(e.a)
        if lo == hi and _overlap(band, [(lo, hi)]):
            slot = (lo - base, 1)
            return Op("ite", 8, (flag_var(slot), out_var(slot), e))
        if _overlap(band, [(lo, hi)]) and (lo, hi) != FULL:
            raise OrderError(f"a read at ${lo:04x}-${hi:04x} may read the SID")
        return None

    def stmt(s):
        if s[0] != "wr" or not _overlap(band, [span(s[1])]):
            return None
        lo, hi = span(s[1])
        if lo != hi:
            raise OrderError(f"a store at ${lo:04x}-${hi:04x} may reach the SID")
        one, two = (lo - base, 1), (lo - base, 2)
        shift = [
            ("set", out_var(two).n, out_var(one)),
            ("set", flag_var(two).n, flag_var(one)),
        ]
        return (shift if one[0] in second else []) + [
            ("set", out_var(one).n, s[2]),
            ("set", flag_var(one).n, TRUE),
        ]

    head = [("set", flag_var(s).n, FALSE) for s in slots]
    head += [("set", out_var(s).n, K(0, 8)) for s in slots]
    return head + rewrite(body, lambda e: rebuild(e, leaf), stmt)


def flush(slot: Slot, base: int = SID) -> tuple:
    """The statement writing ``slot``'s output if its register was stored."""
    return ("if", flag_var(slot), [("wr", K(base + slot[0], 16), out_var(slot))], [])


class _Placer:
    """Flush placement over an instrumented body (see :func:`place`)."""

    def __init__(self, slots: Sequence[Slot], base: int):
        self.slots, self.base = tuple(slots), base
        self.effects = Effects()
        self.flags = {flag_var((r, 1)).n: r for r, _ in slots}

    def regs(self, s) -> FrozenSet[int]:
        """The SID registers statement ``s`` may store."""
        return frozenset(self.flags[n] for n in self.effects(s).sets if n in self.flags)

    def advance(self, done: int, future: FrozenSet[int], out: list) -> int:
        """Flush slots from ``done`` while none of ``future`` stores their register."""
        while done < len(self.slots) and self.slots[done][0] not in future:
            out.append(flush(self.slots[done], self.base))
            done += 1
        return done

    def early(self, s, done: int, after: FrozenSet[int]) -> bool:
        """Whether ``s`` is an ``if`` that may finish, or with an arm that may flush slot
        ``done`` before its end (``after``: the registers stored after ``s``)."""
        if s[0] != "if" or done == len(self.slots):
            return False
        reg = self.slots[done][0]
        return self.effects(s).finishes or any(
            arm
            and (
                reg not in self.regs(arm[-1]) | after
                or self.early(arm[-1], done, after)
            )
            for arm in s[2:]
        )

    def seq(self, stmts, done: int, after: FrozenSet[int]) -> Tuple[list, int]:
        """``stmts`` with flushes placed from ``done`` slots flushed; ``after`` holds the
        registers stored after the sequence."""
        future = [after]
        for s in reversed(stmts):
            future.append(future[-1] | self.regs(s))
        future.reverse()
        out: list = []
        done = self.advance(done, future[0], out)
        for i, s in enumerate(stmts):
            if s[0] == "finish":
                self.advance(done, frozenset(), out)
                return out + [s], len(self.slots)
            if self.early(s, done, future[i + 1]):
                then, a = self.seq(s[2], done, future[i + 1])
                other, b = self.seq(s[3], done, future[i + 1])
                out.append(("if", s[1], then, other))
                done = min(a, b)
                continue
            out.append(s)
            done = self.advance(done, future[i + 1], out)
        return out, done


def place(body, slots: Sequence[Slot], base: int = SID) -> list:
    """An instrumented ``body`` with each slot flushed once on every path, in order, at
    the earliest boundary after its register's last store, in each arm of a branch
    where that falls inside it."""
    return _Placer(slots, base).seq(body, 0, frozenset())[0]


def flush_order(body, slots: Sequence[Slot]) -> FrozenSet[tuple]:
    """Every sequence of slots a path through a placed ``body`` flushes."""
    at = {flag_var(s).n: s for s in slots}
    ended: set = set()

    def run(stmts, paths):
        for s in stmts:
            if s[0] == "if" and isinstance(s[1], V) and s[1].n in at:
                paths = {p + (at[s[1].n],) for p in paths}
            elif s[0] == "if":
                paths = run(s[2], paths) | run(s[3], paths)
            elif s[0] == "finish":
                ended.update(paths)
                return set()
        return paths

    return frozenset(run(body, {()}) | ended)


# ---- forwarding and dead code --------------------------------------------------------


@dataclass
class _Fact:
    expr: object
    uses: FrozenSet[str]
    spans: List[Tuple[int, int]]


def _leaf(e) -> bool:
    return isinstance(e, (K, V)) or isinstance(e, Rd) and isinstance(e.a, K)


class _Forward:
    """Constant, copy and single-use propagation under the conditions known to hold,
    merging arms at joins into an ``ite`` on a snapshot of the condition."""

    def __init__(self, body):
        self.effects = Effects()
        self.counts: Dict[str, int] = {}
        self.guards = 0
        for s in statements(body):
            for e in _stmt_exprs(s):
                for n in names(e):
                    self.counts[n] = self.counts.get(n, 0) + 1

    def free(self, name: str) -> bool:
        """Whether any expression may be forwarded into ``name``'s uses."""
        return staging(name) or self.counts.get(name, 0) <= 1

    @staticmethod
    def sub(e, env: dict):
        """``e`` with known variables and conditions replaced."""

        def leaf(n):
            fact = env.get(n)
            if fact is None and isinstance(n, V):
                fact = env.get(n.n)
            return None if fact is None else fact.expr

        return rebuild(e, leaf)

    @staticmethod
    def kill(env: dict, eff: Effect) -> None:
        """Drop the facts ``eff`` clobbers."""
        for key in [
            k
            for k, f in env.items()
            if k in eff.sets or f.uses & eff.sets or _overlap(eff.stores, f.spans)
        ]:
            del env[key]

    def learn(self, env, name: str, e) -> None:
        """Record ``name := e`` where it may be forwarded."""
        uses = names(e)
        if name in uses or volatile(e) or not (self.free(name) or _leaf(e)):
            return
        env[name] = _Fact(e, uses, reads(e))

    @staticmethod
    def assume(env: dict, cond, truth: bool) -> dict:
        """``env`` and the fact that ``cond`` is ``truth``."""
        out = dict(env)
        if volatile(cond):
            return out
        fact = _Fact(TRUE if truth else FALSE, names(cond), reads(cond))
        out[cond] = fact
        if isinstance(cond, Op) and cond.op == "lnot":
            out[cond.a[0]] = _Fact(FALSE if truth else TRUE, fact.uses, fact.spans)
        return out

    def seq(self, stmts, env: dict) -> list:
        """``stmts`` rewritten under the facts ``env`` (updated in place)."""
        out: list = []
        for s in stmts:
            if s[0] == "if":
                out += self.branch(s, env)
                continue
            if s[0] == "loop":
                self.kill(env, self.effects(s))
                inner = dict(env)
                out.append(
                    ("loop", self.sub(s[1], inner), self.seq(s[2], inner), *s[3:])
                )
                continue
            s = _map_stmt(s, lambda e: self.sub(e, env))
            out.append(s)
            self.kill(env, self.effects(s))
            if s[0] == "set":
                self.learn(env, s[1], s[2])
            if s[0] == "finish":
                break
        return out

    def arm(self, name: str, mine: dict, arm: list, other: dict):
        """The value ``name`` has at the end of an arm, None where unknown."""
        if name in mine:
            return mine[name].expr
        if name in self.effects.seq(arm).sets:
            return None
        return V(name, width(other[name].expr))

    def branch(self, s, env: dict) -> list:
        """An ``if``: a known condition picks its arm, else both arms run and merge."""
        cond = self.sub(s[1], env)
        if isinstance(cond, K) and cond.v is not None:
            return self.seq(s[2] if cond.v else s[3], env)
        then_env, else_env = self.assume(env, cond, True), self.assume(env, cond, False)
        then = self.seq(s[2], then_env)
        other = self.seq(s[3], else_env)
        eff = self.effects(s)
        self.kill(env, eff)
        merged = {}
        for name in {n for n in set(then_env) | set(else_env) if isinstance(n, str)}:
            if name in eff.sets and self.free(name):
                a = self.arm(name, then_env, s[2], else_env)
                b = self.arm(name, else_env, s[3], then_env)
                if a is not None and b is not None:
                    merged[name] = (a, b)
        out = []
        test = cond
        if merged and not (isinstance(cond, V) and cond.n.startswith("g_")):
            test = None
            if not volatile(cond):
                self.guards += 1
                test = V(f"g_{self.guards}", 1)
                out.append(("set", test.n, cond))
        for name, (a, b) in merged.items():
            if test is not None:
                self.learn(env, name, fold("ite", width(a), (test, a, b)))
        if not then and not other and not volatile(cond):
            return out
        if not then:
            return out + [("if", fold("lnot", 1, (cond,)), other, [])]
        return out + [("if", cond, then, other)]


def forward(body) -> list:
    """``body`` with values forwarded to their uses (see :class:`_Forward`)."""
    return _Forward(body).seq(body, {})


def eliminate(
    stmts, live: FrozenSet[str], end: FrozenSet[str]
) -> Tuple[list, FrozenSet]:
    """``(stmts, live before)`` with dead assignments and empty branches removed;
    ``live`` is read after ``stmts``, ``end`` when the call ends."""
    out: list = []
    for s in reversed(stmts):
        tag = s[0]
        if tag == "set":
            if s[1] not in live and not volatile(s[2]):
                continue
            live = (live - {s[1]}) | names(s[2])
        elif tag == "if":
            then, a = eliminate(s[2], live, end)
            other, b = eliminate(s[3], live, end)
            if not then and not other and not volatile(s[1]):
                continue
            s = ("if", s[1], then, other)
            live = a | b | names(s[1])
        elif tag == "loop":
            inner = live | names(s[1])
            while True:
                body, entry = eliminate(s[2], inner, end)
                if entry <= inner:
                    break
                inner = inner | entry
            s = ("loop", s[1], body, s[3], s[4])
            live = inner
        elif tag == "finish":
            live = end
        else:
            live = live.union(*map(names, _stmt_exprs(s)))
        out.append(s)
    out.reverse()
    return out, live


# ---- the transform -------------------------------------------------------------------


@dataclass
class Counts:
    """Static size of a statement tree."""

    statements: int
    sid_writes: int
    stores: int
    variables: int
    cells: int


def counts(stmts, base: int = SID) -> Counts:
    """The :class:`Counts` of ``stmts``: SID stores, other stores, variables assigned and
    constant-address cells stored."""
    every = list(statements(stmts))
    spans = [span(s[1]) for s in every if s[0] == "wr"]
    sid = [p for p in spans if _overlap([(base, base + REGS - 1)], [p])]
    return Counts(
        len(every),
        len(sid),
        len(spans) - len(sid),
        len({s[1] for s in every if s[0] == "set"}),
        len({lo for lo, hi in spans if lo == hi and (lo, hi) not in sid}),
    )


@dataclass
class Reordered:
    """A rescheduled program and what the transform did.

    ``staged`` lists the output variables still assigned: buffers the order forces.
    """

    program: Program
    slots: Tuple[Slot, ...]
    promoted: List[int]
    staged: List[str]
    before: Counts
    inlined: Counts
    after: Counts
    paths: FrozenSet[tuple] = field(default_factory=frozenset)
    ranges: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    fixed: Dict[int, int] = field(default_factory=dict)


def reorder(
    program: Program,
    order,
    *,
    base: int = SID,
    observed: Iterable[Tuple[int, int]] = (),
    memory: Optional[bytes] = None,
) -> Reordered:
    """``program`` rescheduled to write the kept writes of ``order`` (text or slots) in
    order; ``observed`` lists address intervals other code reads, never promoted;
    reads of ``ro`` cells at fixed addresses take their values from ``memory``."""
    slots = parse_order(order) if isinstance(order, str) else tuple(order)
    body, fixed = fix(nest(inline(program.body, program.procs)), program, memory)
    known = ranges(body, [n for n, _w, _d in program.controls] + list(program.inputs))
    token = RANGES.set(known)
    try:
        return replace(
            _reorder(program, body, slots, base, observed, known), fixed=fixed
        )
    finally:
        RANGES.reset(token)


def _reorder(program, body, slots, base, observed, known) -> Reordered:
    keep = np.zeros(0x10000, bool)
    keep[base : base + REGS] = True
    for lo, hi in observed:
        keep[lo : hi + 1] = True
    for cell in program.cells:
        if "volatile" in cell.attrs:
            keep[cell.extent()] = True
    cells = frame_local(body, keep)
    placed = place(instrument(promote(body, cells), slots, base), slots, base)
    paths = flush_order(placed, slots)
    if paths != {slots}:
        raise OrderError("a path flushes the slots out of order")
    end = frozenset(n for n, _w, _d in program.controls)
    out = eliminate(forward(placed), end, end)[0]
    staged = sorted({s[1] for s in statements(out) if s[0] == "set" and staging(s[1])})
    every = program.body + [s for p in program.procs.values() for s in p]
    return Reordered(
        replace(program, body=out, procs={}),
        slots,
        cells,
        staged,
        counts(every, base),
        counts(body, base),
        counts(out, base),
        paths,
        known,
    )


# ---- evidence ------------------------------------------------------------------------


def _wild(e) -> list:
    return [n.a for n in walk(e) if isinstance(n, Rd) and span(n.a) == FULL]


def _logged(body, slots: Sequence[Slot], base: int, cells: Sequence[int]) -> list:
    """``body`` with each SID store appended to the log variables instead of memory, and
    ``alias`` set where an unbounded read may hit one of ``cells``."""
    regs = frozenset(range(base, base + REGS))

    def log(reg, value) -> list:
        out = [("set", "log_t", value)]
        for i in range(len(slots) + 1):
            at = Op("eq", 1, (V("log_n", 8), K(i, 8)))
            out += [
                ("set", f"log_r{i}", Op("ite", 8, (at, K(reg, 8), V(f"log_r{i}", 8)))),
                (
                    "set",
                    f"log_v{i}",
                    Op("ite", 8, (at, V("log_t", 8), V(f"log_v{i}", 8))),
                ),
            ]
        return out + [("set", "log_n", Op("add", 8, (V("log_n", 8), K(1, 8))))]

    def alias(s) -> list:
        hits = [
            Op("eq", 1, (a, K(c, 16)))
            for e in _stmt_exprs(s)
            for a in _wild(e)
            for c in cells
        ]
        return [("set", "alias", Op("lor", 1, (V("alias", 1), h))) for h in hits]

    def stmt(s):
        if s[0] == "wr" and isinstance(s[1], K) and s[1].v in regs:
            return alias(s) + log(s[1].v - base, s[2])
        if s[0] == "loop":
            return alias(s) + [("loop", s[1], s[2] + alias(s), s[3], s[4])]
        return alias(s) + [s]

    head = [("set", "alias", FALSE), ("set", "log_n", K(0, 8))]
    head += [
        ("set", f"log_{c}{i}", K(0, 8)) for i in range(len(slots) + 1) for c in "rv"
    ]
    return head + rewrite(body, lambda e: e, stmt)


def check(
    original: Program,
    result: Reordered,
    base: int = SID,
    *,
    timeout_ms: int = 0,
    progress: bool = False,
) -> Optional[tuple]:
    """None if, from every memory and control state where no unbounded read hits a
    promoted cell or the SID, ``result`` writes exactly the kept writes of ``original`` in order and
    leaves the same memory and controls outside the SID and the promoted cells; else
    ``(what differs, z3 model)``, the model None where the solver gave up (after
    ``timeout_ms`` per claim, if set)."""
    token = RANGES.set(result.ranges)
    try:
        return _check(original, result, base, timeout_ms, progress)
    finally:
        RANGES.reset(token)


def _check(original, result, base, timeout_ms, progress) -> Optional[tuple]:
    slots, cells = result.slots, result.promoted
    flushes = [flush(s, base) for s in slots]
    reference = rewrite(
        instrument(inline(original.body, original.procs), slots, base) + [("finish",)],
        lambda e: e,
        lambda s: flushes + [s] if s[0] == "finish" else None,
    )
    memo: dict = {}
    a, b = (
        smt.step(
            replace(
                original,
                body=_logged(body, slots, base, [*cells, *range(base, base + REGS)]),
                procs={},
            ),
            smt.symbolic_memory("pre"),
            smt.initial_env(original, "pre_"),
        )
        for body in (reference, result.program.body)
    )
    runs = (a, b)
    stored = {x for st in runs for x in st.writes}
    fixed = sorted({x.as_long() for x in stored if z3.is_bv_value(x)} - set(cells))
    claims = {
        f"${x:04x}": smt.select(a.mem, z3.BitVecVal(x, 16), memo)
        != smt.select(b.mem, z3.BitVecVal(x, 16), memo)
        for x in fixed
        if not base <= x < base + REGS
    }
    if any(not z3.is_bv_value(x) for x in stored):
        probe = z3.BitVec("probe", 16)
        away = [probe != x for x in fixed + list(cells)] + [
            z3.ULT(probe, base) | z3.UGE(probe, base + REGS)
        ]
        claims["memory"] = z3.And(
            *away, smt.select(a.mem, probe, memo) != smt.select(b.mem, probe, memo)
        )
    keys = ["log_n"] + [f"log_{c}{i}" for i in range(len(slots) + 1) for c in "rv"]
    claims.update(
        (n, a.env[n] != b.env[n]) for n in keys + [n for n, _w, _d in original.controls]
    )
    solver = z3.Solver()
    solver.add(z3.Not(a.env["alias"]), z3.Not(b.env["alias"]))
    solver.add(
        *[
            z3.Select(smt.symbolic_memory("pre"), x) == v
            for x, v in result.fixed.items()
        ]
    )
    if timeout_ms:
        solver.set("timeout", timeout_ms)
    for what, claim in tqdm(
        claims.items(), desc="check", unit="claim", disable=not progress
    ):
        claim = z3.simplify(claim)
        if not z3.is_false(claim):
            answer = solver.check(claim)
            if answer != z3.unsat:
                return what, solver.model() if answer == z3.sat else None
    return None


def played(player, frames: int, base: int = SID) -> Tuple[np.ndarray, List[list]]:
    """``(grid, writes)``: ``player``'s register grid over ``frames`` calls and each
    call's ``(register, value)`` SID stores in order."""
    calls: List[list] = []
    store = player._store  # pylint: disable=protected-access

    def logged(addr: int, value: int) -> None:
        if base <= addr < base + REGS:
            calls[-1].append((addr - base, value))
        store(addr, value)

    player._store = logged  # pylint: disable=protected-access
    rows = []
    for _ in range(frames):
        calls.append([])
        player.play_frame()
        rows.append(player.regs)
    return np.asarray(rows, np.uint8).reshape(-1, REGS), calls


def compare(make, order, frames: int, base: int = SID, **options) -> Tuple:
    """``(verdict, results)``: ``verdict`` None if the player ``make(transform)`` builds
    plays the same grid rescheduled to ``order`` as without (``make(None)``), each call
    writing exactly the original's kept writes in order, else where they first differ;
    ``results`` the :class:`Reordered` of each program. ``options`` go to :func:`reorder`;
    where the tune's own 6502 code runs around the spec, no cell is promoted.
    """
    slots = parse_order(order) if isinstance(order, str) else tuple(order)
    results: List[Reordered] = []
    player = make(None)
    grid, writes = played(player, frames, base)
    if player.entry != player._play_addr:  # pylint: disable=protected-access
        options.setdefault("observed", [(0, 0xFFFF)])

    def transform(program: Program, memory: bytes) -> Program:
        results.append(reorder(program, slots, base=base, memory=memory, **options))
        return results[-1].program

    new_grid, new_writes = played(make(transform), frames, base)
    for frame, (old, new) in enumerate(zip(writes, new_writes)):
        if kept(old, slots) != new:
            return f"frame {frame}: writes {new}, keeps {kept(old, slots)}", results
    rows = np.flatnonzero((grid != new_grid).any(axis=1))
    return (f"frame {rows[0]}: grid differs" if len(rows) else None), results


def survey(path: str, order, frames: int, subtune: Optional[int] = None) -> dict:
    """:func:`compare` on a tune an installed format plays with a spec: the format, the
    verdict, and per program its staged variables, promoted cells and counts inlined
    and after."""
    from ..formats import find_format  # pylint: disable=import-outside-toplevel

    data = Path(path).read_bytes()
    fmt = find_format(data)
    model = fmt.load(data, subtune)
    verdict, results = compare(
        lambda transform: fmt.spec_player(model, transform=transform), order, frames
    )
    return {
        "format": fmt.name,
        "verdict": verdict,
        "programs": [
            {
                "staged": r.staged,
                "promoted": len(r.promoted),
                "inlined": asdict(r.inlined),
                "after": asdict(r.after),
            }
            for r in results
        ],
    }


__all__ = [
    "ORDER",
    "Counts",
    "OrderError",
    "Reordered",
    "check",
    "compare",
    "kept",
    "parse_order",
    "played",
    "survey",
    "reorder",
]
