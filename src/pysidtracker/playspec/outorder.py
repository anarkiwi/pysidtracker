"""Reschedule a lowered playspec so its SID register writes come out in a canonical order.

:func:`reorder` inlines, promotes frame-local cells, instruments SID stores as output
variables, places each slot's flush, forwards values and removes dead code; :func:`check`
proves the result with the solver. See docs/playspec.md, "Output order".
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from functools import partial
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import z3
from tqdm import tqdm

from ..errors import SidError
from . import smt
from .emit import declared, emit, isomorphic, pruned
from .ir import K, Op, Program, Rd, V, width
from .lower import elaborate, mk
from .syntax import parse

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
ORDER = "fc_lo,fc_hi,res_filt,mode_vol,voice(segments(sr,ad,ctrl):4,freq_lo,freq_hi,pw_lo,pw_hi)"
EDGE_BITS = 0x09
NATIVE = REGS
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
    if name not in table or not (k == "" or k.isdigit() and int(k) > 0):
        raise OrderError(f"no register {token!r}")
    return (table[name] if voice is None else 7 * voice + table[name], int(k or 1))


def _rounds(token: str) -> List[str]:
    """The tokens of ``segments(a,b,c):N``, oldest segment first, else ``token``."""
    found = re.fullmatch(r"segments\((.*)\):(\d+)", token)
    if found is None:
        return [token]
    names = found.group(1).split(",")
    return [f"{n}:{k}" for k in range(int(found.group(2)), 0, -1) for n in names]


def parse_order(text: str) -> Tuple[Slot, ...]:
    """The ``(register, k)`` slots of an order: song register names, ``voice(...)``
    (each voice's registers in turn) and ``interleave(...)`` (register by register).
    ``name`` is a register's last write in a call; a voice's ``sr``, ``ad`` and ``ctrl``
    also have ``name:k``, the one of the ``k``-th last gate segment, which
    ``segments(sr,ad,ctrl):N`` lists for ``N`` segments, oldest first."""
    slots: List[Slot] = []
    for item in _split(text):
        group = re.fullmatch(r"(voice|interleave)\((.*)\)", item)
        if group is None:
            slots.append(_token(item, None))
            continue
        tokens = [r for t in _split(group.group(2)) for r in _rounds(t)]
        pairs = [(v, t) for v in range(3) for t in tokens]
        if group.group(1) == "interleave":
            pairs = [(v, t) for t in tokens for v in range(3)]
        slots += [_token(t, v) for v, t in pairs]
    at = {s: i for i, s in enumerate(slots)}
    if len(at) != len(slots) or {r for r, k in slots if k == 1} != set(range(REGS)):
        raise OrderError(f"{text!r} must keep every register's last write once")
    if any((r, k - 1) not in at or at[(r, k)] > at[(r, k - 1)] for r, k in at if k > 1):
        raise OrderError(f"{text!r} must list a register's segments oldest first")
    d = depth(slots)
    if any(len({d[7 * v + o] for o in SEGMENTED}) > 1 for v in range(3)) or any(
        d[r] > 1 for r in range(REGS) if r >= 21 or r % 7 not in SEGMENTED
    ):
        raise OrderError(f"{text!r} must give sr, ad and ctrl the same segments")
    return tuple(slots)


def depth(slots: Sequence[Slot]) -> Dict[int, int]:
    """Each register's number of segments in ``slots``."""
    out: Dict[int, int] = {}
    for r, k in slots:
        out[r] = max(out.get(r, 0), k)
    return out


SEGMENTED = (4, 5, 6)


def segments(slots: Sequence[Slot]) -> Dict[int, int]:
    """Each voice ``slots`` writes by gate segment, with its number of segments."""
    d = depth(slots)
    return {v: d[7 * v + 4] for v in range(3) if d[7 * v + 4] > 1}


def segment_regs(deep: Dict[int, int]) -> FrozenSet[int]:
    """The registers of the voices ``deep`` (:func:`segments`) writes by segment."""
    return frozenset(7 * v + o for v in deep for o in SEGMENTED)


def _closed(writes: Sequence[Tuple[int, int]], voice: int, n: int) -> Tuple[list, dict]:
    """``(closed, open)`` of a voice's writes: the segments ended by a gate or test
    edge (the first ctrl write is one), ``(sr, ad, ctrl)`` each, the last ``n - 1``,
    and the writes since the last edge."""
    closed: List[tuple] = []
    pending: Dict[int, int] = {}
    last = None
    for reg, value in writes:
        off = reg - 7 * voice
        if off not in SEGMENTED:
            continue
        if off == 4 and (last is None or (last ^ value) & EDGE_BITS):
            closed.append((pending.get(6), pending.get(5), value))
            pending = {}
        else:
            pending[off] = value
        last = value if off == 4 else last
    return closed[max(len(closed) - n + 1, 0) :], pending


def kept(
    writes: Sequence[Tuple[int, int]], slots: Sequence[Slot], native=frozenset()
) -> List[Tuple]:
    """The ``(register, value)`` writes of one call that ``slots`` keep, in their order,
    after the ``native`` registers' writes in place."""
    deep = {v: n for v, n in segments(slots).items() if 7 * v + 4 not in native}
    slots = live_slots(slots, native)
    regs = np.fromiter((r for r, _ in writes), np.int64, len(writes))
    table: Dict[Slot, int] = {}
    for reg, k in slots:
        at = np.flatnonzero(regs == reg)
        if reg not in segment_regs(deep) and len(at) >= k:
            table[(reg, k)] = int(writes[at[-k]][1])
    for v, n in deep.items():
        closed, pending = _closed(writes, v, n)
        for j, group in enumerate(reversed(closed), 2):
            table.update(
                ((7 * v + 6 - i, j), g) for i, g in enumerate(group) if g is not None
            )
        table.update(((7 * v + o, 1), x) for o, x in pending.items())
    out = [tuple(w) for w in writes if w[0] in native]
    return out + [(r, table[(r, k)]) for r, k in slots if (r, k) in table]


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


def ranges(body, unknown: Iterable[str], procs=()) -> Dict[str, Tuple[int, int]]:
    """An interval holding 0 and every value each variable ``body`` and ``procs`` assign
    takes (``unknown``: variables that also hold values from outside, left out)."""
    unknown = frozenset(unknown)
    assigns = [
        (s[1], s[2])
        for b in [body, *procs]
        for s in statements(b)
        if s[0] == "set" and s[1] not in unknown
    ]
    table: Dict[str, Tuple[int, int]] = {}
    for _ in range(8):
        token = RANGES.set(table)
        new: Dict[str, Tuple[int, int]] = {}
        for name, value in assigns:
            lo, hi = span(value)
            old = new.get(name, (0, 0))
            new[name] = (min(lo, old[0]), max(hi, old[1]))
        RANGES.reset(token)
        if new == table:
            break
        table = new
    return table


BITS: ContextVar[Dict[str, Tuple[int, int]]] = ContextVar("BITS", default={})


def bits(e) -> Tuple[int, int]:
    """``(known, value)``: the bits of ``e`` known whatever the state, and their values
    (a variable's from :data:`BITS`)."""
    top = (1 << max(width(e), 1)) - 1
    if isinstance(e, K):
        return (0, 0) if e.v is None else (top, e.v & top)
    if isinstance(e, V):
        return BITS.get().get(e.n, (0, 0))
    if isinstance(e, Rd):
        return (0, 0)
    op, args = e.op, [bits(a) for a in e.a]
    (m, v), rest = args[0], args[1:]
    if op in ("and", "or", "xor"):
        n, w = rest[0]
        if op == "xor":
            return m & n, (v ^ w) & m & n
        if op == "and":
            zero, one = (m & ~v) | (n & ~w), m & v & n & w
        else:
            zero, one = (m & ~v) & (n & ~w), (m & v) | (n & w)
        return zero | one, one
    if op == "not":
        return m, ~v & m
    if op == "zext":
        return m | (top & ~((1 << width(e.a[0])) - 1)), v
    if op == "lo":
        return m & top, v & top
    if op == "hi":
        return m >> 8, v >> 8
    if op in ("shl", "shr") and isinstance(e.a[1], K) and e.a[1].v is not None:
        k = e.a[1].v
        if op == "shl":
            return ((m << k) | ((1 << k) - 1)) & top, (v << k) & top
        return ((m >> k) | (top & ~(top >> k))) & top, v >> k
    if op == "word":
        n, w = rest[0]
        return (m | n << 8) & top, (v | w << 8) & top
    if op == "ite":
        (n, w), (x, y) = rest
        if isinstance(e.a[0], K) and e.a[0].v is not None:
            return (n, w) if e.a[0].v else (x, y)
        both = n & x & ~(w ^ y)
        return both, w & both
    return (0, 0)


def known_bits(body, unknown: Iterable[str], procs=()) -> Dict[str, Tuple[int, int]]:
    """The bits every variable ``body`` and ``procs`` assign holds in all of its values,
    zero included (``unknown``: variables that also hold values from outside, left out).
    """
    unknown = frozenset(unknown)
    assigns = [
        (s[1], s[2])
        for b in [body, *procs]
        for s in statements(b)
        if s[0] == "set" and s[1] not in unknown
    ]
    table: Dict[str, Tuple[int, int]] = {}
    for _ in range(8):
        token = BITS.set(table)
        new: Dict[str, Tuple[int, int]] = {}
        for name, value in assigns:
            m, v = bits(value)
            top = (1 << max(width(value), 1)) - 1
            old = new.get(name, (top, 0))
            both = old[0] & m & ~(old[1] ^ v)
            new[name] = (both, v & both)
        BITS.reset(token)
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


def tidy(e):
    """``e`` folded bottom-up (:func:`fold`), every node built as lowering builds it."""
    if isinstance(e, (K, V)):
        return e
    if isinstance(e, Rd):
        return Rd(tidy(e.a))
    if e.op == "vol":
        return Op("vol", 8, (tidy(e.a[0]),))
    return fold(e.op, e.w, tuple(tidy(a) for a in e.a))


def stmt_exprs(s) -> tuple:
    tag = s[0]
    if tag == "wr":
        return (s[1], s[2])
    if tag in ("set", "hook"):
        return (s[2],)
    if tag in ("if", "loop", "assert"):
        return (s[1],)
    if tag == "native":
        return (s[1], s[3], *(v for _r, v in s[2]))
    if tag == "obj":
        return s[4]
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
    if tag == "obj":
        return (*s[:4], tuple(f(e) for e in s[4]), *s[5:])
    return s


def branches(s) -> tuple:
    if s[0] == "if":
        return (s[2], s[3])
    return (s[2],) if s[0] in ("loop", "fire") else ()


def _with_arms(s, arms) -> tuple:
    if s[0] == "fire":
        return ("fire", s[1], arms[0])
    return ("if", s[1], *arms) if s[0] == "if" else ("loop", s[1], *arms, s[3], s[4])


def statements(stmts) -> Iterable[tuple]:
    """Every statement in ``stmts``, nested ones included."""
    for s in stmts:
        yield s
        for arm in branches(s):
            yield from statements(arm)


def inline(stmts, procs: Dict[str, list], keep=frozenset()) -> list:
    """``stmts`` with each ``call`` but those of procs in ``keep`` replaced by the proc's
    body (procs share variables)."""
    out = []
    for s in stmts:
        if s[0] == "call" and s[1] not in keep:
            out += inline(procs[s[1]], procs, keep)
        elif s[0] == "fire":
            out += inline(s[2], procs, keep)
        elif branches(s):
            out.append(_with_arms(s, [inline(a, procs, keep) for a in branches(s)]))
        else:
            out.append(s)
    return out


def calls(stmts) -> Iterable[str]:
    """The id of each proc called in ``stmts``, once per call site."""
    return (s[1] for s in statements(stmts) if s[0] == "call")


def reached(body, procs: Dict[str, list]) -> List[str]:
    """The procs ``body`` reaches, each before every proc it calls."""
    order: List[str] = []
    seen: set = set()

    def visit(pid: str) -> None:
        if pid not in seen:
            seen.add(pid)
            for callee in calls(procs[pid]):
                visit(callee)
            order.append(pid)

    for pid in calls(body):
        visit(pid)
    return order[::-1]


def outlined(body, procs: Dict[str, list]) -> FrozenSet[str]:
    """The procs a call to which is smaller than the body inlined (call sites times
    statements above their sum), unless the body, inlined, may ``finish`` or run native
    code: no flush can be placed inside a call."""
    order = reached(body, procs)
    sites: Dict[str, int] = {}
    for stmts in [body, *(procs[pid] for pid in order)]:
        for pid in calls(stmts):
            sites[pid] = sites.get(pid, 0) + 1
    out: set = set()
    for pid in reversed(order):
        flat = list(statements(inline(procs[pid], procs, out)))
        if sites[pid] * len(flat) > sites[pid] + len(flat) and not {
            s[0] for s in flat
        } & {"finish", "native"}:
            out.add(pid)
    return frozenset(out)


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
        if branches(s):
            s = _with_arms(s, [nest(a) for a in branches(s)])
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
        if branches(s):
            s = _with_arms(s, [rewrite(a, expr, stmt) for a in branches(s)])
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
    """Memoised :class:`Effect` of statements (each kept alive while memoised); a ``call``
    has the effect of its proc in ``procs``."""

    def __init__(self, procs: Optional[Dict[str, list]] = None):
        self.memo: Dict[int, tuple] = {}
        self.procs = {} if procs is None else procs
        self.called: Dict[str, Effect] = {}

    def __call__(self, s) -> Effect:
        hit = self.memo.get(id(s))
        if hit is not None:
            return hit[1]
        eff = {
            "set": lambda: Effect(frozenset((s[1],))),
            "wr": lambda: Effect(stores=(span(s[1]),)),
            "obj": lambda: Effect(
                (frozenset(s[3:4]) | frozenset(s[8])) - {None},
                tuple((a, a) for a in s[6]),
            ),
            "finish": lambda: Effect(finishes=True),
        }.get(s[0], Effect)()
        for arm in branches(s):
            eff = eff | self.seq(arm)
        if s[0] == "call":
            if s[1] not in self.called:
                self.called[s[1]] = self.seq(self.procs[s[1]])
            eff = self.called[s[1]]
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


def _lookup(memory, mask, seen: dict, addr: int) -> Optional[int]:
    """The byte at ``addr`` where ``mask`` holds it fixed (recorded in ``seen``)."""
    if not mask[addr]:
        return None
    seen[addr] = memory[addr]
    return memory[addr]


def fix(
    body, program: Program, memory: Optional[bytes], observed=(), proof=None, procs=None
) -> tuple:
    """``(body, procs, fixed)``: ``body`` and ``procs`` forwarded (:func:`forward`) with
    each fixed-address read of a declared byte replaced by its byte in ``memory``, once
    no bounded store of the result reaches any byte so replaced; ``fixed`` those bytes.
    Without ``memory`` nothing is fixed; a non-``ro`` byte in an ``observed`` interval
    never is."""
    procs = {} if procs is None else procs
    table = np.zeros(0x10000, bool)
    if memory is not None:
        for cell in program.cells:
            if "volatile" not in cell.attrs:
                table[cell.extent()] = True
        for lo, hi in observed:
            table[lo : hi + 1] = False
        for cell in program.cells:
            if "ro" in cell.attrs:
                table[cell.extent()] = True
    while True:
        fixed: Dict[int, int] = {}

        constant = partial(_lookup, memory, table, fixed)
        trace: list = []
        out, outs = forward(body, constant, trace, procs)
        hit = table.copy()
        every = [s for b in [out, *outs.values()] for s in statements(b)]
        for lo, hi in (span(s[1]) for s in every if s[0] == "wr"):
            if (lo, hi) != FULL:
                hit[lo : hi + 1] = False
        for s in every:
            if s[0] == "obj":
                hit[list(s[6])] = False
        if not (table & ~hit)[list(fixed)].any():
            if proof is not None:
                proof.extend(trace)
            return out, outs, fixed
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
            for lo, hi in (r for e in stmt_exprs(s) for r in reads(e)):
                if lo == hi:
                    livein[lo] |= not written[lo]
                elif (lo, hi) != FULL:
                    livein[lo : hi + 1] = True
            tag = s[0]
            if tag == "obj":
                for lo, hi in s[5]:
                    livein[lo : hi + 1] = True
                livein[list(s[6])] = True
                if s[7]:
                    livein[:] = True
            elif tag == "wr":
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


def last_var(reg: int) -> V:
    """The last value stored to a segmented register."""
    return V(f"last_{reg}", 8)


def seen_var(reg: int) -> V:
    """Whether a segmented register was stored."""
    return V(f"seen_{reg}", 1)


def staging(name: str) -> bool:
    """Whether ``name`` is an output variable or flag."""
    return re.fullmatch(r"[oh]\d+_\d+", name) is not None


def split(body, base: int = SID, count: Optional[list] = None) -> list:
    """``body`` with each bounded non-constant access that may reach the SID taken
    apart into one constant-address access per SID register it may reach, and the
    access as before where it may reach none (``count``: the numbering of the address
    variables, shared by the bodies of a program)."""
    band = (base, base + REGS - 1)
    count = [0] if count is None else count

    def cases(addr) -> list:
        lo, hi = span(addr)
        if lo == hi or (lo, hi) == FULL or hi < band[0] or lo > band[1]:
            return []
        return list(range(max(lo, band[0]), min(hi, band[1]) + 1))

    def leaf(e):
        if not isinstance(e, Rd) or not cases(e.a):
            return None
        out = Rd(rebuild(e.a, leaf))
        for c in reversed(cases(e.a)):
            hit = Op("eq", 1, (e.a, K(c, 16)))
            out = Op("ite", 8, (hit, Rd(K(c, 16)), out))
        return out

    def stmt(s):
        if s[0] != "wr" or not cases(s[1]):
            return None
        count[0] += 1
        at, value = f"sa_{count[0]}", f"sv_{count[0]}"
        RANGES.get()[at] = span(s[1])
        lo, hi = span(s[1])
        inside = band[0] <= lo and hi <= band[1]
        out = [] if inside else [("wr", V(at, 16), V(value, 8))]
        for c in reversed(cases(s[1])):
            hit = Op("eq", 1, (V(at, 16), K(c, 16)))
            out = [("if", hit, [("wr", K(c, 16), V(value, 8))], out)]
        return [("set", at, s[1]), ("set", value, s[2])] + out

    return rewrite(body, lambda e: rebuild(e, leaf), stmt)


def truth(cond, flags: FrozenSet) -> Optional[bool]:
    """The value of ``cond`` given the one-bit variables ``flags`` hold, None if unknown."""
    if isinstance(cond, K):
        return None if cond.v is None else bool(cond.v)
    if isinstance(cond, V):
        return dict(flags).get(cond.n)
    if not isinstance(cond, Op) or cond.op not in ("lnot", "land", "lor"):
        return None
    got = [truth(a, flags) for a in cond.a]
    if cond.op == "lnot":
        return None if got[0] is None else not got[0]
    stop = cond.op == "lor"
    if stop in got:
        return stop
    return None if None in got else not stop


class _Edges:
    """The most gate and test edges a path through a body may make in stores to one
    ctrl register: a store ends a segment unless its value is the last one stored or has
    the same :data:`EDGE_BITS` known. A path is the state before a store, None (none
    yet) or ``(known, value, expr)`` of the last value stored (``expr`` dropped once an
    assignment or store may change it), with the constants the variables of
    loop conditions hold; a loop runs on while its condition may hold."""

    MERGE = 64

    def __init__(self, addr: int, body, procs: Dict[str, list], watch: FrozenSet[str]):
        self.addr, self.watch, self.procs = addr, watch, procs
        self.effects = Effects(procs)
        self.stores: Dict[int, bool] = {}
        values = [
            s[2]
            for b in [body, *procs.values()]
            for s in statements(b)
            if s[0] == "wr" and isinstance(s[1], K) and s[1].v == addr
        ]
        self.names = watch.union(*map(names, values))
        self.spans = [r for v in values for r in reads(v)]

    def stored(self, s) -> bool:
        """Whether statement ``s`` may store to the register."""
        if id(s) not in self.stores:
            self.stores[id(s)] = (
                s[0] == "wr"
                and isinstance(s[1], K)
                and s[1].v == self.addr
                or s[0] == "call"
                and any(map(self.stored, self.procs[s[1]]))
                or any(self.stored(t) for arm in branches(s) for t in arm)
            )
        return self.stores[id(s)]

    @staticmethod
    def put(out: dict, key, count: int) -> None:
        """Record that ``key`` is reached after ``count`` edges, if the most so far."""
        out[key] = max(out.get(key, 0), count)

    def store(self, state, value) -> tuple:
        """``(state after storing value, whether it ends a segment)``."""
        known, bit = bits(value)
        edge = state is None or not (
            state[2] == value
            or (
                state[0] & known & EDGE_BITS == EDGE_BITS
                and not (state[1] ^ bit) & EDGE_BITS
            )
        )
        kept = None if volatile(value) else value
        return (known & EDGE_BITS, bit & EDGE_BITS, kept), edge

    @staticmethod
    def join(a, b):
        """The state after either of two paths."""
        if a is None or b is None:
            return None
        known = a[0] & b[0] & ~(a[1] ^ b[1])
        return known, a[1] & known, a[2] if a[2] == b[2] else None

    def merged(self, paths: dict) -> dict:
        """``paths``, as one path if there are too many."""
        if len(paths) <= self.MERGE:
            return paths
        (state, flags), count = next(iter(paths.items()))
        for (other, held), more in paths.items():
            state, flags, count = (
                self.join(state, other),
                flags & held,
                max(count, more),
            )
        return {(state, flags): count}

    def run(self, stmts, paths: dict) -> dict:
        """The paths after ``stmts`` from ``paths`` (path: edges so far)."""
        for s in stmts:
            tag, out = s[0], {}
            if tag == "wr" and isinstance(s[1], K) and s[1].v == self.addr:
                for (state, flags), count in paths.items():
                    state, edge = self.store(state, s[2])
                    self.put(out, (state, flags), count + edge)
            elif tag in ("if", "loop", "fire", "call") and not self.stored(s):
                out = self.pass_by(s, paths)
            elif tag == "if":
                arms: tuple = ({}, {})
                for key, count in paths.items():
                    value = truth(s[1], key[1])
                    if value is not False:
                        self.put(arms[0], key, count)
                    if value is not True:
                        self.put(arms[1], key, count)
                for arm, held in zip(s[2:4], arms):
                    for k, n in (self.run(arm, held) if held else {}).items():
                        self.put(out, k, n)
            elif tag == "loop":
                out = self.loop(s, paths)
            elif tag in ("fire", "call"):
                out = self.run(s[2] if tag == "fire" else self.procs[s[1]], paths)
            else:
                out = self.pass_by(s, paths)
            paths = self.merged(out)
        return paths

    def pass_by(self, s, paths: dict) -> dict:
        """The paths after a statement that makes no edge: it drops what it may change."""
        eff, out = self.effects(s), {}
        if (
            eff.sets.isdisjoint(self.names)
            and not _overlap(eff.stores, self.spans)
            and not (s[0] == "set" and s[1] in self.watch)
        ):
            return paths
        for (state, flags), count in paths.items():
            if (
                state is not None
                and state[2] is not None
                and (
                    names(state[2]) & eff.sets or _overlap(eff.stores, reads(state[2]))
                )
            ):
                state = (state[0], state[1], None)
            held = dict(flags)
            for name in eff.sets:
                held.pop(name, None)
            if s[0] == "set" and s[1] in self.watch and isinstance(s[2], K):
                held[s[1]] = bool(s[2].v)
            self.put(out, (state, frozenset(held.items())), count)
        return out

    def loop(self, s, paths: dict) -> dict:
        """The paths after the loop ``s``: those leaving before it runs, after its first
        pass, and after any later pass, which starts from the join of every state a
        pass can leave it in."""
        cond, bound = s[1], s[3] or 1 << 16

        def split(found: dict) -> tuple:
            go, stop = {}, {}
            for key, count in found.items():
                value = truth(cond, key[1])
                if value is not False:
                    self.put(go, key, count)
                if value is not True:
                    self.put(stop, key, count)
            return go, stop

        def gather(found: dict) -> tuple:
            keys = list(found)
            state, flags = keys[0]
            for other, held in keys[1:]:
                state, flags = self.join(state, other), flags & held
            return state, flags

        go, out = split(paths)
        if not go:
            return out
        go, stop = split(self.run(s[2], go))
        out.update((k, max(out.get(k, 0), n)) for k, n in stop.items())
        if bound < 2:
            go, out = {}, {**out, **go}
        if not go:
            return out
        base, general = max(go.values()), gather(go)
        while True:
            more, stop = split(self.run(s[2], {general: 0}))
            wider = gather({**more, general: 0})
            if wider == general:
                break
            general = wider
        step = max(more.values(), default=0)
        for key, count in stop.items():
            self.put(out, key, base + count + step * (bound - 2))
        return out


def peak(stmts, addr: int, procs=None) -> int:
    """The most stores to constant address ``addr`` a path through ``stmts`` makes (a
    loop's body counts its bound times)."""
    procs = {} if procs is None else procs
    total = 0
    for s in stmts:
        if s[0] == "wr":
            total += isinstance(s[1], K) and s[1].v == addr
        elif s[0] == "if":
            total += max(peak(s[2], addr, procs), peak(s[3], addr, procs))
        elif s[0] == "loop":
            total += peak(s[2], addr, procs) * (s[3] or 1 << 16)
        elif s[0] == "call":
            total += peak(procs[s[1]], addr, procs)
    return total


def edges(body, addr: int, procs=None) -> int:
    """The most gate and test edges (:class:`_Edges`) a path through ``body`` may make
    storing to the constant address ``addr``."""
    procs = {} if procs is None else procs
    watch = frozenset(
        n
        for b in [body, *procs.values()]
        for s in statements(b)
        if s[0] == "loop"
        for n in names(s[1])
    )
    start = {(None, frozenset()): 0}
    return max(_Edges(addr, body, procs, watch).run(body, start).values(), default=0)


def native_regs(
    body, slots: Sequence[Slot], base: int = SID, procs=None
) -> FrozenSet[int]:
    """The registers of each segmented voice whose ctrl a path may edge as often as it
    has segments: their writes stay in place, in the program's order."""
    return frozenset(
        7 * v + o
        for v, n in segments(slots).items()
        if edges(body, base + 7 * v + 4, procs) >= n
        for o in SEGMENTED
    )


def live_slots(slots: Sequence[Slot], native: FrozenSet[int]) -> Tuple[Slot, ...]:
    """``slots`` without the ``native`` registers'."""
    return tuple(s for s in slots if s[0] not in native)


def _edge(ctrl: int, value: V):
    """Whether storing ``value`` to register ``ctrl`` ends a segment."""
    moved = mk("and", 8, mk("xor", 8, last_var(ctrl), value), K(EDGE_BITS, 8))
    return mk("lor", 1, mk("lnot", 1, seen_var(ctrl)), mk("ne", 1, moved, K(0, 8)))


def _segment_store(reg: int, value, n: int) -> list:
    """The assignments of a store to a segmented register: a ctrl store that ends a
    segment closes it, shifting the closed segments back."""
    ctrl = 7 * (reg // 7) + 4
    regs = [ctrl + 2, ctrl + 1, ctrl]
    temp = V(f"t_{reg}", 8)

    def move(r: int, dst: int, src: int) -> list:
        return [
            ("set", out_var((r, dst)).n, out_var((r, src))),
            ("set", flag_var((r, dst)).n, flag_var((r, src))),
        ]

    out = [("set", temp.n, value)]
    if reg == ctrl:
        close = [x for j in range(n, 2, -1) for r in regs for x in move(r, j, j - 1)]
        close += [x for r in regs[:2] for x in move(r, 2, 1)]
        close += [
            ("set", out_var((ctrl, 2)).n, temp),
            ("set", flag_var((ctrl, 2)).n, TRUE),
        ] + [("set", flag_var((r, 1)).n, FALSE) for r in regs]
        out.append(
            (
                "if",
                _edge(ctrl, temp),
                close,
                [
                    ("set", out_var((ctrl, 1)).n, temp),
                    ("set", flag_var((ctrl, 1)).n, TRUE),
                ],
            )
        )
    else:
        out += [("set", out_var((reg, 1)).n, temp), ("set", flag_var((reg, 1)).n, TRUE)]
    return out + [("set", last_var(reg).n, temp), ("set", seen_var(reg).n, TRUE)]


def instrument(
    body,
    slots: Sequence[Slot],
    base: int = SID,
    native: FrozenSet[int] = frozenset(),
    head: bool = True,
) -> list:
    """``body`` with SID stores and reads through the slots' output variables (the flags
    cleared first, if ``head``) and no SID store left but the ``native`` registers';
    a store through a :func:`split` address variable has missed every SID register."""
    deep = segments(slots)
    band = [(base, base + REGS - 1)]
    seg = segment_regs(deep) - native

    def leaf(e):
        if not isinstance(e, Rd):
            return None
        lo, hi = span(e.a)
        if lo == hi and _overlap(band, [(lo, hi)]):
            if lo - base in native:
                return None
            if lo - base in seg:
                return Op("ite", 8, (seen_var(lo - base), last_var(lo - base), e))
            slot = (lo - base, 1)
            return Op("ite", 8, (flag_var(slot), out_var(slot), e))
        if _overlap(band, [(lo, hi)]) and (lo, hi) != FULL:
            raise OrderError(f"a read at ${lo:04x}-${hi:04x} may read the SID")
        return None

    def stmt(s):
        if s[0] != "wr" or isinstance(s[1], V) and s[1].n.startswith("sa_"):
            return None
        if not _overlap(band, [span(s[1])]):
            return None
        lo, hi = span(s[1])
        if lo != hi:
            raise OrderError(f"a store at ${lo:04x}-${hi:04x} may reach the SID")
        reg = lo - base
        if reg in native:
            return [s, ("set", flag_var((NATIVE, 1)).n, TRUE)]
        if reg in seg:
            return _segment_store(reg, s[2], deep[reg // 7])
        one = (reg, 1)
        return [("set", out_var(one).n, s[2]), ("set", flag_var(one).n, TRUE)]

    live = live_slots(slots, native)
    first = [("set", out_var(s).n, K(0, 8)) for s in live]
    first += [("set", flag_var(s).n, FALSE) for s in live]
    first += [("set", seen_var(r).n, FALSE) for r in sorted(seg)]
    first += [("set", last_var(r).n, K(0, 8)) for r in sorted(seg)]
    first += [("set", flag_var((NATIVE, 1)).n, FALSE)] * bool(native)
    return first * head + rewrite(body, lambda e: rebuild(e, leaf), stmt)


def flush(slot: Slot, base: int = SID) -> tuple:
    """The statement writing ``slot``'s output if its register was stored."""
    return ("if", flag_var(slot), [("wr", K(base + slot[0], 16), out_var(slot))], [])


def marker(reg: int, deep: Dict[int, int], native: FrozenSet[int]) -> str:
    """The flag a store to ``reg`` sets."""
    if reg in native:
        return flag_var((NATIVE, 1)).n
    return (seen_var(reg) if reg in segment_regs(deep) else flag_var((reg, 1))).n


def triggers(slot: Slot, deep: Dict[int, int]) -> FrozenSet[int]:
    """The registers whose stores assign ``slot``'s variables."""
    reg, k = slot
    voice, off = divmod(reg, 7)
    if reg >= 21 or voice not in deep or off not in SEGMENTED:
        return frozenset((reg,))
    if off == 4 or k > 1:
        return frozenset((7 * voice + 4,))
    return frozenset(7 * voice + o for o in SEGMENTED)


class _Placer:
    """Flush placement over an instrumented body (see :func:`place`)."""

    def __init__(
        self, slots: Sequence[Slot], base: int, native: FrozenSet[int], procs=None
    ):
        deep = segments(slots)
        self.live = live_slots(slots, native)
        self.slots = ((NATIVE, 1),) * bool(native) + self.live
        self.base = base
        self.trigger = [
            frozenset((NATIVE,)) if s[0] == NATIVE else triggers(s, deep)
            for s in self.slots
        ]
        self.effects = Effects(procs)
        regs = [*range(REGS), *([NATIVE] if native else [])]
        self.flags = {marker(r, deep, native): r for r in regs}

    def regs(self, s) -> FrozenSet[int]:
        """The SID registers statement ``s`` may store."""
        return frozenset(self.flags[n] for n in self.effects(s).sets if n in self.flags)

    def advance(self, done: int, future: FrozenSet[int], out: list) -> int:
        """Flush slots from ``done`` while none of ``future`` stores their registers."""
        while done < len(self.slots) and not self.trigger[done] & future:
            if self.slots[done][0] != NATIVE:
                out.append(flush(self.slots[done], self.base))
            done += 1
        return done

    def early(self, s, done: int, after: FrozenSet[int]) -> bool:
        """Whether ``s`` is an ``if`` that may finish, or with an arm that may flush slot
        ``done`` before its end (``after``: the registers stored after ``s``)."""
        if s[0] != "if" or done == len(self.slots):
            return False
        return self.effects(s).finishes or any(
            arm
            and (
                not self.trigger[done] & (self.regs(arm[-1]) | after)
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


def place(
    body,
    slots: Sequence[Slot],
    base: int = SID,
    native: FrozenSet[int] = frozenset(),
    procs=None,
) -> list:
    """An instrumented ``body`` with each slot flushed once on every path, in order, at
    the earliest boundary after the last store that assigns its variables (after every
    native store), in each arm of a branch where that falls inside it."""
    return _Placer(slots, base, native, procs).seq(body, 0, frozenset())[0]


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

    def __init__(
        self,
        body,
        constant=lambda addr: None,
        proof=None,
        procs=None,
        share: FrozenSet = frozenset(),
    ):
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        procs = {} if procs is None else procs
        self.effects = Effects(procs)
        self.counts: Dict[str, int] = {}
        self.guards = 0
        self.constant = constant
        self.proof = proof
        self.share = share
        self.pending: list = []
        self.create = True
        self.procs = {pid: procs[pid] for pid in reached(body, procs)}
        every = [s for b in [body, *self.procs.values()] for s in statements(b)]
        for s in every:
            for e in stmt_exprs(s):
                for n in names(e):
                    self.counts[n] = self.counts.get(n, 0) + 1
        assigned = {s[1] for s in every if s[0] == "set"}
        self.guards = max(
            (int(n[2:]) for n in assigned if re.fullmatch(r"g_\d+", n)), default=0
        )
        self.temps = max(
            (int(n[4:]) for n in assigned if re.fullmatch(r"cse_\d+", n)), default=0
        )

    def free(self, name: str) -> bool:
        """Whether any expression may be forwarded into ``name``'s uses."""
        return staging(name) or self.counts.get(name, 0) <= 1

    def sub(self, e, env: dict):
        """``e`` with known variables and conditions replaced, and reads of fixed
        bytes by their values."""

        def leaf(n):
            fact = env.get(n)
            if fact is None and isinstance(n, V):
                fact = env.get(n.n)
            return None if fact is None else fact.expr

        def known(n):
            if isinstance(n, Rd) and isinstance(n.a, K):
                value = self.constant(n.a.v)
                return None if value is None else K(value, 8)
            return None

        while True:
            folded = rebuild(rebuild(e, leaf), known)
            if folded == e:
                return self.cache(e, env) if self.share and self.create else e
            e = folded

    def cache(self, e, env: dict):
        """``e`` with each subexpression in ``share`` that no variable holds yet assigned
        to a fresh variable first (in :attr:`pending`), and each repeat by that variable.
        """
        if isinstance(e, (K, V)):
            return e
        if e in env:
            return env[e].expr
        out = e
        if isinstance(e, Rd):
            inner = self.cache(e.a, env)
            out = e if inner == e.a else Rd(inner)
        else:
            args = tuple(self.cache(a, env) for a in e.a)
            out = e if args == e.a else Op(e.op, e.w, args)
        if e not in self.share or volatile(e):
            return out
        self.temps += 1
        var = V(f"cse_{self.temps}", width(e))
        RANGES.get()[var.n] = span(e)
        pre = dict(env) if self.proof is not None else None
        self.pending.append(("set", var.n, out))
        env[e] = _Fact(var, names(e) | {var.n}, reads(e))
        self.note("step", pre, ("set", var.n, e), ("set", var.n, out), dict(env))
        return var

    def take(self) -> list:
        """The assignments to place before the statement under way."""
        out, self.pending = self.pending, []
        return out

    @staticmethod
    def kill(env: dict, eff: Effect) -> None:
        """Drop the facts ``eff`` clobbers."""
        if not (eff.sets or eff.stores):
            return
        for key in [
            k
            for k, f in env.items()
            if (eff.sets and (k in eff.sets or not eff.sets.isdisjoint(f.uses)))
            or (eff.stores and f.spans and _overlap(eff.stores, f.spans))
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

    def note(self, *obligation) -> None:
        """Record a proof obligation (see :func:`discharge`)."""
        if self.proof is not None:
            self.proof.append(obligation)

    def seq(self, stmts, env: dict) -> list:
        """``stmts`` rewritten under the facts ``env`` (updated in place)."""
        out: list = []
        for s in stmts:
            if s[0] == "if":
                out += self.branch(s, env)
                continue
            if s[0] == "loop":
                bound = s[3]
                self.create = False
                test = self.sub(s[1], env)
                while test == TRUE and bound > 1:
                    self.note("cond", dict(env), s[1], TRUE)
                    out += self.seq(s[2], env)
                    bound -= 1
                    test = self.sub(s[1], env)
                self.note("cond", dict(env), s[1], test)
                if test == FALSE:
                    self.create = True
                    continue
                s = ("loop", s[1], s[2], bound, s[4])
                self.kill(env, self.effects(s))
                inner = dict(env)
                test = self.sub(s[1], inner)
                self.create = True
                self.note("cond", dict(inner), s[1], test)
                body = self.seq(s[2], inner)
                self.note("join", inner, None, None, dict(env))
                out.append(("loop", test, body, *s[3:]))
                continue
            new = _map_stmt(s, lambda e: self.sub(e, env))
            pre = dict(env) if self.proof is not None else None
            out += self.take()
            out.append(new)
            self.kill(env, self.effects(new))
            if new[0] == "set":
                self.learn(env, new[1], new[2])
            if new[0] != "call":
                self.note("step", pre, s, new, dict(env))
            if new[0] == "finish":
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
        head = self.take()
        self.note("cond", dict(env), s[1], cond)
        if isinstance(cond, K) and cond.v is not None:
            return head + self.seq(s[2] if cond.v else s[3], env)
        then_env, else_env = self.assume(env, cond, True), self.assume(env, cond, False)
        self.note("entry", dict(env), cond, TRUE, dict(then_env))
        self.note("entry", dict(env), cond, FALSE, dict(else_env))
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
        out = head
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
        held = test if merged else None
        self.note("join", then_env, held, TRUE, dict(env))
        self.note("join", else_env, held, FALSE, dict(env))
        if not then and not other and not volatile(cond):
            return out
        if not then:
            return out + [("if", fold("lnot", 1, (cond,)), other, [])]
        return out + [("if", cond, then, other)]


def eliminate_all(
    body, procs: Dict[str, list], live: FrozenSet[str], end: FrozenSet[str]
) -> Tuple[list, Dict[str, list]]:
    """``(body, procs)`` with dead assignments and empty branches removed (see
    :func:`eliminate`); a proc keeps what is live after any call of it, and a proc left
    empty is not called."""
    seen, after = proc_reads(procs), {}
    body = eliminate(body, live, end, seen, after)[0]
    out = {}
    for pid in reached(body, procs):
        out[pid] = eliminate(procs[pid], after.get(pid, frozenset()), end, seen, after)[
            0
        ]
    empty = {pid for pid, b in out.items() if not b}

    def drop(stmts):
        return rewrite(
            stmts,
            lambda e: e,
            lambda s: [] if s[0] == "call" and s[1] in empty else None,
        )

    return drop(body), {pid: drop(b) for pid, b in out.items() if pid not in empty}


def shared_exprs(body, procs: Dict[str, list]) -> FrozenSet:
    """The expressions of ``body`` and ``procs`` that a variable holding them makes
    smaller: those whose repeats outweigh the assignment."""
    count: Dict = {}
    for stmts in [body, *procs.values()]:
        for s in statements(stmts):
            for e in stmt_exprs(s):
                for n in walk(e):
                    if isinstance(n, (Op, Rd)) and not _leaf(n):
                        count[n] = count.get(n, 0) + 1
    size: Dict = {}

    def nodes(n) -> int:
        if n not in size:
            size[n] = 1 + sum(map(nodes, _subs(n))) if isinstance(n, (Op, Rd)) else 1
        return size[n]

    return frozenset(
        n for n, c in count.items() if (c - 1) * (nodes(n) - 1) > 2 and not volatile(n)
    )


def forward(
    body, constant=lambda addr: None, proof=None, procs=None, share=frozenset()
):
    """``body`` with values forwarded to their uses and each loop whose condition is
    known unrolled (see :class:`_Forward`); ``constant`` gives fixed bytes, and
    ``proof`` (a list) collects the obligations that justify the result. With ``procs``
    the result is ``(body, procs)``, each proc reached from ``body`` forwarded from no
    known facts and the call of one leaving only the facts it cannot change. Each
    expression in ``share`` is computed once into a variable while the same value
    stays in it."""
    run = _Forward(body, constant, proof, procs, share)
    out = run.seq(body, {})
    if procs is None:
        return out
    return out, {pid: run.seq(b, {}) for pid, b in run.procs.items()}


def proc_reads(procs: Dict[str, list]) -> Dict[str, FrozenSet[str]]:
    """The variables each proc, and the procs it calls, read."""
    out: Dict[str, FrozenSet[str]] = {}

    def of(pid: str) -> FrozenSet[str]:
        if pid not in out:
            found: set = set()
            for s in statements(procs[pid]):
                found.update(*map(names, stmt_exprs(s)))
                found.update(s[8] if s[0] == "obj" else ())
            out[pid] = frozenset(found).union(*map(of, set(calls(procs[pid]))))
        return out[pid]

    for pid in procs:
        of(pid)
    return out


def eliminate(
    stmts,
    live: FrozenSet[str],
    end: FrozenSet[str],
    seen: Optional[Dict[str, FrozenSet[str]]] = None,
    after: Optional[Dict[str, FrozenSet[str]]] = None,
) -> Tuple[list, FrozenSet]:
    """``(stmts, live before)`` with dead assignments and empty branches removed;
    ``live`` is read after ``stmts``, ``end`` when the call ends. A ``call`` reads what
    its proc does in ``seen`` (:func:`proc_reads`), and adds the variables live after it
    to ``after[proc]``."""
    out: list = []
    for s in reversed(stmts):
        tag = s[0]
        if tag == "set":
            if s[1] not in live and not volatile(s[2]):
                continue
            live = (live - {s[1]}) | names(s[2])
        elif tag == "if":
            then, a = eliminate(s[2], live, end, seen, after)
            other, b = eliminate(s[3], live, end, seen, after)
            if not then and not other and not volatile(s[1]):
                continue
            s = (
                ("if", s[1], then, other)
                if then
                else ("if", fold("lnot", 1, (s[1],)), other, [])
            )
            live = a | b | names(s[1])
        elif tag == "call":
            after[s[1]] = after.get(s[1], frozenset()) | live
            live = live | seen[s[1]]
        elif tag == "loop":
            inner = live | names(s[1])
            while True:
                body, entry = eliminate(s[2], inner, end, seen, after)
                if entry <= inner:
                    break
                inner = inner | entry
            s = ("loop", s[1], body, s[3], s[4])
            live = inner
        elif tag == "finish":
            live = end
        elif tag == "obj":
            live = (live - set(s[3:4])).union(*map(names, s[4]), s[8])
        else:
            live = live.union(*map(names, stmt_exprs(s)))
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

    ``slots`` are the slots flushed, ``native`` the registers left in place (see
    :func:`native_regs`) and ``staged`` the output variables still assigned: buffers the
    order forces.
    """

    program: Program
    slots: Tuple[Slot, ...]
    promoted: List[int]
    staged: List[str]
    before: Counts
    inlined: Counts
    after: Counts
    paths: FrozenSet[tuple] = field(default_factory=frozenset)
    native: FrozenSet[int] = frozenset()
    ranges: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    fixed: Dict[int, int] = field(default_factory=dict)
    text: str = ""
    roundtrip: Optional[str] = None
    source: Optional[tuple] = None


def reorder(
    program: Program,
    order=None,
    *,
    base: Optional[int] = None,
    observed: Iterable[Tuple[int, int]] = (),
    memory: Optional[bytes] = None,
    proof: Optional[dict] = None,
) -> Reordered:
    """``program`` rescheduled to write the kept writes of ``order`` (text or slots) in
    order; ``observed`` lists address intervals other code reads, never promoted;
    reads of declared bytes no store reaches take their values from ``memory``;
    ``proof`` (a dict) collects each stage and the obligations of :func:`prove`.

    A proc called from several sites may stay a proc (see :func:`outlined`), which keeps
    the result small; inlining every proc, which lets the constants at each call site
    simplify its body, is tried when that leaves registers ``native``, and the schedule
    with fewer of them wins."""
    order = program.outorder if order is None else order
    if order is None:
        raise OrderError("no order given and the spec declares no output order")
    base = (SID if program.sid is None else program.sid) if base is None else base
    slots = parse_order(order) if isinstance(order, str) else tuple(order)
    keep = outlined(program.body, program.procs)
    runs: List[tuple] = []
    for kept in [keep, frozenset()] if keep else [frozenset()]:
        if runs and not runs[0][0]:
            break
        stages: dict = {}
        native, finish = _attempt(program, slots, base, observed, memory, kept, stages)
        runs.append((native, finish, stages))
    runs.sort(key=lambda run: len(run[0]))
    for at, (_native, finish, stages) in enumerate(runs):
        try:
            best = finish(True)
        except OrderError:
            if at == len(runs) - 1:
                raise
            continue
        if proof is not None:
            proof.update(stages)
        return best
    raise OrderError("no schedule")


def size(program: Program) -> int:
    """The statements and expression nodes of ``program``."""
    return sum(
        1 + sum(1 for e in stmt_exprs(s) for _ in walk(e))
        for b in [program.body, *program.procs.values()]
        for s in statements(b)
    )


def _attempt(program, slots, base, observed, memory, keep, proof):
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    """``(native, finish)``: the registers left in place when the procs in ``keep`` stay
    procs, and ``finish(share)``, the :class:`Reordered` (its stages in ``proof``), with
    the repeated subexpressions computed once if ``share``."""
    nested = nest(inline(program.body, program.procs, keep))
    kept = {pid: nest(inline(program.procs[pid], program.procs, keep)) for pid in keep}
    proof["nested"], proof["nested_procs"] = nested, kept
    proof["fix"] = []
    body, procs, fixed = fix(nested, program, memory, observed, proof["fix"], kept)
    proof["fixed_body"], proof["fixed_procs"] = body, procs
    unknown = [n for n, _w, _d in program.controls] + list(program.inputs)
    known = ranges(body, unknown, procs.values())
    marks = known_bits(body, unknown, procs.values())
    token, mark = RANGES.set(known), BITS.set(marks)
    try:
        native, later = _reorder(program, body, procs, slots, base, observed, proof)
    finally:
        BITS.reset(mark)
        RANGES.reset(token)

    def finish(share: bool) -> Reordered:
        mine = dict(known)
        token, mark = RANGES.set(mine), BITS.set(marks)
        try:
            return replace(later(share, mine), fixed=fixed)
        finally:
            BITS.reset(mark)
            RANGES.reset(token)

    return native, finish


def _reorder(program, body, procs, slots, base, observed, proof):
    # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
    for s in (s for b in [body, *procs.values()] for s in statements(b)):
        if s[0] == "obj" and any(base <= a < base + REGS for a in s[6]):
            raise OrderError(f"object {s[1]} writes the SID: state it with stores")
    keep = np.zeros(0x10000, bool)
    keep[base : base + REGS] = True
    for lo, hi in observed:
        keep[lo : hi + 1] = True
    for cell in program.cells:
        if "volatile" in cell.attrs:
            keep[cell.extent()] = True
    cells = frame_local(inline(body, procs), keep)
    count = [0]
    split_body = split(promote(body, cells), base, count)
    split_procs = {
        pid: split(promote(b, cells), base, count) for pid, b in procs.items()
    }
    native = native_regs(split_body, slots, base, split_procs)
    end = frozenset(n for n, _w, _d in program.controls)
    cache: dict = {}

    def first():
        if not cache:
            live = live_slots(slots, native)
            inst = instrument(split_body, slots, base, native)
            inst_procs = {
                pid: instrument(b, slots, base, native, False)
                for pid, b in split_procs.items()
            }
            placed = place(inst, slots, base, native, inst_procs)
            cache["paths"] = flush_order(placed, live)
            if cache["paths"] != {live}:
                raise OrderError("a path flushes the slots out of order")
            proof["placed"], proof["placed_procs"] = placed, inst_procs
            trace: list = []
            cache["trace"] = trace
            cache["forwarded"] = forward(placed, proof=trace, procs=inst_procs)
            cache["common"] = shared_exprs(*cache["forwarded"])
        return cache["forwarded"], cache["trace"], cache["common"], cache["paths"]

    def later(share: bool, known: dict) -> Reordered:
        (forwarded, forwarded_procs), trace, common, paths = first()
        proof["forward"], proof["shared"], proof["shared_procs"] = [*trace], [], {}
        if share and common:
            forwarded, forwarded_procs = forward(
                forwarded, proof=proof["forward"], procs=forwarded_procs, share=common
            )
            proof["shared"], proof["shared_procs"] = forwarded, forwarded_procs
            forwarded, forwarded_procs = eliminate_all(
                forwarded, forwarded_procs, end, end
            )
            forwarded, forwarded_procs = forward(
                forwarded, proof=proof["forward"], procs=forwarded_procs
            )
        proof["forwarded"], proof["forwarded_procs"] = forwarded, forwarded_procs
        out, outs = finalize(forwarded, forwarded_procs, program)
        staged = sorted(
            {
                s[1]
                for b in [out, *outs.values()]
                for s in statements(b)
                if s[0] == "set" and staging(s[1])
            }
        )
        every = program.body + [s for p in program.procs.values() for s in p]
        return Reordered(
            replace(program, body=out, procs=outs),
            slots,
            cells,
            staged,
            counts(every, base),
            counts(inline(program.body, program.procs), base),
            counts(out + [s for b in outs.values() for s in b], base),
            paths,
            native,
            known,
        )

    return native, later


def shared(body, procs: Dict[str, list]) -> FrozenSet[str]:
    """The variables more than one of ``body`` and ``procs`` use."""
    users: Dict[str, int] = {}
    for stmts in [body, *procs.values()]:
        found: set = set()
        for s in statements(stmts):
            found.update(*map(names, stmt_exprs(s)))
            found.update(s[8] if s[0] == "obj" else ())
            found.update(s[1:2] if s[0] == "set" else s[3:4] if s[0] == "obj" else ())
        for n in found:
            users[n] = users.get(n, 0) + 1
    return frozenset(n for n, k in users.items() if k > 1 and n is not None)


def exposure(stmts, defs: FrozenSet[str], summary: dict) -> Tuple[set, FrozenSet[str]]:
    """``(read, assigned)``: the variables ``stmts`` may read before assigning them,
    given those ``defs`` assigned, and those they assign on every path; ``summary`` holds
    each proc's pair from no ``defs``."""
    read: set = set()
    for s in stmts:
        tag = s[0]
        read |= set().union(*map(names, stmt_exprs(s))) - defs
        if tag == "obj":
            read |= set(s[8]) - defs
            defs = defs | set(s[3:4]) - {None}
        elif tag == "set":
            defs = defs | {s[1]}
        elif tag == "if":
            (r1, d1), (r2, d2) = (exposure(a, defs, summary) for a in s[2:4])
            read, defs = read | r1 | r2, d1 & d2
        elif tag in ("loop", "fire"):
            inner, after = exposure(s[2], defs, summary)
            read |= inner
            defs = after if tag == "fire" else defs
        elif tag == "call":
            read |= summary[s[1]][0] - defs
            defs = defs | summary[s[1]][1]
    return read, defs


def finalize(forwarded, procs, program: Program) -> Tuple[list, Dict[str, list]]:
    """The reordered ``(body, procs)`` from the forwarded ones: dead code removed, hooks
    dropped, expressions tidied and the variables declared. A variable used by the body
    and a proc, or by two procs, is declared by the body (``v := 0``) where a path may
    read it first."""
    end = frozenset(n for n, _w, _d in program.controls)
    hook = lambda s: [] if s[0] == "hook" else None  # noqa: E731
    out, outs = eliminate_all(forwarded, procs, end, end)
    out = rewrite(out, tidy, hook)
    outs = {pid: rewrite(b, tidy, hook) for pid, b in outs.items()}
    out, outs = eliminate_all(out, outs, end, end)
    common = shared(out, outs)
    keep = [*end, *program.inputs, *common]
    summary: dict = {}
    for pid in reversed(reached(out, outs)):
        summary[pid] = exposure(outs[pid], frozenset(), summary)
    first = exposure(out, frozenset(), summary)[0] & common - end - set(program.inputs)
    wide = var_widths(out, *outs.values())
    head = [("set", n, K(0, wide[n])) for n in sorted(first)]
    return head + declared(out, keep), {
        pid: declared(b, keep) for pid, b in outs.items()
    }


def var_widths(*bodies) -> Dict[str, int]:
    """The width of each variable the statements ``bodies`` assign or read."""
    out: Dict[str, int] = {}
    for body in bodies:
        for s in statements(body):
            if s[0] == "set":
                out.setdefault(s[1], width(s[2]))
            if s[0] == "obj" and s[3]:
                out.setdefault(s[3], 1)
            for e in stmt_exprs(s):
                for n in walk(e):
                    if isinstance(n, V):
                        out[n.n] = n.w
    return out


# ---- evidence ------------------------------------------------------------------------


def _wild(e) -> list:
    return [n.a for n in walk(e) if isinstance(n, Rd) and span(n.a) == FULL]


def log_size(body, result: "Reordered", base: int) -> int:
    """The entries a path through the instrumented ``body`` may log: the live slots'
    flushes and the native stores."""
    size = len(live_slots(result.slots, result.native)) + 1
    size += sum(peak(body, base + r) for r in result.native)
    if size > 255:
        raise OrderError(f"a path may log {size} writes: use prove")
    return size


def _logged(
    body, size: int, base: int, cells: Sequence[int], head: bool = True
) -> list:
    """``body`` with each SID store appended to the ``size`` log variables instead of
    memory, and ``alias`` set where an unbounded read may hit one of ``cells``; the log
    is cleared first if ``head``."""
    regs = frozenset(range(base, base + REGS))

    def log(reg, value) -> list:
        out = [("set", "log_t", value)]
        for i in range(size):
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
            for e in stmt_exprs(s)
            for a in _wild(e)
            for c in cells
        ]
        if s[0] == "obj":
            hits += [
                Op("ule", 1, (Op("sub", 16, (K(c, 16), s[4][i])), K(extra, 16)))
                for i, extra in s[7]
                for c in cells
            ]
        return [("set", "alias", Op("lor", 1, (V("alias", 1), h))) for h in hits]

    def stmt(s):
        if s[0] == "wr" and isinstance(s[1], K) and s[1].v in regs:
            return alias(s) + log(s[1].v - base, s[2])
        if s[0] == "loop":
            return alias(s) + [("loop", s[1], s[2] + alias(s), s[3], s[4])]
        return alias(s) + [s]

    first = [("set", "alias", FALSE), ("set", "log_n", K(0, 8))]
    first += [("set", f"log_{c}{i}", K(0, 8)) for i in range(size) for c in "rv"]
    return first * head + rewrite(body, lambda e: e, stmt)


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
    flushes = [flush(s, base) for s in live_slots(result.slots, result.native)]
    reference = rewrite(
        instrument(
            split(inline(original.body, original.procs), base),
            result.slots,
            base,
            result.native,
        )
        + [("finish",)],
        lambda e: e,
        lambda s: flushes + [s] if s[0] == "finish" else None,
    )
    return equal(
        original,
        (reference, result.program.body),
        result,
        procs=({}, result.program.procs),
        base=base,
        timeout_ms=timeout_ms,
        progress=progress,
    )


def equal(
    original: Program,
    bodies: tuple,
    result: Reordered,
    *,
    procs: tuple = ({}, {}),
    base: int = SID,
    timeout_ms: int = 0,
    progress: bool = False,
) -> Optional[tuple]:
    """None if the two instrumented ``bodies`` (with their ``procs``) log the same SID writes and leave the
    same memory and controls outside the SID and ``result``'s promoted cells, under
    ``result``'s premises; else ``(what differs, z3 model or None)``."""
    cells = result.promoted
    size = log_size(bodies[0], result, base)
    memo: dict = {}
    watch = [*cells, *range(base, base + REGS)]
    a, b = (
        smt.step(
            replace(
                original,
                body=_logged(body, size, base, watch),
                procs={
                    pid: _logged(p, size, base, watch, False)
                    for pid, p in table.items()
                },
            ),
            smt.symbolic_memory("pre"),
            smt.initial_env(original, "pre_"),
        )
        for body, table in zip(bodies, procs)
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
    keys = ["log_n"] + [f"log_{c}{i}" for i in range(size) for c in "rv"]
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
    ``results`` the :class:`Reordered` of each program. The player plays each program as
    its emitted spec (:attr:`Reordered.text`) elaborated again, which must be the
    reordered program up to names. ``options`` go to :func:`reorder`;
    where the tune's own 6502 code runs around the spec, no cell is promoted.
    """
    slots = parse_order(order) if isinstance(order, str) else tuple(order)
    results: List[Reordered] = []
    player = make(None)
    grid, writes = played(player, frames, base)
    if player.entry != player._play_addr:  # pylint: disable=protected-access
        options.setdefault("observed", [(0, 0xFFFF)])

    def transform(program: Program, memory: bytes) -> Program:
        result = reorder(program, slots, base=base, memory=memory, **options)
        result.text = emit(result.program, f"{program.name}_ordered", base)
        back = elaborate(parse(result.text), {}, standard=program.standard)
        flat = pruned(result.program)
        result.roundtrip = isomorphic(
            inline(flat.body, flat.procs),
            inline(back.body, back.procs),
            (result.program.objects, back.objects),
        )
        result.source = (program, memory, options.get("observed", ()))
        results.append(result)
        return back

    new_grid, new_writes = played(make(transform), frames, base)
    for result in results:
        if result.roundtrip is not None:
            return f"emitted spec differs: {result.roundtrip}", results
    native = frozenset().union(*(r.native for r in results))
    for frame, (old, new) in enumerate(zip(writes, new_writes)):
        if kept(old, slots, native) != new:
            return (
                f"frame {frame}: writes {new}, keeps {kept(old, slots, native)}",
                results,
            )
    rows = np.flatnonzero((grid != new_grid).any(axis=1))
    return (f"frame {rows[0]}: grid differs" if len(rows) else None), results


def census(result: Reordered) -> dict:
    """What a reordered program still uses: statement and expression forms, tables,
    persistent and promoted cells, staging holds, SID registers per voice, loops,
    self-counting cells and the object reactions by kind."""
    body, every = result.program.body, list(statements(result.program.body))
    ops = {
        n.op for s in every for e in stmt_exprs(s) for n in walk(e) if isinstance(n, Op)
    }
    ops |= {
        "read"
        for s in every
        for e in stmt_exprs(s)
        for n in walk(e)
        if isinstance(n, Rd)
    }
    stores = [span(s[1]) for s in every if s[0] == "wr"]
    sid = sorted({lo - SID for lo, hi in stores if lo == hi and SID <= lo < SID + REGS})
    counters = {
        s[1]
        for s in every
        if s[0] == "wr"
        and isinstance(s[2], Op)
        and s[2].op in ("add", "sub")
        and s[2].a[0] == Rd(s[1])
        and isinstance(s[2].a[1], K)
    }
    tables = re.findall(r"^cell (\w+): u8\[(\d+)\]", result.text, re.M)
    del body
    return {
        "statements": {
            k: sum(s[0] == k for s in every) for k in sorted({s[0] for s in every})
        },
        "operators": sorted(ops),
        "tables": [name for name, _n in tables if name != "sid"],
        "persistent_cells": result.after.cells,
        "promoted_cells": len(result.promoted),
        "fixed_bytes": len(result.fixed),
        "staged": result.staged,
        "voices": sorted({r // 7 for r in sid if r < 21}),
        "song_registers": [r for r in sid if r >= 21],
        "loops": [s[3] for s in every if s[0] == "loop"],
        "counters": len(counters),
        "objects": {
            k: sum(
                result.program.objects[s[1]].kind == k for s in every if s[0] == "obj"
            )
            for k in sorted(
                {result.program.objects[s[1]].kind for s in every if s[0] == "obj"}
            )
        },
        "finish": any(s[0] == "finish" for s in every),
    }


def survey(
    path: str,
    order,
    frames: int,
    subtune: Optional[int] = None,
    prove_ms: int = 0,
    budget_s: Optional[float] = None,
    jobs: int = 1,
) -> dict:
    """:func:`compare` on a tune an installed format plays with a spec: the format, the
    verdict, and per program its staged variables, counts inlined and after, emitted
    spec, :func:`census` and, with ``prove_ms`` (a per-query timeout), whole-call proof.
    """
    from ..formats import find_format  # pylint: disable=import-outside-toplevel
    from .outproof import prove  # pylint: disable=import-outside-toplevel

    data = Path(path).read_bytes()
    fmt = find_format(data)
    model = fmt.load(data, subtune)
    verdict, results = compare(
        lambda transform: fmt.spec_player(model, transform=transform), order, frames
    )
    programs = []
    for r in results:
        entry = {
            "staged": r.staged,
            "promoted": len(r.promoted),
            "inlined": asdict(r.inlined),
            "after": asdict(r.after),
            "census": census(r),
            "text": r.text,
        }
        if prove_ms:
            program, memory, observed = r.source
            proof = prove(
                program,
                order,
                memory=memory,
                observed=observed,
                timeout_ms=prove_ms,
                budget_s=budget_s,
                jobs=jobs,
            )
            entry["proof"] = {
                "holds": proof.holds,
                "open": proof.open[:5],
                "obligations": {k: d.total for k, d in proof.passes.items()},
                "unchecked": sum(d.unchecked for d in proof.passes.values()),
                "seconds": round(proof.seconds, 1),
                "peak_mb": proof.peak_mb,
            }
        programs.append(entry)
    return {"format": fmt.name, "verdict": verdict, "programs": programs}


__all__ = [
    "ORDER",
    "Counts",
    "OrderError",
    "Reordered",
    "census",
    "check",
    "compare",
    "kept",
    "parse_order",
    "played",
    "survey",
    "reorder",
]
