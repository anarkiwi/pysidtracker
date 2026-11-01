"""The executable semantics of event-driven objects, the oracle of the native core.

A reaction is ``step(R, W, operands) -> int``: ``R`` and ``W`` read and store a byte and the
result is the wrap flag. docs/playspec.md, "Events and objects", states each kind.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from ..errors import SidError
from .ir import Obj, Program

JUMPS = 256

Step = Callable[[Callable, Callable, tuple], int]


class SpecRuntimeError(SidError):
    """A spec assertion failed, a bounded loop ran past its bound or a walker jumped forever."""


def load(R, loc) -> int:
    """The little-endian value of the byte addresses ``loc``."""
    return R(loc[0]) | (R(loc[1]) << 8 if len(loc) > 1 else 0)


def store(W, loc, value: int) -> None:
    """``value`` stored to ``loc``, kept to its width."""
    W(loc[0], value & 0xFF)
    if len(loc) > 1:
        W(loc[1], value >> 8 & 0xFF)


def xf(spec: tuple, cols) -> int:
    """``spec`` = ``(column, mask, shl, shr)`` applied to a row's bytes: the constant
    ``mask`` for column -1, else ``(((byte & mask) << shl) >> shr) & 0xFF``."""
    col, mask, shl, shr = spec
    return mask if col < 0 else ((cols[col] & mask) << shl >> shr) & 0xFF


def _counter(obj: Obj, action: tuple) -> Step:
    loc = obj.locs["cell"]
    top = (1 << 8 * len(loc)) - 1
    kind, delta = action[0], action[1] if len(action) > 1 else 0

    def step(R, W, ops):
        limit, reload = ops
        value = load(R, loc)
        if kind == "reset":
            store(W, loc, reload)
            return 0
        wrap = value + delta > limit if kind == "up" else value < delta
        after = value + delta if kind == "up" else value - delta
        store(W, loc, (reload if wrap else after) & top)
        return int(wrap)

    return step


def _owner(obj: Obj, action: tuple) -> Step:
    loc = obj.locs["cell"]

    def step(R, W, ops):
        ident, none, cond = ops
        if cond and action[0] == "claim":
            store(W, loc, ident)
        elif cond and load(R, loc) == ident:
            store(W, loc, none)
        return 0

    return step


def _walker(obj: Obj, action: tuple) -> Step:
    width, columns, rows = obj.spec
    pos, cnt = obj.locs["pos"], obj.locs["cnt"]
    values = [obj.locs[f"v{i}"] for i in range(len(obj.locs) - 2)]
    room = (1 << 8 * len(pos)) - 1

    def assign(W, cols, sets):
        for index, part, spec in sets:
            if part:
                W(values[index][part - 1], xf(spec, cols))
            else:
                store(W, values[index], xf(spec, cols))

    def sweep(R, W, into, frac, delta):
        loc = values[into]
        if not frac:
            store(W, loc, load(R, loc) + delta)
            return
        low = (1 << frac) - 1
        total = ((R(loc[1]) << frac) | (R(loc[0]) & low)) + delta
        total &= (1 << 8 + frac) - 1
        W(loc[1], total >> frac)
        W(loc[0], total & low)

    def reset(R, W, ops):
        del R
        store(W, pos, ops[columns] & room)
        W(cnt[0], 0)
        return 0

    def walk(R, W, ops):
        at, count = load(R, pos), R(cnt[0])
        for _ in range(JUMPS):
            cols = [R((ops[j] + at) & 0xFFFF) for j in range(columns)]
            row = next((r[2] for r in rows if r[0] <= cols[0] <= r[1]), ("stop",))
            kind = row[0]
            if kind == "jump":
                target = xf(row[1], cols)
                if target == at:
                    break
                at = target
                continue
            if kind in ("sweep", "delay"):
                if count != xf(row[2] if kind == "sweep" else row[1], cols):
                    count = (count + 1) & 0xFF
                    if kind == "sweep":
                        delta = xf(row[3], cols)
                        sweep(R, W, row[1], row[4], delta - 256 * (delta >> 7))
                    break
                count = 0
            if kind == "set":
                assign(W, cols, row[1])
            if kind != "stop":
                at = (at + width) & room
                assign(W, cols, row[-1])
            break
        else:
            raise SpecRuntimeError("walker jumps forever")
        store(W, pos, at)
        W(cnt[0], count)
        return 0

    return reset if action[0] == "reset" else walk


def _cursor(obj: Obj, action: tuple) -> Step:
    run, fields = obj.spec
    pos, pending = obj.locs["pos"], obj.locs.get("run")

    def reset(R, W, ops):
        del R
        W(pos[0], ops[1] & 0xFF)
        if pending:
            W(pending[0], 0)
        return 0

    def read(R, W, ops):
        at, used, left = R(pos[0]), 0, R(pending[0]) if pending else 0
        first, empty = 0, bool(left)
        if empty:
            W(pending[0], left - 1)
        else:
            first, used = R((ops[0] + at) & 0xFFFF), 1
            if run and run[0] <= first <= run[1]:
                W(pending[0], first - run[2])
                first, empty = 0, True
        row = [first]
        while not empty and len(row) < len(fields):
            mask, want, equal = fields[len(row)][0]
            if ((row[-1] & mask) == want) != equal:
                break
            row.append(R((ops[0] + at + used) & 0xFFFF))
            used += 1
        for byte, (_cond, out, mask, shl, shr) in zip(row, fields):
            if out is not None:
                store(W, obj.locs[f"o{out}"], ((byte & mask) << shl >> shr) & 0xFF)
        W(pos[0], (at + used) & 0xFF)
        return 0

    return reset if action[0] == "reset" else read


_KINDS: Dict[str, Callable[[Obj, tuple], Step]] = {
    "counter": _counter,
    "owner": _owner,
    "walker": _walker,
    "cursor": _cursor,
}


def step_of(obj: Obj, event: str) -> Step:
    """The step function of ``obj``'s reaction to ``event``."""
    return _KINDS[obj.kind](obj, obj.reactions[event])


def steps(program: Program) -> Dict[Tuple[str, str], Step]:
    """The step function of every reaction of every object of ``program``."""
    return {
        (obj.name, event): step_of(obj, event)
        for obj in program.objects.values()
        for event in obj.reactions
    }


__all__ = ["JUMPS", "SpecRuntimeError", "load", "step_of", "steps", "store", "xf"]
