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
    gated = len(action) > 2

    def step(R, W, ops):
        limit, reload = ops
        value = load(R, loc)
        if kind == "reset":
            store(W, loc, reload)
            return 0
        if gated:
            wrap = limit != 0
        else:
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


def _accum(obj: Obj, action: tuple) -> Step:
    loc, dirs = obj.locs["cell"], obj.locs.get("dir")
    top = (1 << 8 * len(loc)) - 1
    mode = action[0]

    def step(R, W, ops):
        amount, target = ops
        value = load(R, loc)
        if mode == "flip":
            W(dirs[0], R(dirs[0]) ^ 1)
            return 0
        if mode == "to":
            if value == target:
                return 0
            moved = (
                min(value + amount, target)
                if value < target
                else max(value - amount, target)
            )
            store(W, loc, moved)
            return int(moved == target)
        down = mode == "tri" and R(dirs[0]) & 1
        after = value - amount if down else value + amount
        store(W, loc, after & top)
        return int(after < 0 or after > top)

    return step


def to_signed(byte: int) -> int:
    """The signed value of ``byte``."""
    return byte - 256 * (byte >> 7)


def passes(test: tuple, cols) -> bool:
    """Whether a row-byte test ``(column, mask, value, equal)`` holds of ``cols``."""
    col, mask, value, equal = test
    return ((cols[col] & mask) == value) == bool(equal)


def _sweep(R, W, loc, frac: int, delta: int) -> int:
    if not frac:
        carry = int(R(loc[0]) + (delta & 0xFF) > 0xFF)
        store(W, loc, load(R, loc) + delta)
        return carry
    low = (1 << frac) - 1
    total = ((R(loc[1]) << frac) | (R(loc[0]) & low)) + delta
    total &= (1 << 8 + frac) - 1
    W(loc[1], total >> frac)
    W(loc[0], total & low)
    return 0


def _cursor(obj: Obj, action: tuple) -> Step:
    adv, width, columns, rows = obj.spec
    locs = obj.locs
    pos, cnt, run, ret = (locs.get(k) for k in ("pos", "cnt", "run", "ret"))
    values = [locs[f"v{i}"] for i in range(len(locs)) if f"v{i}" in locs]
    room = (1 << 8 * len(pos)) - 1
    empty = next((r[3][2] for r in rows if r[3][0] == "run"), ())

    def assign(W, cols, sets, left):
        for index, part, spec, guard in sets:
            if guard and not passes(guard, cols):
                continue
            if index == len(values):
                left[0] = xf(spec, cols)
            elif part:
                W(values[index][part - 1], xf(spec, cols))
            else:
                store(W, values[index], xf(spec, cols))

    def settle(R, W, ops):
        del ops
        if not (run and R(run[0])):
            store(W, pos, load(R, pos) + 1 & room)
        return 0

    def reset(R, W, ops):
        del R
        store(W, pos, ops[columns] - (adv == 1) & room)
        for cell in (cnt, run):
            if cell:
                W(cell[0], 0)
        return 0

    def walk(R, W, ops):
        at, left = load(R, pos), [R(cnt[0]) if cnt else 0]
        pending = R(run[0]) if run else 0
        if pending:
            W(run[0], pending - 1)
            assign(W, [0] * columns, empty, left)
            return 0
        at = at + (adv == 1) & room
        carry, via, jumped, used = 0, None, False, 0

        def rest():
            return at + used - (adv > 0) & room

        for _ in range(JUMPS):
            cols = [R((ops[j] + at) & 0xFFFF) for j in range(columns)]
            first = cols[0] if via is None else via
            row = next(
                (
                    r[3]
                    for r in rows
                    if r[0] <= first <= r[1] and all(passes(t, cols) for t in r[2])
                ),
                ("stop",),
            )
            kind = row[0]
            if kind in ("jump", "call", "return"):
                target = load(R, ret) if kind == "return" else xf(row[1], cols)
                if kind == "call":
                    store(W, ret, at + width & room)
                carry, jumped = int(target >= at), True
                if kind == "jump" and target == at:
                    break
                at, via = target, row[3] if kind == "jump" else None
                hold = row[2] if kind == "jump" else None
                if hold and hold[0] <= R((ops[0] + at) & 0xFFFF) <= hold[1]:
                    left[0] = 0
                    break
                continue
            if kind == "stop":
                carry = 1
                break
            used = width
            if kind in ("sweep", "delay"):
                limit = xf(row[2] if kind == "sweep" else row[1], cols)
                if left[0] != limit:
                    left[0], used = (left[0] + 1) & 0xFF, 0
                    carry = 0
                    if kind == "sweep":
                        delta = to_signed(xf(row[3], cols))
                        carry = _sweep(R, W, values[row[1]], row[4], delta)
                    break
                left[0], carry = 0, 1
            elif kind == "set":
                assign(W, cols, row[1], left)
                carry, used = carry if jumped else 0, row[2]
            elif kind == "run":
                W(run[0], (cols[0] - row[1]) & 0xFF)
                assign(W, [0] * columns, row[2], left)
                carry, used = 0, 1
            else:
                carry, used = 1, row[1]
            if kind != "run":
                assign(W, cols, row[-1], left)
            break
        else:
            raise SpecRuntimeError("cursor jumps forever")
        store(W, pos, rest())
        if cnt:
            W(cnt[0], left[0])
        return carry

    return {"reset": reset, "settle": settle}.get(action[0], walk)


_KINDS: Dict[str, Callable[[Obj, tuple], Step]] = {
    "counter": _counter,
    "owner": _owner,
    "accum": _accum,
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
