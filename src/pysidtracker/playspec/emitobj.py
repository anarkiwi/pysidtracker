"""Playspec source of bound event-driven objects (the declarations :func:`~.emit.emit` writes)."""

from __future__ import annotations

from typing import Dict, List

from .ir import Obj


def operand(spec: tuple, column: str) -> str:
    """Source of the operand ``(column, mask, shl, shr)`` over the byte named ``column``."""
    index, mask, shl, shr = spec
    if index < 0:
        return f"{mask:#x}"
    text = column if mask == 0xFF else f"({column} & {mask:#x})"
    if shl:
        text = f"({text} << {shl})"
    if shr:
        text = f"({text} >> {shr})"
    return text


def cell(name: str, loc: tuple) -> str:
    """The declaration of a cell at the byte addresses ``loc``."""
    if len(loc) == 1:
        return f"cell {name}: u8 @ {loc[0]:#06x};"
    return f"cell {name}: u16 @ ({loc[0]:#06x}, {loc[1]:#06x});"


def guard(test: tuple) -> str:
    """Source of a row byte test ``(column, mask, value, equal)``."""
    col, mask, value, equal = test
    left = f"c{col}" if mask == 0xFF else f"(c{col} & {mask:#x})"
    return f" when {left} {'==' if equal else '!='} {value:#x}"


def assigns(sets: tuple, values: List[str], column) -> str:
    """Source of ``value[.part] = operand`` items."""
    parts = ("", ".lo", ".hi")
    return ", ".join(
        f"{values[i]}{parts[part]} = {operand(spec, column(spec))}"
        f"{'' if test is None else guard(test)}"
        for i, part, spec, test in sets
    )


def row(spec: tuple, values: List[str]) -> str:
    """Source of a walker row ``(lo, hi, action)``."""
    lo, hi, action = spec

    def column(xf):
        return f"c{xf[0]}"

    span = f"{lo:#x}" if lo == hi else f"{lo:#x}..{hi:#x}"
    kind = action[0]
    then = ""
    if kind == "sweep":
        _k, into, count, step, frac, after = action
        args = [
            f"into={values[into]}",
            f"count={operand(count, column(count))}",
            f"step={operand(step, column(step))}",
        ] + [f"frac={frac}"] * bool(frac)
        body, then = f"sweep({', '.join(args)})", after
    elif kind == "delay":
        body, then = f"delay(count={operand(action[1], column(action[1]))})", action[2]
    elif kind == "set":
        body, then = f"set({assigns(action[1], values, column)})", action[2]
    elif kind == "jump":
        body = f"jump({operand(action[1], column(action[1]))})"
        if action[2]:
            body += f" hold {action[2][0]:#x}..{action[2][1]:#x}"
        if action[3] is not None:
            body += f" as {action[3]:#x}"
    else:
        body = "stop"
    tail = f" then {assigns(then, values, column)}" if then else ""
    return f"  row {span}: {body}{tail};"


def condition(cond: tuple, prev: str) -> str:
    """Source of a field's presence test on the byte named ``prev``."""
    mask, value, equal = cond
    left = prev if mask == 0xFF else f"({prev} & {mask:#x})"
    return f"{left} {'==' if equal else '!='} {value:#x}"


def declare(obj: Obj, cells: Dict[str, str], name: str) -> List[str]:
    """The lines declaring ``obj`` as ``name``; ``cells`` maps each loc role to its cell."""
    head = f"{obj.kind} {name}"
    if obj.kind == "counter":
        out = [f"{head}: {cells['cell']} {{"]
        for event, action in obj.reactions.items():
            amount = (
                action[0] == "reset"
                and "reset"
                or (f"{'+=' if action[0] == 'up' else '-='} {action[1]}")
            )
            out.append(f"  on {event} {amount};")
        gated = any(len(a) > 2 for a in obj.reactions.values())
        if gated or obj.fires:
            out.append(
                f"  wrap{' when 0' * gated}{f' -> {obj.fires}' * bool(obj.fires)};"
            )
        return out + ["}"]
    if obj.kind == "owner":
        out = [f"{head}: {cells['cell']} {{"]
        out += [f"  on {e} {a[0]};" for e, a in obj.reactions.items()]
        return out + ["}"]
    if obj.kind == "walker":
        width, columns, rows = obj.spec
        values = [cells[f"v{i}"] for i in range(len(cells) - 2)] + [cells["cnt"]]
        bases = ", ".join(["0"] * columns)
        out = [f"{head} {{", f"  table @ {bases} width {width};"]
        out += [f"  pos {cells['pos']};", f"  count {cells['cnt']};"]
        out += [f"  values {', '.join(values[:-1])};"]
        out += [f"  on {e} {a[0]};" for e, a in obj.reactions.items()]
        return out + [row(r, values) for r in rows] + ["}"]
    run, fields, last = obj.spec
    out = [f"{head} @ 0 pos {cells['pos']}{' last' * last} {{"]
    out += [f"  on {e} {a[0]};" for e, a in obj.reactions.items()]
    if run is not None:
        out.append(
            f"  run {run[0]:#x}..{run[1]:#x} base {run[2]:#x} -> {cells['run']};"
        )
    for k, (cond, stores) in enumerate(fields):
        puts = []
        for j, (keep, shl, shr, test) in enumerate(stores):
            put = f"{cells[f'o{k}_{j}']} = {operand((0, keep, shl, shr), f'f{k}')}"
            if test != (0, 0, True):
                put += f" when {condition(test, f'f{k}')}"
            puts.append(put)
        tail = f" -> {', '.join(puts)}" if puts else ""
        when = f" when {condition(cond, f'f{k - 1}')}" if k else ""
        out.append(f"  field f{k}{when}{tail};")
    return out + ["}"]
