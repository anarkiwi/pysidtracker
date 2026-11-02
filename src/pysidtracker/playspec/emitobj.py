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
    return f"{left} {'==' if equal else '!='} {value:#x}"


def assigns(sets: tuple, values: List[str], column) -> str:
    """Source of ``value[.part] = operand`` items."""
    parts = ("", ".lo", ".hi")
    return ", ".join(
        f"{values[i]}{parts[part]} = {operand(spec, column(spec))}"
        f"{'' if test is None else ' when ' + guard(test)}"
        for i, part, spec, test in sets
    )


def row(spec: tuple, values: List[str]) -> str:
    """Source of a cursor row ``(lo, hi, tests, action)`` as a ``match`` arm."""
    lo, hi, tests, action = spec

    def column(xf):
        return f"c{xf[0]}"

    def op(xf):
        return operand(xf, column(xf))

    span = f"{lo:#x}" if lo == hi else f"{lo:#x}..{hi:#x}"
    when = f" when {' && '.join(guard(t) for t in tests)}" if tests else ""
    kind = action[0]
    then = ()
    if kind == "sweep":
        _k, into, count, step, frac, part, then = action
        lane = ("", ".lo", ".hi")[part]
        args = [f"into={values[into]}{lane}", f"count={op(count)}", f"step={op(step)}"]
        body = f"sweep({', '.join(args + [f'frac={frac}'] * bool(frac))})"
    elif kind == "delay":
        body, then = f"hold({op(action[1])})", action[2]
    elif kind == "set":
        body = f"value({assigns(action[1], values, column)}) take {action[2]}"
        then = action[3]
    elif kind == "end":
        body, then = f"end take {action[1]}", action[2]
    elif kind == "jump":
        body = f"jump({op(action[1])})"
        if action[2]:
            body += f" hold {action[2][0]:#x}..{action[2][1]:#x}"
        if action[3] is not None:
            body += f" as {action[3]:#x}"
    elif kind == "call":
        body = f"call({op(action[1])})"
    elif kind == "run":
        body, then = f"run({action[1]:#x})", action[2]
    else:
        body = kind
    tail = f" then {assigns(then, values, column)}" if then else ""
    return f"    {span}{when} => {body}{tail};"


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
    if obj.kind == "accum":
        out = [f"{head}: {cells['cell']} {{"]
        if "dir" in cells:
            out.append(f"  dir {cells['dir']};")
        modes = {
            "ramp": "ramp",
            "to": "ramp to 0 then stop",
            "tri": "triangle",
            "flip": "flip",
        }
        wraps = f" -> {obj.fires}" if obj.fires else ""
        out += [
            f"  on {e} {modes[a[0]]}{wraps * (a[0] != 'flip')};"
            for e, a in obj.reactions.items()
        ]
        return out + ["}"]
    if obj.kind == "owner":
        out = [f"{head}: {cells['cell']} {{"]
        out += [f"  on {e} {a[0]};" for e, a in obj.reactions.items()]
        return out + ["}"]
    adv, width, columns, rows = obj.spec
    values = [cells[f"v{i}"] for i in range(len(cells)) if f"v{i}" in cells]
    values += [cells["cnt"]] if "cnt" in cells else [""]
    out = [f"{head} {{", f"  table @ {', '.join(['0'] * columns)} width {width};"]
    for role in ("pos", "cnt", "run", "ret"):
        if role in cells:
            out.append(f"  {'count' if role == 'cnt' else role} {cells[role]};")
    out += [f"  values {', '.join(values[:-1])};"] * bool(values[:-1])
    out += [f"  advance {'pre' if adv else 'post'};"]
    out += [f"  on {e} {a[0]};" for e, a in obj.reactions.items()]
    out += [f"  wrap -> {obj.fires};"] * bool(obj.fires)
    return out + ["  match c0 {"] + [row(r, values) for r in rows] + ["  }", "}"]
