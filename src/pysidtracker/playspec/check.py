"""Static checks of a spec as written, every build's branches included: a ``let``, a parameter
or a ``state`` nothing reads is an error."""

from __future__ import annotations

from typing import Iterator, List, Tuple

from .syntax import Spec


def _reads(node) -> Iterator[str]:
    """The names ``node`` reads: every name but the target of a plain assignment."""
    if isinstance(node, list):
        for item in node:
            yield from _reads(item)
    elif isinstance(node, tuple) and node and not isinstance(node[0], str):
        for item in node:
            yield from _reads(item)
    elif isinstance(node, tuple) and node:
        if node[0] == "name":
            yield node[2]
        elif node[0] in ("let", "own"):
            yield from _reads(node[4])
        elif node[0] == "proc":
            yield from _reads(node[-1])
        elif node[0] == "assign":
            lval = node[2]
            if not (lval[0] == "name" and node[3] == ":="):
                yield from _reads(lval)
            yield from _reads(node[4])
        else:
            for item in node[1:]:
                yield from _reads(item)
    elif isinstance(node, str):
        yield node


def _lets(body) -> Iterator[tuple]:
    """The ``let`` statements anywhere in ``body``."""
    if isinstance(body, list):
        for item in body:
            yield from _lets(item)
    elif isinstance(body, tuple) and body:
        if body[0] == "let":
            yield body
        for item in body[1:] if isinstance(body[0], str) else body:
            yield from _lets(item)


def unread(spec: Spec) -> List[Tuple[int, str]]:
    """``(line, message)`` of each let, parameter and state that nothing reads."""
    errors = []
    bodies = [*spec.procs.values()]
    for machine in spec.machines.values():
        bodies += [item for item in machine[6] if item[0] in ("proc", "handler")]
    for item in bodies:
        body = item[-1]
        read = set(_reads(body))
        for let in _lets(body):
            if let[2] not in read:
                errors.append((let[1], f"let {let[2]} is never read"))
        if item[0] == "proc":
            for name, _typ in item[3]:
                if name not in read:
                    errors.append(
                        (item[1], f"parameter {name} of {item[2]} is never read")
                    )
    states = [*spec.states]
    for machine in spec.machines.values():
        states += [item for item in machine[6] if item[0] == "own"]
    everything = set(
        _reads([*spec.machines.values(), *spec.procs.values(), *spec.objects])
    )
    everything |= set(_reads([*spec.tables, *spec.views, *spec.outputs, spec.schedule]))
    for item in states:
        if item[2] not in everything:
            errors.append((item[1], f"state {item[2]} is never read"))
    return errors
