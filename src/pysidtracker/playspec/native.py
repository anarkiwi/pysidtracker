"""The native call machine: concrete execution of lowered programs.

:func:`encode` flattens a :class:`~.ir.Program` to the ``(op, arg)`` instruction stream the Rust
interpreter runs, with the semantics of :mod:`.pyexec`. ``PYSIDTRACKER_SPEC_NATIVE=0``
disables the machine; :class:`~.player.SpecPlayer` then runs :func:`~.pyexec.compile_program`.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
from array import array
from typing import Dict, List, Tuple

from ..notes import table
from .ir import K, Obj, Op, Own, Program, Rd, V, width

try:
    _core = importlib.import_module("pysidtracker._specvm")
except ImportError:  # pragma: no cover - the Python executor stands in
    _core = None  # pylint: disable=invalid-name

ENV = "PYSIDTRACKER_SPEC_NATIVE"
OWN = 0x10000
REGISTER = {name: i for i, name in enumerate("axyczidvn")}
Core = _core.Core if _core else None
OPS: Dict[str, int] = (
    {name: i for i, name in enumerate(_core.opcodes())} if _core else {}
)
_MASKED = {
    "add": "Add",
    "sub": "Sub",
    "mul": "Mul",
    "shl": "Shl",
    "not": "Not",
    "neg": "Neg",
}
_PLAIN = {
    "and": "And",
    "or": "Or",
    "xor": "Xor",
    "shr": "Shr",
    "eq": "Eq",
    "ne": "Ne",
    "ult": "Ult",
    "ule": "Ule",
    "word": "Word",
    "own": "Eq",
    "lo": "Lo",
    "hi": "Hi",
    "lnot": "Lnot",
    "sext": "Sext",
}
_SIGNED = {"slt": "Slt", "sle": "Sle"}
_COMMUTES = frozenset({"add", "mul", "and", "or", "xor", "eq", "ne"})
_IMMEDIATE = {
    "and": "AndK",
    "or": "OrK",
    "xor": "XorK",
    "shr": "ShrK",
    "eq": "EqK",
    "ne": "NeK",
    "ult": "UltK",
    "ule": "UleK",
}
_PACKED = {"add": "AddK", "sub": "SubK", "shl": "ShlK"}
_JUMPS = {
    ("eq", True): "JEq",
    ("eq", False): "JNe",
    ("ne", True): "JNe",
    ("ne", False): "JEq",
    ("ult", True): "JLt",
    ("ult", False): "JGe",
    ("ule", True): "JLe",
    ("ule", False): "JGt",
}
TARGET = 0xFFFFFF
I32 = range(-(1 << 31), 1 << 31)
I39 = range(-(1 << 38), 1 << 38)
SETK_VARS = 1 << 20
I43 = range(-(1 << 43), 1 << 43)


class Unsupported(Exception):
    """A program the native machine does not run."""


def _loc(loc) -> List[int]:
    """The flat form of byte addresses ``loc``."""
    return [loc[0], loc[1] if len(loc) > 1 else -1]


def _xf(spec) -> List[int]:
    return list(spec)


def _assigns(sets) -> List[int]:
    out = [len(sets)]
    for index, part, spec, guard in sets:
        out += [index, part, *_xf(spec), *(guard or (-1, 0, 0, 1))]
    return out


KINDS = ("stop", "jump", "sweep", "delay", "set", "end", "call", "return", "run")


def _row(row) -> List[int]:
    """The flat form of a cursor row ``(lo, hi, tests, action)``."""
    lo, hi, tests, action = row
    kind = action[0]
    blank = (0, 0, 0, 0)
    into = frac = base = part = 0
    take = 1
    first = second = blank
    sets: tuple = ()
    then: tuple = ()
    hold, via = (1, 0), -1
    if kind == "jump":
        first = action[1]
        hold = action[2] or hold
        via = -1 if action[3] is None else action[3]
    elif kind == "sweep":
        into, first, second, frac, part, then = action[1:]
    elif kind == "delay":
        first, then = action[1:]
    elif kind == "set":
        sets, take, then = action[1:]
    elif kind == "end":
        take, then = action[1:]
    elif kind == "call":
        first = action[1]
    elif kind == "run":
        base, sets = action[1:]
    return [
        lo,
        hi,
        len(tests),
        *[x for t in tests for x in (*t[:3], int(t[3]))],
        KINDS.index(kind),
        take,
        into,
        frac,
        base,
        part,
        *_xf(first),
        *_xf(second),
        *_assigns(sets),
        *_assigns(then),
        *hold,
        via,
    ]


def describe(obj: Obj, event: str, out: int, slot=None) -> List[int]:
    """The flat description of ``obj``'s reaction to ``event`` for the native core; ``slot``
    maps a ``state`` variable to its slot, its byte :class:`~.ir.Own` to ``OWN + 2 * slot +
    lane``."""
    if slot is not None:
        lanes = {
            role: tuple(
                OWN + 2 * slot(a.n) + a.lane if isinstance(a, Own) else a for a in loc
            )
            for role, loc in obj.locs.items()
        }
        obj = dataclasses.replace(obj, locs=lanes)
    action = obj.reactions[event]
    head = [
        {"counter": 0, "owner": 1, "cursor": 2, "accum": 3}[obj.kind],
        out,
        obj.arity(),
    ]
    if obj.kind == "counter":
        code = {"reset": 0, "up": 1, "down": 2}[action[0]] + 2 * (len(action) > 2)
        return head + [
            *_loc(obj.locs["cell"]),
            code,
            action[1] if len(action) > 1 else 0,
        ]
    if obj.kind == "owner":
        return head + [*_loc(obj.locs["cell"]), int(action[0] == "release")]
    if obj.kind == "accum":
        dirs = obj.locs.get("dir", (-1, -1))
        mode = ("ramp", "to", "tri", "flip").index(action[0])
        return head + [*_loc(obj.locs["cell"]), mode, *_loc(dirs)]
    adv, width, columns, rows = obj.spec
    locs = obj.locs
    values = [locs[f"v{i}"] for i in range(len(locs)) if f"v{i}" in locs]
    out_ = head + [adv, width, columns, ("step", "reset", "settle").index(action[0])]
    for role in ("pos", "cnt", "run", "ret"):
        out_ += _loc(locs.get(role, (-1, -1)))
    out_ += [len(values)] + [x for v in values for x in _loc(v)]
    out_ += [len(rows)] + [x for r in rows for x in _row(r)]
    for tab in obj.tabs or (None,) * columns:
        out_ += [-1] if tab is None else [len(tab), *tab]
    return out_


def available() -> bool:
    """Whether the native core is built and not disabled by ``PYSIDTRACKER_SPEC_NATIVE=0``."""
    return _core is not None and os.environ.get(ENV, "1") != "0"


class _Encoder:
    """One program's instruction stream."""

    def __init__(self, program: Program):
        self.code: List[int] = []
        self.program = program
        self.vars: Dict[str, int] = {
            n: i for i, (n, _w, _d) in enumerate(program.controls)
        }
        self.tables: Dict[tuple, int] = {}
        self.calls: List[Tuple[int, str]] = []
        self.objects: List[List[int]] = []

    def emit(self, op: str, arg: int = 0) -> int:
        """Append an instruction; its index."""
        self.code += (OPS[op], arg)
        return len(self.code) // 2 - 1

    def here(self) -> int:
        """The index of the next instruction."""
        return len(self.code) // 2

    def patch(self, at: int, target: int) -> None:
        """Make the jump at ``at`` go to ``target``."""
        self.code[2 * at + 1] = self.code[2 * at + 1] & ~TARGET | target

    def var(self, name: str) -> int:
        """The variable slot of ``name``: controls first, then locals as met."""
        return self.vars.setdefault(name, len(self.vars))

    def expr(self, e) -> None:
        """Code that pushes the value of ``e``."""
        if isinstance(e, K):
            if e.v is None:
                raise Unsupported("a program lowered without a binding")
            self.emit("Konst", e.v)
        elif isinstance(e, Rd):
            if isinstance(e.a, K) and e.a.v is not None and 0 <= e.a.v <= 0xFFFF:
                self.emit("RdK", e.a.v)
            else:
                self.expr(e.a)
                self.emit("Rd")
        elif isinstance(e, V):
            self.emit("Var", self.var(e.n))
        else:
            self.operator(e)

    def operator(self, e: Op) -> None:
        """Code that pushes the value of the operator ``e``."""
        name = e.op
        if name == "note":
            key = (*(k.v for k in e.a[:3]), 1 << max(8, width(e.a[3])))
            self.expr(e.a[3])
            self.emit("Note", self.tables.setdefault(key, len(self.tables)))
            return
        if name == "lut":
            self.expr(e.a[1])
            self.emit(
                "Note", self.tables.setdefault(("lut", e.a[0].v), len(self.tables))
            )
            return
        if name == "aget":
            slots = [self.var(v.n) for v in e.a[1:]]
            if slots != list(range(slots[0], slots[0] + len(slots))):
                raise Unsupported("array elements that are not consecutive variables")
            self.expr(e.a[0])
            self.emit("VarX", slots[0] | len(slots) << 24)
            return
        args = list(e.a)
        if name in ("land", "lor", "ite"):
            self.lazy(name, args)
            return
        if name in _COMMUTES and len(args) == 2 and self.constant(args[0]) is not None:
            args.reverse()
        if len(args) == 2 and self.immediate(name, e, args):
            return
        for arg in args:
            self.expr(arg)
        if name == "zext":
            return
        if name == "vol":
            self.emit("Rd")
        elif name in _MASKED:
            self.emit(_MASKED[name], (1 << e.w) - 1)
        elif name in _SIGNED:
            self.emit(_SIGNED[name], 1 << (width(e.a[0]) - 1))
        elif name in _PLAIN:
            self.emit(_PLAIN[name])
        else:
            raise Unsupported(f"no native form for {name}")

    def lazy(self, name: str, args) -> None:
        """Code for ``a and b``, ``a or b`` and ``b if c else d``, which evaluate only the arm
        they take (Python's forms, so an arm that would raise is not run)."""
        self.expr(args[0])
        if name == "ite":
            other = self.emit("Jz")
            self.expr(args[1])
            over = self.emit("Jmp")
            self.patch(other, self.here())
            self.expr(args[2])
            self.patch(over, self.here())
        else:
            skip = self.emit("JzKeep" if name == "land" else "JnzKeep")
            self.expr(args[1])
            self.patch(skip, self.here())

    @staticmethod
    def constant(e):
        """The value of ``e`` if it is a known constant, else None."""
        return e.v if isinstance(e, K) else None

    def immediate(self, name: str, e: Op, args) -> bool:
        """Code for the operator ``name`` over ``args`` with a constant right operand, if it has
        an immediate form that fits; whether it was emitted."""
        k = self.constant(args[1])
        mask = (1 << e.w) - 1
        if k is None:
            return False
        if name in _IMMEDIATE:
            self.expr(args[0])
            self.emit(_IMMEDIATE[name], k)
            return True
        if name in _PACKED and k in I32 and mask < 1 << 31:
            self.expr(args[0])
            self.emit(_PACKED[name], mask << 32 | (k & 0xFFFFFFFF))
            return True
        return False

    def branch(self, cond, when: bool) -> int:
        """Code that jumps, to a target the caller patches, when the truth of ``cond`` is
        ``when``; the index of the jump."""
        if isinstance(cond, Op):
            if cond.op == "lnot":
                return self.branch(cond.a[0], not when)
            jump = _JUMPS.get((cond.op, when))
            if jump:
                left, right = cond.a
                k = self.constant(right)
                self.expr(left)
                if k is not None and k in I39:
                    return self.emit(jump + "K", k << 24)
                self.expr(right)
                return self.emit(jump)
        self.expr(cond)
        return self.emit("Jnz" if when else "Jz")

    def block(self, stmts) -> None:
        """Code for ``stmts``."""
        for s in stmts:
            self.stmt(s)

    def stmt(self, s) -> None:
        """Code for one statement."""
        tag = s[0]
        if tag == "wr":
            self.write(s[1], s[2])
        elif tag == "set":
            self.assign(s[1], s[2])
        elif tag == "if":
            skip = self.branch(s[1], False)
            self.block(s[2])
            if s[3]:
                over = self.emit("Jmp")
                self.patch(skip, self.here())
                self.block(s[3])
                self.patch(over, self.here())
            else:
                self.patch(skip, self.here())
        elif tag == "loop":
            self.loop(s)
        elif tag == "assert":
            self.expr(s[1])
            self.emit("Assert")
        elif tag == "native":
            self.native(s)
        elif tag == "finish":
            self.emit("Halt")
        elif tag == "call":
            self.calls.append((self.emit("Call"), s[1]))
        elif tag == "fire":
            self.block(s[2])
        elif tag == "obj":
            self.react(s)
        elif tag != "hook":
            raise Unsupported(f"no native form for {tag}")

    def react(self, s) -> None:
        """Code that evaluates the operands of the object reaction ``s`` and steps it."""
        for operand in s[4]:
            self.expr(operand)
        out = -1 if s[3] is None else self.var(s[3])
        obj = self.program.objects.get(s[1])
        if obj is None:
            raise Unsupported(f"no object {s[1]!r}")
        self.objects.append(describe(obj, s[2], out, self.var))
        self.emit("Obj", len(self.objects) - 1 | len(s[4]) << 24)

    def native(self, s) -> None:
        """Code that runs the tune's 6502 code at ``s[1]`` with the registers ``s[2]`` until it
        reaches ``s[3]``."""
        given = dict(s[2])
        self.expr(s[1])
        mask = 0
        for bit, name in enumerate("axy"):
            if name in given:
                self.expr(given[name])
                mask |= 1 << bit
        self.expr(s[3])
        self.emit("Native", mask)

    def write(self, addr, value) -> None:
        """A store of ``value`` to ``addr``."""
        k = self.constant(addr)
        if k is not None and 0 <= k <= 0xFFFF:
            self.expr(value)
            self.emit("WrK", k)
        else:
            self.expr(addr)
            self.expr(value)
            self.emit("Wr")

    def assign(self, name: str, value) -> None:
        """An assignment of ``value`` to variable ``name``."""
        k, slot = self.constant(value), self.var(name)
        if k is not None and k in I43 and slot < SETK_VARS:
            self.emit("SetK", slot | k << 20)
        else:
            self.expr(value)
            self.emit("Set", slot)

    def loop(self, s) -> None:
        """``for _ in range(bound): if not cond: break; body`` and, past the bound, the
        failure of a condition that still holds."""
        count = self.var(f"loop@{self.here()}")
        self.emit("SetK", count)
        top = self.here()
        self.emit("Var", count)
        spent = self.emit("JGeK", s[3] << 24)
        leave = self.branch(s[1], False)
        self.block(s[2])
        self.emit("Var", count)
        self.emit("AddK", ((1 << 31) - 1) << 32 | 1)
        self.emit("Set", count)
        self.emit("Jmp", top)
        self.patch(spent, self.here())
        out = self.branch(s[1], False)
        self.emit("Fail")
        self.patch(leave, self.here())
        self.patch(out, self.here())

    def encode(self):
        """``(code, tables, inputs, variables, controls, objects)`` of the program."""
        program = self.program
        inputs = [(self.var(n), REGISTER[n[4:]]) for n in program.inputs]
        self.block(program.body)
        self.emit("Halt")
        starts = {}
        for pid, body in program.procs.items():
            starts[pid] = self.here()
            self.block(body)
            self.emit("Ret")
        for at, pid in self.calls:
            self.patch(at, starts[pid])
        tables = [
            list(key[1]) if key[0] == "lut" else table(*key)
            for key in sorted(self.tables, key=self.tables.get)
        ]
        code = array("q", self.code).tobytes()
        return (
            code,
            [list(t) for t in tables],
            inputs,
            len(self.vars),
            len(program.controls),
            self.objects,
        )


def encode(program: Program):
    """``(code, tables, inputs, variables, controls, objects)`` for ``Core.add``.

    Raises:
      Unsupported: ``program`` uses a form the machine does not run.
    """
    return _Encoder(program).encode()


__all__ = ["Core", "ENV", "Unsupported", "available", "encode"]
