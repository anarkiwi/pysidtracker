"""Concrete execution: a lowered :class:`~.ir.Program` compiled to one Python function.

``frame(R, W, C, G, P, N)``: ``R``/``W`` read and store a byte, ``C`` holds hidden
control states, ``G(index, base)`` (or None) sees each stream open, ``P`` maps
the entry registers the spec reads (``a``, ``x``, ``y``, flag letters) and
``N(addr, regs, resume)`` (or None) runs the tune's 6502 code for ``native``.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Tuple

from ..errors import SidError
from ..notes import table
from .ir import K, Program, Rd, V, width


class SpecRuntimeError(SidError):
    """A spec assertion failed or a bounded loop ran past its bound."""


_BIN = {
    "add": "(({0} + {1}) & {m})",
    "sub": "(({0} - {1}) & {m})",
    "mul": "(({0} * {1}) & {m})",
    "and": "({0} & {1})",
    "or": "({0} | {1})",
    "xor": "({0} ^ {1})",
    "shl": "(({0} << {1}) & {m})",
    "shr": "({0} >> {1})",
    "eq": "({0} == {1})",
    "ne": "({0} != {1})",
    "ult": "({0} < {1})",
    "ule": "({0} <= {1})",
    "land": "({0} and {1})",
    "lor": "({0} or {1})",
    "word": "({0} | {1} << 8)",
}
_UN = {
    "not": "({0} ^ {m})",
    "neg": "((-{0}) & {m})",
    "zext": "int({0})",
    "lo": "({0} & 255)",
    "hi": "({0} >> 8)",
    "lnot": "(not {0})",
    "sext": "(({0} | 65280) if {0} & 128 else {0})",
}


CHUNK = 100
_FINISH = "return True"
_VAR = re.compile(r"\bv_\w+")


class _Gen:
    """Python source for one program."""

    def __init__(self, program: Program):
        self.helpers: List[List[str]] = []
        self.tables: Dict[tuple, str] = {}
        self.controls: Dict[str, int] = {
            n: i for i, (n, _w, _d) in enumerate(program.controls)
        }

    def name(self, ident: str) -> str:
        """The Python spelling of an IR variable."""
        if ident in self.controls:
            return f"C[{self.controls[ident]}]"
        return "v_" + re.sub(r"\W", "_", ident)

    def ex(self, e) -> str:
        """The Python expression of ``e``."""
        if isinstance(e, K):
            if e.v is None:
                raise SidError("cannot execute a program lowered without a binding")
            return str(e.v)
        if isinstance(e, Rd):
            return f"R({self.ex(e.a)})"
        if isinstance(e, V):
            return self.name(e.n)
        if e.op == "note":
            key = (*(k.v for k in e.a[:3]), 1 << max(8, width(e.a[3])))
            name = self.tables.setdefault(key, f"T{len(self.tables)}")
            return f"{name}[{self.ex(e.a[3])}]"
        args = [self.ex(a) for a in e.a]
        mask = (1 << e.w) - 1
        if e.op in _BIN:
            return _BIN[e.op].format(*args, m=mask)
        if e.op in _UN:
            return _UN[e.op].format(*args, m=mask)
        if e.op == "ite":
            return f"({args[1]} if {args[0]} else {args[2]})"
        if e.op in ("slt", "sle"):
            sign = 1 << (width(e.a[0]) - 1)
            cmp = "<" if e.op == "slt" else "<="
            return f"(({args[0]} ^ {sign}) {cmp} ({args[1]} ^ {sign}))"
        raise SidError(f"no concrete form for {e.op}")

    def block(self, stmts) -> List[str]:
        """The lines of ``stmts``; a block over :data:`CHUNK` lines is split into helpers."""
        parts = [self.stmt(s) for s in stmts] or [["pass"]]
        if sum(map(len, parts)) <= CHUNK:
            return [line for part in parts for line in part]
        out: List[str] = []
        group: List[str] = []
        for part in parts + [None]:
            if group and (part is None or len(group) + len(part) > CHUNK):
                out += self.helper(group)
                group = []
            if part is not None:
                group += part
        return out

    def helper(self, lines: List[str]) -> List[str]:
        """A function holding ``lines`` and the lines that call it."""
        name = f"_f{len(self.helpers)}"
        args = "R, W, C, G, N, V"
        self.helpers.append(
            [f"def {name}({args}):"] + ["    " + line for line in lines]
        )
        call = f"{name}({args})"
        if any(_FINISH in line for line in lines):
            return [f"if {call}:", f"    {_FINISH}"]
        return [call]

    def stmt(self, s) -> List[str]:
        """The lines of one statement."""
        tag = s[0]
        if tag == "wr":
            return [f"W({self.ex(s[1])}, {self.ex(s[2])})"]
        if tag == "set":
            return [f"{self.name(s[1])} = {self.ex(s[2])}"]
        if tag == "if":
            return self.branch(s)
        if tag == "loop":
            cond = self.ex(s[1])
            return (
                [f"for _ in range({s[3]}):", f"    if not {cond}:", "        break"]
                + indent(self.block(s[2]))
                + [
                    "else:",
                    f"    if {cond}:",
                    f"        raise SpecRuntimeError('loop at line {s[4]} ran past {s[3]}')",
                ]
            )
        if tag == "assert":
            return [f"if not {self.ex(s[1])}:", f"    raise SpecRuntimeError({s[2]!r})"]
        if tag == "native":
            regs = ", ".join(f"{r!r}: {self.ex(v)}" for r, v in s[2])
            return [
                "if N is None:",
                "    raise SpecRuntimeError('native code needs a CPU (SpecPlayer)')",
                f"N({self.ex(s[1])}, {{{regs}}}, {self.ex(s[3])})",
            ]
        if tag == "finish":
            return [_FINISH]
        if tag == "hook":
            return ["if G is not None:", f"    G({s[1]}, {self.ex(s[2])})"]
        return []

    def branch(self, s) -> List[str]:
        """An ``if`` with an ``elif`` chain flattened."""
        out: List[str] = []
        keyword = "if"
        while True:
            out.append(f"{keyword} {self.ex(s[1])}:")
            out += indent(self.block(s[2]))
            orelse = s[3]
            if len(orelse) == 1 and orelse[0][0] == "if":
                s, keyword = orelse[0], "elif"
                continue
            if orelse:
                out.append("else:")
                out += indent(self.block(orelse))
            return out


def indent(lines: List[str]) -> List[str]:
    """``lines`` one level deeper."""
    return ["    " + line for line in lines]


def _generate(program: Program) -> Tuple[str, Dict[str, tuple]]:
    """``(source, tables)``: the frame function and the ``table`` arguments of the
    note tables (``T0``, ...) it reads.

    Long blocks become helper functions (``_f0`` ...) so no code object is large; a
    variable used by more than one function lives in the list ``V``.
    """
    gen = _Gen(program)
    prelude = [f"{gen.name(name)} = P[{name[4:]!r}]" for name in program.inputs]
    body = gen.block(program.body)
    funcs = gen.helpers + [
        ["def frame(R, W, C, G=None, P=None, N=None):"] + indent(prelude + body)
    ]
    users: Dict[str, int] = {}
    for func in funcs:
        for var in set(_VAR.findall("\n".join(func))):
            users[var] = users.get(var, 0) + 1
    shared = {
        v: f"V[{i}]" for i, v in enumerate(sorted(v for v, n in users.items() if n > 1))
    }
    text = "\n".join("\n".join(func) for func in funcs) + "\n"
    text = _VAR.sub(lambda m: shared.get(m.group(), m.group()), text)
    if shared:
        text = text.replace("N=None):\n", f"N=None):\n    V = [0] * {len(shared)}\n")
    last = text.rindex("def frame(")
    text = text[:last] + text[last:].replace(_FINISH, "return")
    return text, {n: k for k, n in gen.tables.items()}


def source(program: Program) -> str:
    """The Python source of ``program``'s frame function."""
    return _generate(program)[0]


def compile_program(program: Program) -> Callable:
    """``frame(R, W, C, G=None, P=None, N=None)`` running one call of ``program``."""
    text, tables = _generate(program)
    namespace = {"SpecRuntimeError": SpecRuntimeError}
    namespace.update((name, table(*key)) for name, key in tables.items())
    exec(
        compile(text, f"<playspec {program.name}>", "exec"), namespace
    )  # pylint: disable=exec-used
    return namespace["frame"]


def controls(program: Program) -> List[int]:
    """The initial hidden control states of ``program``."""
    return [default for _n, _w, default in program.controls]


def run(
    program: Program, memory: bytearray, frames: int = 1, registers=None
) -> bytearray:
    """Run ``frames`` calls of ``program`` on a flat 64 KiB ``memory`` in place.

    ``registers`` maps entry registers (``a``, ``x``, ``y``, flag letters) the spec reads.
    """
    frame = compile_program(program)
    state = controls(program)
    regs = {name[4:]: 0 for name in program.inputs}
    regs.update(registers or {})
    for _ in range(frames):
        frame(memory.__getitem__, memory.__setitem__, state, None, regs)
    return memory


__all__ = ["SpecRuntimeError", "compile_program", "controls", "run", "source"]
