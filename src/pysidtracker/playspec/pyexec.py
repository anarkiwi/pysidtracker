"""Concrete execution: a lowered :class:`~.ir.Program` compiled to one Python function.

``frame(R, W, C, G, P, N)``: ``R``/``W`` read and store a byte, ``C`` holds hidden
control states, ``G(index, base)`` (or None) sees each stream open, ``P`` maps
the entry registers the spec reads (``a``, ``x``, ``y``, flag letters) and
``N(addr, regs, resume)`` (or None) runs the tune's 6502 code for ``native``.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List

from ..errors import SidError
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


class _Gen:
    """Python source for one program."""

    def __init__(self, program: Program):
        self.lines: List[str] = []
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

    def emit(self, stmts, depth: int) -> None:
        """Append ``stmts`` at indentation ``depth``."""
        pad = "    " * depth
        if not stmts:
            self.lines.append(pad + "pass")
        for s in stmts:
            tag = s[0]
            if tag == "wr":
                self.lines.append(f"{pad}W({self.ex(s[1])}, {self.ex(s[2])})")
            elif tag == "set":
                self.lines.append(f"{pad}{self.name(s[1])} = {self.ex(s[2])}")
            elif tag == "if":
                self.branch(s, depth)
            elif tag == "loop":
                cond = self.ex(s[1])
                self.lines.append(f"{pad}for _ in range({s[3]}):")
                self.lines.append(f"{pad}    if not {cond}:")
                self.lines.append(f"{pad}        break")
                self.emit(s[2], depth + 1)
                self.lines.append(f"{pad}else:")
                self.lines.append(f"{pad}    if {cond}:")
                self.lines.append(
                    f"{pad}        raise SpecRuntimeError('loop at line {s[4]} ran past {s[3]}')"
                )
            elif tag == "assert":
                self.lines.append(f"{pad}if not {self.ex(s[1])}:")
                self.lines.append(f"{pad}    raise SpecRuntimeError({s[2]!r})")
            elif tag == "native":
                regs = ", ".join(f"{r!r}: {self.ex(v)}" for r, v in s[2])
                self.lines.append(f"{pad}if N is None:")
                self.lines.append(
                    f"{pad}    raise SpecRuntimeError('native code needs a CPU (SpecPlayer)')"
                )
                self.lines.append(
                    f"{pad}N({self.ex(s[1])}, {{{regs}}}, {self.ex(s[3])})"
                )
            elif tag == "finish":
                self.lines.append(f"{pad}return")
            elif tag == "hook":
                self.lines.append(f"{pad}if G is not None:")
                self.lines.append(f"{pad}    G({s[1]}, {self.ex(s[2])})")

    def branch(self, s, depth: int) -> None:
        """An ``if`` with an ``elif`` chain flattened."""
        pad = "    " * depth
        keyword = "if"
        while True:
            self.lines.append(f"{pad}{keyword} {self.ex(s[1])}:")
            self.emit(s[2], depth + 1)
            orelse = s[3]
            if len(orelse) == 1 and orelse[0][0] == "if":
                s, keyword = orelse[0], "elif"
                continue
            if orelse:
                self.lines.append(f"{pad}else:")
                self.emit(orelse, depth + 1)
            return


def source(program: Program) -> str:
    """The Python source of ``program``'s frame function."""
    gen = _Gen(program)
    gen.lines.append("def frame(R, W, C, G=None, P=None, N=None):")
    for name in program.inputs:
        gen.lines.append(f"    {gen.name(name)} = P[{name[4:]!r}]")
    gen.emit(program.body, 1)
    return "\n".join(gen.lines) + "\n"


def compile_program(program: Program) -> Callable:
    """``frame(R, W, C, G=None, P=None, N=None)`` running one call of ``program``."""
    namespace = {"SpecRuntimeError": SpecRuntimeError}
    exec(
        compile(source(program), f"<playspec {program.name}>", "exec"), namespace
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
