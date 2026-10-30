"""Replace a spec's ``native`` hooks with the spec statements that do the same.

Each ``native addr(...) resumes pc;`` is lifted: the tune's 6502 from ``addr`` runs
symbolically (:class:`~.tv.Machine6502`) until it reaches ``pc`` (the spec carries on)
or returns (the call ends, a ``finish``). The memory each exit leaves is stated as
spec statements, spliced over the original statement's source text.

The statements read every memory byte they need before they store any, so they see
the memory the hook saw. The spec models no register outputs, no stack page and no
code writes, so the hook may not depend on those: such a hook is refused.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import z3

from ..errors import SidError
from .lower import elaborate
from .smt import symbolic_memory
from .syntax import Spec, parse
from .tv import BUDGET, FLAGS, REGS, STACK, Machine6502, Path, TVError

_TYPES = {1: "bool", 8: "u8", 16: "u16"}
_SHIFTS = {z3.Z3_OP_BSHL: "<<", z3.Z3_OP_BLSHR: ">>"}
_ARITH = {
    z3.Z3_OP_BADD: "+",
    z3.Z3_OP_BSUB: "-",
    z3.Z3_OP_BMUL: "*",
    z3.Z3_OP_BAND: "&",
    z3.Z3_OP_BOR: "|",
    z3.Z3_OP_BXOR: "^",
}
_COMPARE = {
    z3.Z3_OP_EQ: "==",
    z3.Z3_OP_DISTINCT: "!=",
    z3.Z3_OP_ULT: "<",
    z3.Z3_OP_ULEQ: "<=",
    z3.Z3_OP_UGT: ">",
    z3.Z3_OP_UGEQ: ">=",
}
_SIGNED = {
    z3.Z3_OP_SLT: "slt",
    z3.Z3_OP_SLEQ: "sle",
    z3.Z3_OP_SGT: "sgt",
    z3.Z3_OP_SGEQ: "sge",
}


class Stater:
    """z3 terms over the entry memory and registers, stated as playspec ``let``s.

    Anything that reads memory, or is shared, is bound once by a ``let`` so the
    statements that follow cannot see a store they make; the rest stays inline.
    """

    def __init__(self, mem0, inputs: Dict[str, str], prefix: str):
        self.mem0 = mem0
        self.inputs = inputs
        self.prefix = prefix
        self.lets: List[str] = []
        self.done: Dict[object, Tuple[str, int]] = {}
        self.uses: Dict[object, int] = {}
        self.reads: Dict[object, bool] = {}

    def count(self, roots) -> None:
        """Count each subterm's uses and mark those that read memory."""
        for root in roots:
            self._count(root)

    def _count(self, term) -> bool:
        key = term
        self.uses[key] = self.uses.get(key, 0) + 1
        if key in self.reads:
            return self.reads[key]
        below = [self._count(term.arg(i)) for i in range(term.num_args())]
        reads = z3.is_select(term) or any(below)
        self.reads[key] = reads
        return reads

    def width(self, term) -> int:
        """1 for a truth value, else the bits of its byte or word container (8 or 16).

        A narrower bit vector lives in the container with its high bits zero."""
        if z3.is_bool(term):
            return 1
        if term.size() > 16:
            raise TVError("the hook computes a value wider than 16 bits")
        return 8 if term.size() <= 8 else 16

    def state(self, term) -> Tuple[str, int]:
        """``(text, container width)`` of ``term``."""
        key = term
        if key in self.done:
            return self.done[key]
        text = self._text(term)
        w = self.width(term)
        simple = z3.is_bv_value(term) or z3.is_const(term) or z3.is_true(term)
        if not simple and (self.reads[key] or self.uses[key] > 1):
            name = f"{self.prefix}_t{len(self.lets)}"
            self.lets.append(f"let {name}: {_TYPES[w]} = {text};")
            text = name
        self.done[key] = (text, w)
        return text, w

    def literal(self, term, w: int) -> str:
        """A constant as a literal, as a ``word`` when a bare literal would be untyped."""
        value = term.as_long()
        if w == 16:
            return f"word(0x{value & 255:02x}, 0x{value >> 8:02x})"
        return f"0x{value:02x}"

    def operand(self, term) -> str:
        """``term`` as an operand next to a typed one (a bare literal takes its width)."""
        if z3.is_bv_value(term):
            return f"0x{term.as_long():0{self.width(term) // 4}x}"
        return self.state(term)[0]

    def arm(self, term) -> str:
        """``term`` as an ``ite`` arm or a stored value."""
        if z3.is_bv_value(term):
            return self.literal(term, self.width(term))
        return self.state(term)[0]

    def masked(self, text: str, term) -> str:
        """``text`` kept to the bits of ``term`` (its container may be wider)."""
        w = self.width(term)
        if term.size() == w:
            return text
        return f"({text} & 0x{(1 << term.size()) - 1:0{w // 4}x})"

    def widen(self, term, w: int) -> str:
        """``term`` as a container of ``w`` bits."""
        if self.width(term) == w:
            return self.operand(term)
        if z3.is_bv_value(term):
            return f"0x{term.as_long():04x}"
        return f"word({self.state(term)[0]}, 0)"

    def _text(self, term) -> str:
        # pylint: disable=too-many-return-statements,too-many-branches
        if z3.is_true(term):
            return "true"
        if z3.is_false(term):
            return "false"
        if z3.is_bv_value(term):
            return self.literal(term, self.width(term))
        if z3.is_const(term):
            name = term.decl().name()
            if name not in self.inputs:
                raise TVError(
                    f"the hook reads {name}, which the native arguments do not give it"
                )
            return self.inputs[name]
        if z3.is_select(term):
            if not term.arg(0).eq(self.mem0):
                raise TVError("the hook reads memory through an unresolved store")
            return f"mem[{self.operand(term.arg(1))}]"
        kind = term.decl().kind()
        args = [term.arg(i) for i in range(term.num_args())]
        if kind in _ARITH or kind in _SHIFTS or kind in _COMPARE:
            if kind in _SHIFTS and not z3.is_bv_value(args[1]):
                raise TVError("the hook shifts by a variable amount")
            if kind in _COMPARE and z3.is_bool(args[0]):
                raise TVError("the hook compares truth values")
            if kind in _SHIFTS:
                amount = args[1].as_long()
                text = f"({self.operand(args[0])} {_SHIFTS[kind]} {amount})"
            else:
                op = _ARITH.get(kind) or _COMPARE[kind]
                text = "(" + f" {op} ".join(self._side(a, args) for a in args) + ")"
            if kind in (z3.Z3_OP_BADD, z3.Z3_OP_BSUB, z3.Z3_OP_BMUL, z3.Z3_OP_BSHL):
                return self.masked(text, term)
            return text
        if kind in _SIGNED:
            if args[0].size() != self.width(args[0]):
                raise TVError("the hook compares a partial byte as signed")
            return f"{_SIGNED[kind]}({self._side(args[0], args)}, {self._side(args[1], args)})"
        if kind == z3.Z3_OP_AND:
            return "(" + " && ".join(self.state(a)[0] for a in args) + ")"
        if kind == z3.Z3_OP_OR:
            return "(" + " || ".join(self.state(a)[0] for a in args) + ")"
        if kind == z3.Z3_OP_NOT:
            return f"(!{self.state(args[0])[0]})"
        if kind == z3.Z3_OP_BNOT:
            return self.masked(f"(~{self.operand(args[0])})", term)
        if kind == z3.Z3_OP_BNEG:
            return self.masked(f"(-{self.operand(args[0])})", term)
        if kind == z3.Z3_OP_ITE:
            return f"ite({self.state(args[0])[0]}, {self.arm(args[1])}, {self.arm(args[2])})"
        if kind == z3.Z3_OP_ZERO_EXT:
            return self.widen(args[0], self.width(term))
        if kind == z3.Z3_OP_CONCAT:
            w = self.width(term)
            if w == 16 and [a.size() for a in args] == [8, 8]:
                return f"word({self.arm(args[1])}, {self.arm(args[0])})"
            parts, offset = [], 0
            for part in reversed(args):
                text = self.widen(part, w)
                parts.append(f"({text} << {offset})" if offset else text)
                offset += part.size()
            return "(" + " | ".join(parts) + ")"
        if kind == z3.Z3_OP_EXTRACT:
            hi, lo = term.params()
            inner = args[0]
            text = self.state(inner)[0]
            if lo:
                text = f"({text} >> {lo})"
            cont = self.width(inner)
            if hi + 1 < inner.size() or (lo and hi + 1 < cont):
                text = f"({text} & 0x{(1 << (hi - lo + 1)) - 1:0{cont // 4}x})"
            if self.width(term) < cont:
                text = f"lo({text})"
            return text
        raise TVError(
            f"the hook computes {term.decl().name()}, which a spec cannot state"
        )

    def _side(self, term, args) -> str:
        """An operand of a binary operator: a literal only beside a non-literal."""
        if z3.is_bv_value(term) and all(z3.is_bv_value(a) for a in args):
            return self.literal(term, self.width(term))
        return self.operand(term)


class Flat:
    """Terms with every read through a store resolved to the entry memory."""

    def __init__(self, mem0):
        self.mem0 = mem0
        self.memo: Dict[object, object] = {}

    def __call__(self, term):
        key = term
        if key not in self.memo:
            if z3.is_select(term):
                out = self.read(term.arg(0), self(term.arg(1)))
            elif term.num_args() == 0:
                out = term
            else:
                out = term.decl()(*[self(term.arg(i)) for i in range(term.num_args())])
            self.memo[key] = out
        return self.memo[key]

    def read(self, mem, index):
        """The byte at ``index`` of memory term ``mem``."""
        if mem.eq(self.mem0):
            return z3.Select(mem, index)
        if z3.is_store(mem):
            at, value = self(mem.arg(1)), self(mem.arg(2))
            below = self.read(mem.arg(0), index)
            if z3.is_bv_value(at) and z3.is_bv_value(index):
                return value if at.as_long() == index.as_long() else below
            return z3.If(at == index, value, below)
        if z3.is_app_of(mem, z3.Z3_OP_ITE):
            return z3.If(
                self(mem.arg(0)),
                self.read(mem.arg(1), index),
                self.read(mem.arg(2), index),
            )
        raise TVError("the hook reads memory the spec cannot state")


def _lift(
    image: bytes,
    operands,
    addr: int,
    resume: int,
    regs: List[str],
    budget: int,
):
    """Run the hook symbolically: ``(machine, escape path or None, resume path or None)``."""
    mem0 = symbolic_memory("M0")
    cpu = Machine6502(image, operands, z3.Solver(), mem0)
    entry = {}
    for name, index in {**REGS, **FLAGS}.items():
        if name == "SP":
            entry[index] = z3.BitVecVal(0xFF, 8)
        elif name.lower() in regs:
            entry[index] = z3.BitVec(f"in_{name.lower()}", 8)
        else:
            entry[index] = z3.BitVec(f"free_{name}", 8)
    for index in range(16):
        entry.setdefault(index, z3.BitVecVal(0, 8))
    escape = cpu.run(Path(z3.BoolVal(True), entry, mem0, addr), budget, stop=resume)
    return mem0, cpu, escape, cpu.resumed


def state_hook(
    image: bytes,
    operands,
    addr: int,
    resume: int,
    args: Dict[str, str],
    prefix: str,
    indent: str,
    budget: int = BUDGET,
) -> str:
    """The spec statements for the hook at ``addr`` (``args``: register -> expression)."""
    mem0, cpu, escape, resumed = _lift(
        image, operands, addr, resume, list(args), budget
    )
    exits = [("escape", escape), ("resume", resumed)]
    exits = [(k, p) for k, p in exits if p is not None]
    flat = Flat(mem0)
    written = []
    seen = set()
    for at in cpu.writes:
        at = z3.simplify(flat(at))
        if z3.is_bv_value(at) and at.as_long() in STACK:
            continue
        if at not in seen:
            seen.add(at)
            written.append(at)
    stores: Dict[str, List[Tuple[object, object]]] = {}
    for kind, path in exits:
        stores[kind] = []
        for at in written:
            value = z3.simplify(flat(z3.Select(path.mem, at)))
            if not value.eq(z3.simplify(z3.Select(mem0, at))):
                stores[kind].append((at, value))
    conds = {k: z3.simplify(flat(p.cond)) for k, p in exits}
    roots = [t for k in stores for pair in stores[k] for t in pair]
    if len(exits) == 2:
        roots.append(conds["escape"])
    inputs = {f"in_{r}": f"{prefix}_{r}" for r in args}
    lets = Stater(mem0, inputs, prefix)
    lets.count(roots)
    pad = indent
    out = [f"let {prefix}_{r}: u8 = {text};" for r, text in args.items()]

    def block(kind: str, depth: str) -> List[str]:
        lines = [
            f"{depth}mem[{lets.operand(at)}] := {lets.arm(value)};"
            for at, value in stores[kind]
        ]
        if kind == "escape":
            lines.append(f"{depth}finish;")
        return lines

    if len(exits) == 2:
        cond = _bare(lets.state(conds["escape"])[0])
        body = [f"if {cond} {{"] + block("escape", "    ")
        rest = block("resume", "    ")
        body += ["} else {"] + rest + ["}"] if rest else ["}"]
    else:
        body = block(exits[0][0], "")
    # the lets are discovered while the bodies are written, so splice them first
    return f"\n{pad}".join(out + lets.lets + body)


def _bare(text: str) -> str:
    """``text`` without one pair of parentheses that wraps all of it."""
    if text.startswith("(") and text.endswith(")"):
        depth = 0
        for i, ch in enumerate(text):
            depth += (ch == "(") - (ch == ")")
            if depth == 0 and i < len(text) - 1:
                return text
        return text[1:-1]
    return text


def count_natives(node) -> int:
    """The number of ``native`` statements in an AST."""
    if isinstance(node, tuple):
        return int(bool(node) and node[0] == "native") + sum(
            count_natives(n) for n in node
        )
    if isinstance(node, list):
        return sum(count_natives(n) for n in node)
    if isinstance(node, dict):
        return sum(count_natives(n) for n in node.values())
    return 0


def unnative(spec: Spec, binding: dict, image: bytes, *, budget: int = BUDGET) -> Spec:
    """``spec`` with every ``native`` hook replaced by spec statements.

    ``binding`` is the layout the spec is elaborated against and ``image`` the 64 KiB
    the tune plays from (its post-init memory). Raises :class:`~..errors.SidError` for
    a hook the spec cannot state.
    """
    program = elaborate(spec, binding)
    hooks: Dict[object, Tuple[int, int]] = {}
    for span, addr, resume in program.natives:
        if addr.v is None or resume.v is None:
            raise SidError("a native hook's address does not fold to a constant")
        if hooks.setdefault(span, (addr.v, resume.v)) != (addr.v, resume.v):
            raise SidError(
                f"the native hook at offset {span.start} has a different address per instance"
            )
    if count_natives(spec.machines) + count_natives(spec.procs) != len(hooks):
        raise SidError("a native hook is never reached by the schedule")
    operands = {a for c in program.cells if "operand" in c.attrs for a in c.extent()}
    text = spec.source
    for number, span in sorted(
        enumerate(sorted(hooks, key=lambda s: s.start)), key=lambda p: -p[1].start
    ):
        addr, resume = hooks[span]
        args = {
            reg: text[a:b].split("=", 1)[1].strip().rstrip(";")
            for reg, (a, b) in span.args
        }
        line_start = text.rfind("\n", 0, span.start) + 1
        indent = text[line_start : span.start]
        indent = (
            indent if not indent.strip() else " " * (len(indent) - len(indent.lstrip()))
        )
        stated = state_hook(
            bytes(image), operands, addr, resume, args, f"nat{number}", indent, budget
        )
        text = text[: span.start] + stated + text[span.end :]
    out = parse(text)
    elaborate(out, binding)
    return out


def equivalent(
    make_player: Callable[[Spec], object], old: Spec, new: Spec, frames: int = 500
) -> None:
    """Play ``frames`` calls with both specs; raise unless every call's SID registers
    and every byte outside the stack page match."""
    a, b = make_player(old), make_player(new)
    for number in range(frames):
        a.play_frame()
        b.play_frame()
        if bytes(a.regs) != bytes(b.regs):
            raise SidError(f"frame {number}: the SID registers differ")
        ma, mb = a.view(), b.view()
        if ma != mb:
            diff = [i for i in range(0x10000) if ma[i] != mb[i] and i not in STACK]
            if diff:
                raise SidError(
                    f"frame {number}: memory differs at ${diff[0]:04x}"
                    f" ({ma[diff[0]]:02x} vs {mb[diff[0]]:02x})"
                )


__all__ = ["equivalent", "unnative"]
