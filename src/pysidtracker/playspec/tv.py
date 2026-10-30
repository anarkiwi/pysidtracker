"""Translation validation: a spec routine against the 6502 code it models.

The 6502 side runs deity-informant's p-code lift (jennings' semantics) over the spec's
z3 memory; z3 decides whether a byte outside the stack page and ``scratch`` cells can
differ, assuming no data read aliases a return address the code itself pushed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import z3
from deity_informant import lift

from ..errors import SidError
from .ir import Program
from .lower import elaborate
from .smt import State, select, step, symbolic_memory
from .syntax import Spec

REGS = {"A": 0, "X": 1, "Y": 2, "SP": 3}
FLAGS = {"C": 8, "Z": 9, "I": 10, "D": 11, "B": 12, "V": 13, "N": 14}
STACK = range(0x100, 0x200)
BUDGET = 20000


class TVError(SidError):
    """Code outside what the symbolic 6502 executor models."""


_PCODE = {
    "INT_ADD": lambda a, b: a + b,
    "INT_SUB": lambda a, b: a - b,
    "INT_AND": lambda a, b: a & b,
    "INT_OR": lambda a, b: a | b,
    "INT_XOR": lambda a, b: a ^ b,
    "INT_LEFT": lambda a, b: a << b,
    "INT_RIGHT": z3.LShR,
}
_TEST = {
    "INT_EQUAL": lambda a, b: a == b,
    "INT_NOTEQUAL": lambda a, b: a != b,
    "INT_LESS": z3.ULT,
    "INT_LESSEQUAL": z3.ULE,
    "INT_CARRY": lambda a, b: z3.ULT(a + b, a),
}


@dataclass
class Path:
    """One 6502 execution path: condition, registers, memory, pc, return stack."""

    cond: object
    regs: Dict[int, object]
    mem: object
    pc: int
    calls: Tuple[int, ...] = ()
    steps: int = 0


@dataclass
class TVResult:
    """``proven``, else a ``counterexample``: the differing address and both bytes, the
    initial bytes of the spec's cells, and the registers on entry."""

    proven: bool
    routine: str
    pc: int
    paths: int
    counterexample: Optional[Dict[str, object]] = None
    unmodelled: List[int] = field(default_factory=list)


class Machine6502:
    """Forking, merging symbolic execution of 6502 code from a concrete image."""

    def __init__(self, image: bytes, operands: Iterable[int], solver: z3.Solver, mem0):
        self.image = image
        self.mem0 = mem0
        self.operands = set(operands)
        self.solver = solver
        self.code: set = set()
        self.ops: Dict[int, dict] = {}
        self.paths = 0
        self.writes: List[object] = []
        self.loads: List[object] = []
        self.pushed: set = set()
        self.resumed: Optional[Path] = None

    def lifted(self, pc: int) -> dict:
        """The lift of the instruction at ``pc``; its non-operand bytes become constraints."""
        if pc not in self.ops:
            try:
                rec = lift(self.image, pc)
            except NotImplementedError as exc:
                raise TVError(str(exc)) from exc
            self.ops[pc] = rec
            for k in range(rec["len"]):
                addr = (pc + k) & 0xFFFF
                if addr not in self.operands and addr not in self.code:
                    self.code.add(addr)
                    self.solver.add(z3.Select(self.mem0, addr) == self.image[addr])
        return self.ops[pc]

    def constant(self, path: Path, pc: int, rec: dict, at, vn):
        """A constant varnode, read from memory when it comes from an operand cell."""
        _space, value, size = vn
        prov = rec["prov"]["ops"].get(at)
        if prov is not None:
            srcs, kind = prov
            addrs = [(pc + s) & 0xFFFF for s in srcs]
            if any(a in self.operands for a in addrs):
                parts = [z3.Select(path.mem, a) for a in addrs]
                term = parts[0] if len(parts) == 1 else z3.Concat(parts[1], parts[0])
                if kind == "hi1":
                    term = term + 1
                return (
                    z3.ZeroExt(8 * size - term.size(), term)
                    if term.size() < 8 * size
                    else term
                )
        return z3.BitVecVal(value, 8 * size)

    def step(self, path: Path) -> List[Path]:
        """Run the instruction at ``path.pc``; the paths that continue (or end)."""
        rec = self.lifted(path.pc)
        pc = path.pc
        uniq: Dict[int, object] = {}

        def get(at, vn):
            space, off, _size = vn
            if space == "c":
                return self.constant(path, pc, rec, at, vn)
            return (path.regs if space == "r" else uniq)[off]

        for i, (name, dst, ins) in enumerate(rec["ops"]):
            args = [get((i, j), vn) for j, vn in enumerate(ins)]
            if name == "STORE":
                self.store(path, args[0], args[1])
                continue
            if name == "LOAD":
                self.loads.append(args[0])
                value = z3.Select(path.mem, args[0])
            elif name == "COPY":
                value = args[0]
            elif name == "INT_ZEXT":
                value = z3.ZeroExt(8 * dst[2] - args[0].size(), args[0])
            elif name in _PCODE:
                a, b = args
                if b.size() < a.size():
                    b = z3.ZeroExt(a.size() - b.size(), b)
                value = _PCODE[name](a, b)
            elif name in _TEST:
                a, b = args
                value = z3.If(_TEST[name](a, b), z3.BitVecVal(1, 8), z3.BitVecVal(0, 8))
            else:
                raise TVError(f"p-code {name} at ${pc:04x}")
            (path.regs if dst[0] == "r" else uniq)[dst[1]] = value
        path.steps += 1
        return self.control(path, rec)

    def store(self, path: Path, addr, value) -> None:
        """A byte store; a constant store into lifted code is refused."""
        simple = z3.simplify(addr)
        if z3.is_bv_value(simple) and simple.as_long() in self.code:
            raise TVError(f"stores into its own code at ${simple.as_long():04x}")
        path.mem = z3.Store(path.mem, addr, value)
        self.writes.append(addr)

    def push(self, path: Path, byte: int) -> None:
        """Push a constant byte at the concrete stack pointer."""
        sp = z3.simplify(path.regs[REGS["SP"]]).as_long()
        path.mem = z3.Store(path.mem, 0x100 | sp, z3.BitVecVal(byte, 8))
        self.pushed.add(0x100 | sp)
        path.regs[REGS["SP"]] = z3.BitVecVal((sp - 1) & 0xFF, 8)

    def control(self, path: Path, rec: dict) -> List[Path]:
        """Follow the instruction's control transfer."""
        kind = rec["ctrl"][0]
        nxt = (path.pc + rec["len"]) & 0xFFFF
        if kind == "next":
            path.pc = nxt
            return [path]
        if kind == "jmp":
            target = rec["ctrl"][1]
            if target <= path.pc and self.solver.check(path.cond) == z3.unsat:
                return []
            path.pc = target
            return [path]
        if kind == "jsr":
            ret = (path.pc + 2) & 0xFFFF
            self.push(path, ret >> 8)
            self.push(path, ret & 0xFF)
            path.calls += (ret,)
            path.pc = rec["ctrl"][1]
            return [path]
        if kind == "rts":
            if not path.calls:
                path.pc = -1
                return [path]
            sp = (z3.simplify(path.regs[REGS["SP"]]).as_long() + 2) & 0xFF
            path.regs[REGS["SP"]] = z3.BitVecVal(sp, 8)
            path.pc = (path.calls[-1] + 1) & 0xFFFF
            path.calls = path.calls[:-1]
            return [path]
        if kind == "br":
            _br, flag, pol, taken, fall = rec["ctrl"]
            cond = path.regs[flag[1]] == pol
            out = []
            for test, target in ((cond, taken), (z3.Not(cond), fall)):
                both = z3.simplify(z3.And(path.cond, test))
                if z3.is_false(both):
                    continue
                if target <= path.pc and self.solver.check(both) == z3.unsat:
                    continue
                out.append(
                    Path(
                        both, dict(path.regs), path.mem, target, path.calls, path.steps
                    )
                )
            return out
        raise TVError(f"control {kind} at ${path.pc:04x}")

    def run(
        self, entry: Path, budget: int = BUDGET, stop: Optional[int] = None
    ) -> Optional[Path]:
        """Every path from ``entry`` to its final ``rts``, merged into one state.

        With ``stop``, paths reaching that pc are merged into ``self.resumed`` instead,
        and the result is None when none returns.
        """
        pending: Dict[Tuple[int, Tuple[int, ...]], Path] = {
            (entry.pc, entry.calls): entry
        }
        done: Optional[Path] = None
        while pending:
            key = min(pending, key=lambda k: (-len(k[1]), k[0]))
            path = pending.pop(key)
            if path.steps > budget:
                raise TVError(
                    f"a path runs past {budget} instructions at ${path.pc:04x}"
                )
            for nxt in self.step(path):
                if stop is not None and nxt.pc == stop:
                    if nxt.calls:
                        raise TVError(f"${stop:04x} reached inside a subroutine")
                    self.paths += 1
                    res = self.resumed
                    self.resumed = nxt if res is None else merge(res, nxt)
                    continue
                if nxt.pc == -1:
                    self.paths += 1
                    done = nxt if done is None else merge(done, nxt)
                    continue
                k = (nxt.pc, nxt.calls)
                pending[k] = merge(pending[k], nxt) if k in pending else nxt
        if done is None and (stop is None or self.resumed is None):
            raise TVError("no path returns")
        return done


def merge(a: Path, b: Path) -> Path:
    """One path standing for ``a`` or ``b``."""
    regs = {r: z3.If(a.cond, a.regs[r], b.regs[r]) for r in a.regs}
    return Path(
        z3.Or(a.cond, b.cond),
        regs,
        z3.If(a.cond, a.mem, b.mem),
        a.pc,
        a.calls,
        max(a.steps, b.steps),
    )


def entry_registers(registers: Dict[str, int]) -> Dict[int, object]:
    """Registers and flags on entry: the given ones concrete, the rest free; SP $FF."""
    regs: Dict[int, object] = {}
    for name, index in {**REGS, **FLAGS}.items():
        if name == "SP":
            regs[index] = z3.BitVecVal(0xFF, 8)
        elif name in registers:
            regs[index] = z3.BitVecVal(registers[name], 8)
        elif name in FLAGS:
            regs[index] = z3.ZeroExt(7, z3.BitVec(f"flag_{name}", 1))
        else:
            regs[index] = z3.BitVec(f"reg_{name}", 8)
    for index in range(16):
        regs.setdefault(index, z3.BitVecVal(0, 8))
    return regs


def validate(
    spec: Spec,
    binding: dict,
    routine: str,
    image: bytes,
    pc: int,
    *,
    registers: Optional[Dict[str, int]] = None,
    preserve: Iterable[str] = (),
    budget: int = BUDGET,
) -> TVResult:
    """Prove spec ``routine`` (``"voice[0].parse"``) equal to the 6502 code at ``pc``.

    Both start from one symbolic memory whose lifted code bytes are ``image``'s (bar
    the spec's ``operand`` cells); ``preserve`` registers must also survive the code.
    """
    program = elaborate(spec, binding, routine=routine)
    operands = {a for c in program.cells if "operand" in c.attrs for a in c.extent()}
    scratch = {a for c in program.cells if "scratch" in c.attrs for a in c.extent()}
    mem0 = symbolic_memory("M0")
    solver = z3.Solver()
    cpu = Machine6502(bytes(image), operands, solver, mem0)
    regs = entry_registers(dict(registers or {}))
    end = cpu.run(Path(z3.BoolVal(True), dict(regs), mem0, pc), budget)
    env = {}
    for name, w in program.inputs.items():
        key = name[4:].upper()
        reg = regs[REGS[key]] if key in REGS else regs[FLAGS[key]]
        env[name] = reg != 0 if w == 1 else reg
    spec_state: State = step(program, mem0, env)
    for load in cpu.loads:
        solver.add(*[load != a for a in sorted(cpu.pushed)])
    memo: dict = {}
    pairs = []
    for at in _distinct(spec_state.writes + cpu.writes):
        if z3.is_bv_value(at) and (at.as_long() in STACK or at.as_long() in scratch):
            continue
        keep = z3.And(
            [z3.Or(z3.ULT(at, STACK.start), z3.UGE(at, STACK.stop))]
            + [at != a for a in sorted(scratch)]
        )
        pairs.append(
            (at, keep, select(spec_state.mem, at, memo), select(end.mem, at, memo))
        )
    claims = [z3.And(keep, a != b) for _at, keep, a, b in pairs]
    claims += [end.regs[REGS[r]] != regs[REGS[r]] for r in preserve]
    claims += [cond for cond, _what in spec_state.obligations]
    result = TVResult(True, routine, pc, cpu.paths)
    if solver.check(z3.Or(claims)) == z3.unsat:
        return result
    model = solver.model()
    result.proven = False
    found = next(
        (
            (at, a, b)
            for at, keep, a, b in pairs
            if z3.is_true(model.eval(z3.And(keep, a != b), model_completion=True))
        ),
        None,
    )
    out: Dict[str, object] = {
        "registers": {r: _byte(model, regs[i]) for r, i in REGS.items()},
        "registers_out": {r: _byte(model, end.regs[REGS[r]]) for r in preserve},
        "cells": _cells(model, program, mem0),
        "memory": _memory(model, mem0),
    }
    if found is not None:
        out.update(
            address=_byte(model, found[0]),
            spec=_byte(model, found[1]),
            code=_byte(model, found[2]),
        )
    result.counterexample = out
    return result


def _memory(model, mem0) -> bytes:
    """The counterexample's initial 64 KiB as the model assigns it."""
    return bytes(_byte(model, z3.Select(mem0, a)) for a in range(0x10000))


def replay(
    spec: Spec,
    binding: dict,
    routine: str,
    memory: bytes,
    pc: int,
    registers: Optional[Dict[str, int]] = None,
) -> List[int]:
    """The addresses (bar the stack page) where the spec routine and the jennings run
    of the code at ``pc`` leave ``memory`` differently: a concrete counterexample check.
    """
    from jennings.devices.mpu6502 import MPU  # pylint: disable=import-outside-toplevel

    from .pyexec import run  # pylint: disable=import-outside-toplevel

    regs = {k.lower(): v for k, v in (registers or {}).items()}
    spec_mem = run(
        elaborate(spec, binding, routine=routine), bytearray(memory), registers=regs
    )
    cpu_mem = bytearray(memory)
    mpu = MPU(memory=cpu_mem)
    for name, value in (registers or {}).items():
        setattr(mpu, name.lower(), value)
    mpu.sp, mpu.pc, depth = 0xFF, pc, 0
    for _ in range(BUDGET):
        opcode = cpu_mem[mpu.pc]
        if opcode == 0x60 and depth == 0:
            break
        depth += {0x20: 1, 0x60: -1}.get(opcode, 0)
        mpu.step()
    return [a for a in range(0x10000) if spec_mem[a] != cpu_mem[a] and a not in STACK]


def _distinct(terms) -> list:
    """``terms`` with duplicates (by z3 identity after simplification) removed."""
    seen, out = set(), []
    for term in terms:
        term = z3.simplify(term)
        if term.get_id() not in seen:
            seen.add(term.get_id())
            out.append(term)
    return out


def _byte(model, term) -> int:
    return model.eval(term, model_completion=True).as_long()


def _cells(model, program: Program, mem0) -> Dict[str, list]:
    """The initial bytes (first 8) of every cell, by cell name."""
    out = {}
    for cell in program.cells:
        values = [_byte(model, z3.Select(mem0, a)) for a in cell.extent()[:8]]
        if values:
            out[f"{cell.owner}.{cell.name}" if cell.owner else cell.name] = values
    return out


__all__ = ["Machine6502", "TVError", "TVResult", "replay", "validate"]
