"""Why a player's grid diverges from the C64: its Python execution beside the 6502's.

:class:`PyTrace` records every line of the player's own code (values read, branch
taken, values assigned); :func:`cpu_history` every jennings instruction (registers,
memory touched). :func:`explain_divergence` aligns both on their SID writes.
"""

from __future__ import annotations

import argparse
import ast
import enum
import importlib
import inspect
import json
import linecache
import numbers
import os
import reprlib
import sys
import textwrap
import types
from dataclasses import asdict, dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from vagrant_moses import memmap as mm

from . import gridcmp
from . import registers as reg
from .emu import observe_data_access, run_to_rts
from .image import SidImage
from .oracle import boot_flat, register_grid

_IMPURE = (
    ast.Call,
    ast.Lambda,
    ast.NamedExpr,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Starred,
    ast.JoinedStr,
)
_HIDDEN = (types.ModuleType, type, types.FunctionType, types.BuiltinFunctionType)
_HIDDEN += (types.MethodType, types.BuiltinMethodType)
_FLAGS = "NV-BDIZC"


def _pure(node) -> bool:
    """Whether evaluating ``node`` can run no user code beyond attribute reads."""
    return not any(isinstance(n, _IMPURE) for n in ast.walk(node))


def _compile(node) -> types.CodeType:
    return compile(ast.unparse(node), "<whydiff>", "eval")


def _expr(node) -> Tuple[str, types.CodeType, tuple]:
    """``(text, code, key spec)`` of a pure name/attribute/subscript chain."""
    if isinstance(node, ast.Name):
        spec: tuple = ("N", node.id)
    elif isinstance(node, ast.Attribute):
        spec = ("A", _compile(node.value), node.attr)
    else:
        chain = isinstance(node.value, (ast.Name, ast.Attribute, ast.Subscript))
        base = _expr(node.value)[2] if chain else ("",)
        part = node.slice
        if any(isinstance(n, ast.Slice) for n in ast.walk(part)) and not isinstance(
            part, ast.Slice
        ):
            spec = ("",)
        elif isinstance(part, ast.Slice):
            bounds = (part.lower, part.upper, part.step)
            text = ", ".join("None" if b is None else ast.unparse(b) for b in bounds)
            item = compile(f"slice({text})", "<whydiff>", "eval")
            spec = ("Z", _compile(node.value), item, base)
        else:
            spec = ("S", _compile(node.value), _compile(part), base)
    return ast.unparse(node), _compile(node), spec


def _reads(nodes) -> List[Tuple[str, types.CodeType, tuple]]:
    """The maximal pure name/attribute/subscript chains ``nodes`` read, in order."""
    out: Dict[str, tuple] = {}

    def visit(node, store=False):
        chain = isinstance(node, (ast.Name, ast.Attribute, ast.Subscript))
        if chain and not store and _pure(node):
            got = _expr(node)
            out.setdefault(got[0], got)
            if isinstance(node, ast.Subscript):
                visit(node.slice)
            return
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                visit(node.func.value)
            elif not isinstance(node.func, ast.Name):
                visit(node.func)
            for arg in node.args + [k.value for k in node.keywords]:
                visit(arg)
            return
        if store and isinstance(node, (ast.Attribute, ast.Subscript)):
            visit(node.value)
            if isinstance(node, ast.Subscript):
                visit(node.slice)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, store and isinstance(node, (ast.Tuple, ast.List)))

    for node, store in nodes:
        visit(node, store)
    return list(out.values())


def _targets(nodes) -> List[Tuple[str, types.CodeType, tuple]]:
    """The assignment targets of ``nodes``, tuples flattened, as load expressions."""
    out = []
    for node in nodes:
        if isinstance(node, (ast.Tuple, ast.List)):
            out += _targets(node.elts)
        elif isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)) and _pure(node):
            out.append(_expr(node))
    return out


@dataclass(frozen=True)
class _Stmt:
    text: str
    reads: tuple
    cond: Optional[types.CodeType]
    sets: tuple
    start: int
    control: Optional[int]
    fresh: bool


_BRANCHING = (ast.If, ast.For, ast.AsyncFor, ast.While)


def _header(node):
    """``(last header line, [(read node, is_store)], condition, targets)`` of a statement."""
    end = node.end_lineno
    reads, cond, sets = [], None, []
    if isinstance(node, (ast.If, ast.While)):
        end, reads, cond = node.test.end_lineno, [(node.test, False)], node.test
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        end, reads, sets = node.iter.end_lineno, [(node.iter, False)], [node.target]
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        end = node.items[-1].context_expr.end_lineno
        reads = [(i.context_expr, False) for i in node.items]
        sets = [i.optional_vars for i in node.items if i.optional_vars is not None]
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        end = node.body[0].lineno - 1
    elif isinstance(node, ast.Match):
        end, reads = node.subject.end_lineno, [(node.subject, False)]
    elif hasattr(node, "handlers"):
        end = node.lineno
    else:
        reads = [(node, False)]
        if isinstance(node, ast.Assign):
            reads = [(node.value, False)] + [(t, True) for t in node.targets]
            sets = node.targets
        elif isinstance(node, ast.AugAssign):
            reads = [(node.target, False), (node.value, False)]
            sets = [node.target]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            reads, sets = [(node.value, False), (node.target, True)], [node.target]
        elif isinstance(node, ast.Assert):
            cond = node.test
    return end, reads, cond, sets


def _statements(node, control=None):
    """Every statement under ``node`` with the line of its nearest enclosing branch."""
    for value in ast.iter_fields(node):
        for child in value[1] if isinstance(value[1], list) else [value[1]]:
            if isinstance(child, ast.stmt):
                yield child, control
                inner = child.lineno if isinstance(child, _BRANCHING) else control
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    inner = None
                yield from _statements(child, inner)
            elif isinstance(child, ast.AST) and not isinstance(child, ast.expr):
                yield from _statements(child, control)


def _index(path: str) -> Dict[int, _Stmt]:
    """Each source line of ``path`` to the innermost statement header it is part of."""
    lines = linecache.getlines(path)
    try:
        tree = ast.parse("".join(lines))
    except (SyntaxError, ValueError):
        return {}
    index: Dict[int, _Stmt] = {}
    for node, control in _statements(tree):
        end, reads, cond, sets = _header(node)
        end = max(end, node.lineno)
        stmt = _Stmt(
            " ".join(s.strip() for s in lines[node.lineno - 1 : end]),
            tuple(_reads(reads)),
            _compile(cond) if cond is not None and _pure(cond) else None,
            tuple(_targets(sets)),
            node.lineno,
            control,
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and not isinstance(
                node.value, (ast.Name, ast.Attribute, ast.Subscript, type(None))
            ),
        )
        for line in range(node.lineno, end + 1):
            held = index.get(line)
            if held is None or held.start <= node.lineno:
                index[line] = stmt
    return index


class _Repr(reprlib.Repr):
    """Short value text: ints with hex, plain objects as stable ``<Class#n>`` handles."""

    def __init__(self):
        super().__init__()
        self.maxstring, self.maxother, self.maxlist, self.maxtuple = 48, 48, 8, 8
        self.maxdict, self.maxlevel = 6, 2
        self._ids: Dict[int, int] = {}
        self._held: List[object] = []

    def repr_int(self, x, level):  # pylint: disable=unused-argument
        return f"{x}/${x:02X}" if 9 < x < 0x10000 else str(x)

    def repr_bool(self, x, level):  # pylint: disable=unused-argument
        return str(x)

    def repr_bytes(self, x, level):  # pylint: disable=unused-argument
        head = " ".join(f"{b:02X}" for b in x[:12])
        return f"{type(x).__name__}[{len(x)}] {head}{' ...' if len(x) > 12 else ''}"

    repr_bytearray = repr_bytes

    def repr_ndarray(self, x, level):  # pylint: disable=unused-argument
        flat = ", ".join(str(v) for v in x.ravel()[:8])
        return f"ndarray{list(x.shape)}[{flat}{', ...' if x.size > 8 else ''}]"

    def repr_instance(self, x, level):
        if isinstance(x, numbers.Integral):
            return self.repr_int(int(x), level)
        if isinstance(x, (enum.Enum, numbers.Number)) or x is None:
            return str(x)
        if type(x).__repr__ is object.__repr__ or hasattr(x, "__dataclass_fields__"):
            n = self._ids.get(id(x))
            if n is None:
                n = self._ids[id(x)] = len(self._ids)
                self._held.append(x)
            return f"<{type(x).__name__}#{n}>"
        return super().repr_instance(x, level)


@dataclass
class PyEvent:
    """One step of a traced Python run.

    ``call`` (``reads`` the arguments), ``line`` (``reads`` before it runs, ``cond``
    a condition's outcome, ``sets`` what it assigned), ``return``, ``exception`` or
    ``write`` (SID ``register``/``value``); ``deps`` the events its values came from.
    """

    kind: str
    depth: int
    func: str
    path: str
    line: int
    text: str
    reads: List[Tuple[str, str]] = field(default_factory=list)
    cond: Optional[bool] = None
    sets: List[Tuple[str, str]] = field(default_factory=list)
    register: Optional[int] = None
    value: Optional[int] = None
    deps: List[int] = field(default_factory=list)
    site: Optional[int] = None
    rets: List[int] = field(default_factory=list)
    live: List[Tuple[str, str]] = field(default_factory=list)
    state: List[Tuple[str, str, tuple]] = field(default_factory=list, repr=False)


def _scalar(value) -> bool:
    return value is None or isinstance(value, (numbers.Number, str))


class PyTrace:
    """A context manager tracing the Python code under ``roots`` line by line.

    A root is a path, or the pseudo-filename of exec'd code, its text in ``sources``. A
    call of a ``writes`` code object (``(self, addr, value)``, as ``MemPlayer._wr``) to a
    SID address records a ``write``. :func:`sys.settrace` is installed only while entered.
    """

    def __init__(
        self,
        roots: Sequence,
        writes: Sequence[types.CodeType] = (),
        sources: Optional[Dict[str, str]] = None,
    ):
        self.sources = dict(sources or {})
        self.roots = tuple(os.path.realpath(r) for r in roots if r not in self.sources)
        self.events: List[PyEvent] = []
        self._writes = frozenset(writes)
        self._files: Dict[str, Dict[int, _Stmt]] = {}
        self._repr = _Repr()
        self._paths: Dict[str, bool] = {}
        self._depth: Dict[object, int] = {}
        self._serial: Dict[object, int] = {}
        self._pending: Dict[object, Tuple[int, _Stmt]] = {}
        self._branch: Dict[object, Dict[int, int]] = {}
        self._last: Dict[tuple, int] = {}
        self._fresh: set = set()
        self.objects: Dict[int, object] = {}
        self._saved = None

    def __enter__(self) -> "PyTrace":
        for name, text in self.sources.items():
            linecache.cache[name] = (len(text), None, text.splitlines(True), name)
        self._saved = sys.gettrace()
        sys.settrace(self._global)
        return self

    def __exit__(self, *exc) -> None:
        sys.settrace(self._saved)
        for held in (self._depth, self._serial, self._pending, self._branch):
            held.clear()

    def _traced(self, path: str) -> bool:
        hit = self._paths.get(path)
        if hit is None and path in self.sources:
            hit = self._paths[path] = True
        elif hit is None:
            real = os.path.realpath(path)
            hit = self._paths[path] = any(
                real == r or real.startswith(r + os.sep) for r in self.roots
            )
        return hit

    def _stmt(self, path: str, line: int) -> Optional[_Stmt]:
        index = self._files.get(path)
        if index is None:
            index = self._files[path] = _index(path)
        return index.get(line)

    def _keys(self, frame, spec, local) -> List[tuple]:
        """The storage a chain names: a local, global, attribute, item or slice's items."""
        if not spec[0]:
            return []
        try:
            if spec[0] == "N":
                if spec[1] in local:
                    return [("L", self._serial[frame], spec[1])]
                return [("G", spec[1])]
            base = eval(spec[1], frame.f_globals, local)  # pylint: disable=eval-used
            self.objects[id(base)] = base
            if spec[0] == "A":
                return [(id(base), spec[2])]
            item = eval(spec[2], frame.f_globals, local)  # pylint: disable=eval-used
            if spec[0] == "Z":
                return [(id(base), i) for i in range(*item.indices(len(base)))]
            hash(item)
            return [(id(base), item)]
        except Exception:  # pylint: disable=broad-except
            return []

    def _eval(self, frame, exprs):
        """``(text, value text, keys, container keys, scalar)`` of each chain that evaluates."""
        out = []
        local = frame.f_locals
        for text, code, spec in exprs:
            try:
                value = eval(code, frame.f_globals, local)  # pylint: disable=eval-used
            except Exception:  # pylint: disable=broad-except
                continue
            if not isinstance(value, _HIDDEN):
                held = (
                    self._keys(frame, spec[3], local) if spec[0] in ("S", "Z") else []
                )
                own = self._keys(frame, spec, local)
                out.append(
                    (text, self._repr.repr(value), own, held, _scalar(value), spec[0])
                )
        return out

    def store(self, container, index) -> Optional[int]:
        """The event that last stored ``container[index]`` in the trace, if one did."""
        return self._last.get((id(container), index))

    def _emit(self, frame, kind: str, text: str, **kw) -> int:
        code = frame.f_code
        self.events.append(
            PyEvent(
                kind,
                self._depth.get(frame, 0),
                code.co_qualname,
                code.co_filename,
                frame.f_lineno,
                text,
                **kw,
            )
        )
        return len(self.events) - 1

    def _settle(self, frame) -> Optional[int]:
        """Attach what the frame's pending statement assigned; return its event."""
        held = self._pending.pop(frame, None)
        if held is None:
            return None
        idx, stmt = held
        if stmt.sets:
            got = self._eval(frame, stmt.sets)
            self.events[idx].sets = [(t, v) for t, v, *_ in got]
            for _, _, own, *_ in got:
                for key in own:
                    self._last[key] = idx
        return idx

    def _caller(self, frame):
        parent = frame.f_back
        while parent is not None and parent not in self._depth:
            parent = parent.f_back
        return parent

    def _site(self, frame) -> Optional[int]:
        """The pending line of ``frame``'s traced caller: the call site."""
        held = self._pending.get(self._caller(frame))
        return held[0] if held else None

    def _global(self, frame, event, arg):  # pylint: disable=unused-argument
        code = frame.f_code
        if code in self._writes:
            self._write(frame)
        if not self._traced(code.co_filename):
            return None
        parent = self._caller(frame)
        self._depth[frame] = self._depth[parent] + 1 if parent is not None else 0
        self._serial[frame] = len(self.events)
        nargs = code.co_argcount + code.co_kwonlyargcount
        nargs += bool(code.co_flags & 0x04) + bool(code.co_flags & 0x08)
        local = frame.f_locals
        names = [n for n in code.co_varnames[:nargs] if n in local]
        idx = self._emit(
            frame,
            "call",
            code.co_name,
            reads=[(n, self._repr.repr(local[n])) for n in names],
            site=self._site(frame),
        )
        for n in names:
            self._last[("L", self._serial[frame], n)] = idx
        return self._local

    def _write(self, frame) -> None:
        try:
            addr, value = (
                int(frame.f_locals[n]) for n in frame.f_code.co_varnames[1:3]
            )
        except (KeyError, TypeError, ValueError):
            return
        offset = (addr & 0xFFFF) - mm.SID_BASE
        if not 0 <= offset < reg.SID_REG_COUNT:
            return
        caller = self._caller(frame)
        self.events.append(
            PyEvent(
                "write",
                self._depth.get(caller, -1) + 1,
                caller.f_code.co_qualname if caller else "",
                caller.f_code.co_filename if caller else "",
                caller.f_lineno if caller else 0,
                f"SID ${offset:02X} {gridcmp.REG_NAMES[offset]} = ${value & 0xFF:02X}",
                register=offset,
                value=value & 0xFF,
                site=self._site(frame),
            )
        )

    def _line(self, frame) -> None:
        stmt = self._stmt(frame.f_code.co_filename, frame.f_lineno)
        held = self._pending.get(frame)
        if stmt is not None and held is not None and held[1] is stmt:
            return
        self._settle(frame)
        if stmt is None:
            text = linecache.getline(frame.f_code.co_filename, frame.f_lineno)
            self._emit(frame, "line", text.strip())
            return
        cond = None
        if stmt.cond is not None:
            try:
                cond = bool(
                    eval(  # pylint: disable=eval-used
                        stmt.cond, frame.f_globals, frame.f_locals
                    )
                )
            except Exception:  # pylint: disable=broad-except
                cond = None
        got = self._eval(frame, stmt.reads)
        branches = self._branch.setdefault(frame, {})
        deps = [branches[stmt.control]] if stmt.control in branches else []
        live, state = [], []
        last = self._last
        for text, value, own, held, scalar, kind in got:
            if not scalar:
                continue
            hits = [last[k] for k in own if k in last] or [
                last[k] for k in held if last.get(k) in self._fresh
            ]
            if hits:
                deps += hits
            else:
                live.append((text, value))
                if kind in ("A", "S") and own:
                    state.append((text, kind, own[0]))
        idx = self._emit(
            frame,
            "line",
            stmt.text,
            reads=[(t, v) for t, v, *_ in got],
            cond=cond,
            deps=sorted(set(deps)),
            live=live,
            state=state,
        )
        branches[stmt.start] = idx
        if stmt.fresh:
            self._fresh.add(idx)
        self._pending[frame] = (idx, stmt)

    def _local(self, frame, event, arg):
        if event == "line":
            self._line(frame)
        elif event == "return":
            last = self._settle(frame)
            idx = self._emit(
                frame,
                "return",
                "return",
                sets=[("", self._repr.repr(arg))],
                deps=[] if last is None else [last],
            )
            site = self._site(frame)
            if site is not None:
                self.events[site].rets.append(idx)
            for held in (self._depth, self._serial, self._branch):
                held.pop(frame, None)
        elif event == "exception":
            kind, value, _ = arg
            self._emit(frame, "exception", f"raise {kind.__name__}: {value}")
        return self._local

    def slice(self, root: int, limit: int = 200) -> List[int]:
        """The events ``root``'s value came from, nearest first up to ``limit``, in order.

        A line reached through a call site counts only the returns of calls made before it.
        """
        end = len(self.events)
        seen = {root: end}
        frontier = [(root, end)]
        while frontier and len(seen) < limit:
            step = []
            for i, bound in frontier:
                event = self.events[i]
                step += [(d, end) for d in event.deps]
                step += [(r, end) for r in event.rets if r < bound]
                if event.site is not None:
                    step.append((event.site, i))
            frontier = []
            for i, bound in step:
                if seen.get(i, -1) < bound and len(seen) < limit:
                    seen[i] = bound
                    frontier.append((i, bound))
        return sorted(seen)


@dataclass
class CpuStep:
    """One executed 6502 instruction: the registers it left and the memory it touched.

    ``flow`` is its :func:`jennings.dataflow.flow`: each location it set, to its inputs.
    """

    pc: int
    raw: bytes
    text: str
    a: int
    x: int
    y: int
    sp: int
    p: int
    cycle: int
    reads: List[Tuple[int, int]] = field(default_factory=list)
    writes: List[Tuple[int, int]] = field(default_factory=list)
    flow: Dict[object, frozenset] = field(default_factory=dict)

    def sid_writes(self) -> List[Tuple[int, int]]:
        """The ``(register offset, value)`` SID writes this instruction made."""
        out = []
        for addr, value in self.writes:
            offset = addr - mm.SID_BASE
            if 0 <= offset < 0x400 and offset % mm.SID_STRIDE < reg.SID_REG_COUNT:
                out.append((offset % mm.SID_STRIDE, value))
        return out


def cpu_slice(steps: Sequence[CpuStep], root: int):
    """The steps ``root``'s SID store came from, and the state it read at entry.

    A backward dynamic slice over each step's ``flow``: ``(indices, live)``, ``live``
    mapping each location read before this call set it to the value read (registers None).
    """
    top = steps[root]
    sid = {a for a, _ in top.writes if a - mm.SID_BASE in range(0x400)}
    needed = set().union(*(top.flow.get(a, ()) for a in sid))
    values = dict(top.reads)
    keep = [root]
    for i in range(root - 1, -1, -1):
        hit = needed & steps[i].flow.keys()
        if not hit:
            continue
        keep.append(i)
        needed = (needed - hit).union(*(steps[i].flow[loc] for loc in hit))
        values.update(steps[i].reads)
    order = sorted(needed, key=lambda x: (isinstance(x, int), str(x)))
    return sorted(keep), {loc: values.get(loc) for loc in order}


def cpu_history(
    data,
    frame: int,
    *,
    subtune: int = 0,
    max_cycles: int = 8_000_000,
    labels: Optional[Dict[str, int]] = None,
) -> List[CpuStep]:
    """Every instruction of play call ``frame`` (0-based) of a tune on jennings.

    Booted as :func:`~pysidtracker.oracle.register_grid` is, earlier calls untraced;
    each step has its disassembly as the code stood, the registers after it, and the
    bytes it loaded and stored.
    """
    from jennings.dataflow import flow
    from jennings.devices.mpu6502 import MPU
    from jennings.disassembler import Disassembler
    from jennings.utils.addressing import AddressParser

    image = data if isinstance(data, SidImage) else SidImage.from_bytes(bytes(data))
    subject, mpu, mem, play = boot_flat(image, subtune, max_cycles)
    for _ in range(frame):
        run_to_rts(mpu, mem, play, 0, max_cycles)
    dis = Disassembler(
        MPU(memory=subject), AddressParser(labels=dict(labels or {}, PLAY=play))
    )
    reads: List[Tuple[int, int]] = []
    writes: List[Tuple[int, int]] = []
    observe_data_access(
        mpu,
        on_read=lambda addr: reads.append((addr, mem[addr] & 0xFF)),
        on_write=lambda addr, value: writes.append((addr, value)),
    )
    steps: List[CpuStep] = []
    code = bytearray(0x10000)
    start_sp = mpu.sp
    for byte in (0x00, 0x01):
        mem[0x0100 + mpu.sp] = byte
        mpu.sp = (mpu.sp - 1) & 0xFF
    mpu.a, mpu.pc = 0, play
    start = mpu.processorCycles
    while mpu.sp < start_sp and mpu.processorCycles - start <= max_cycles:
        pc = mpu.pc
        size, text = dis.instruction_at(pc)
        raw = bytes(subject[pc : pc + size])
        for i, byte in enumerate(raw):
            code[(pc + i) & 0xFFFF] = byte
        mpu.step()
        steps.append(
            CpuStep(
                pc,
                raw,
                text,
                mpu.a,
                mpu.x,
                mpu.y,
                mpu.sp,
                mpu.p,
                mpu.processorCycles - start,
                reads[:],
                writes[:],
                flow(code, pc, [a for a, _ in reads], [a for a, _ in writes]),
            )
        )
        reads.clear()
        writes.clear()
    mpu._vm._on_read = mpu._vm._on_write = None  # pylint: disable=protected-access
    return steps


def _flags(p: int) -> str:
    return "".join(c if p & (0x80 >> i) else "." for i, c in enumerate(_FLAGS))


def cpu_line(step: CpuStep) -> str:
    """One instruction as a trace-log line: code, registers after it, memory touched."""
    raw = " ".join(f"{b:02X}" for b in step.raw)
    mems = [f"r${a:04X}={v:02X}" for a, v in step.reads]
    mems += [f"w${a:04X}={v:02X}" for a, v in step.writes]
    return (
        f"${step.pc:04X} {raw:<8} {step.text:<16} A={step.a:02X} X={step.x:02X} "
        f"Y={step.y:02X} SP={step.sp:02X} {_flags(step.p)} {' '.join(mems)}"
    ).rstrip()


def py_line(event: PyEvent) -> str:
    """One Python event as a trace-log line, indented by its call depth."""
    pad = "  " * max(event.depth, 0)
    loc = f"{Path(event.path).name}:{event.line}"
    if event.kind == "write":
        return f"{pad}** {event.text}  [{event.func} {loc}]"
    if event.kind == "call":
        args = ", ".join(f"{n}={v}" for n, v in event.reads)
        return f"{pad}-> {event.func}({args})  [{loc}]"
    if event.kind == "return":
        return f"{pad}<- {event.func} return {event.sets[0][1]}"
    if event.kind == "exception":
        return f"{pad}!! {event.text}  [{loc}]"
    out = f"{pad}{event.line:5d}  {event.text}"
    if event.cond is not None:
        out += f"  => {event.cond}"
    if event.reads:
        out += "  | " + ", ".join(f"{n}={v}" for n, v in event.reads)
    if event.sets:
        out += "  := " + ", ".join(f"{n}={v}" for n, v in event.sets)
    return out


def _loc(loc) -> str:
    return f"${loc:04X}" if isinstance(loc, int) else loc


def _columns(left, right, width, lmarks=(), rmarks=(), top=False) -> List[str]:
    """Two wrapped columns, ``>>``-marked lines, padded at the top when ``top``."""

    def cells(lines, marks):
        out = []
        for i, line in enumerate(lines):
            indent = " " * min(len(line) - len(line.lstrip()) + 4, 40)
            wrapped = textwrap.wrap(line, width - 3, subsequent_indent=indent)
            out += [(">> " if i in marks else "   ") + w for w in wrapped or [""]]
        return out

    a, b = cells(left, set(lmarks)), cells(right, set(rmarks))
    if top:
        n = max(len(a), len(b))
        a, b = [""] * (n - len(a)) + a, [""] * (n - len(b)) + b
    return [f"{x:<{width}} | {y}".rstrip() for x, y in zip_longest(a, b, fillvalue="")]


@dataclass
class Side:
    """One side of a divergence: its frame's history and the diverging value's slice.

    ``frame`` is the diverging frame, ``traced`` the one traced: the last to store the
    value. ``root`` is the store, ``keep`` its backward slice, ``live`` its earlier inputs.
    """

    label: str
    frame: int
    traced: int
    events: list
    lines: List[str]
    root: Optional[int]
    keep: List[int]
    live: Dict[str, str]
    state: List[Tuple[str, object, bool, Optional[tuple]]] = field(default_factory=list)
    history: Optional[Callable] = field(default=None, repr=False, compare=False)
    retrace: Optional[Callable] = field(default=None, repr=False, compare=False)
    ram: Optional[Callable] = field(default=None, repr=False, compare=False)

    def summary(self) -> dict:
        """The machine-readable form for ``json.dumps``."""
        events = [
            (
                dict(
                    asdict(e),
                    raw=e.raw.hex(),
                    sid_writes=e.sid_writes(),
                    flow={_loc(k): sorted(map(_loc, v)) for k, v in e.flow.items()},
                )
                if isinstance(e, CpuStep)
                else asdict(e)
            )
            for e in self.events
        ]
        return dict(
            label=self.label,
            frame=self.frame,
            traced=self.traced,
            root=self.root,
            slice=self.keep,
            live=self.live,
            state=[name for name, _, sliced, _ in self.state if sliced],
            events=events,
        )


def sid_cell(player, register: int):
    """The ``(container, index)`` holding a :class:`MemPlayer`'s SID ``register``."""
    return player._mem, player.SID_BASE + register  # pylint: disable=protected-access


def player_roots(player) -> List[str]:
    """The source roots of ``player``'s own packages: every class in its MRO but ``object``.

    A :class:`~pysidtracker.player.MemPlayer` adds only the base's module, for its stores.
    """
    from .player import MemPlayer

    roots = set()
    for cls in type(player).__mro__:
        if cls.__module__.partition(".")[0] in sys.stdlib_module_names | {"builtins"}:
            continue
        if cls is MemPlayer:
            roots.add(inspect.getsourcefile(MemPlayer))
            continue
        path = sys.modules[cls.__module__.partition(".")[0]].__file__
        roots.add(os.path.dirname(path) if path.endswith("__init__.py") else path)
    return sorted(roots)


_LEAVES = (str, bytes, bytearray, memoryview, np.ndarray, numbers.Number, type(None))
_UNSET = -(1 << 62)


def _children(obj):
    """``((kind, key), child)`` of an object graph node: attributes, items, values."""
    if isinstance(obj, _LEAVES + _HIDDEN):
        return
    if isinstance(obj, (list, tuple)):
        yield from ((("S", i), c) for i, c in enumerate(obj))
    elif isinstance(obj, dict):
        yield from ((("S", k), c) for k, c in obj.items() if isinstance(k, (int, str)))
    yield from ((("A", k), v) for k, v in getattr(obj, "__dict__", {}).items())


def object_path(root, target, depth: int = 5) -> Optional[tuple]:
    """The ``(kind, key)`` steps from ``root`` to ``target``, breadth first, else None."""
    if root is target:
        return ()
    seen, frontier = {id(root)}, [(root, ())]
    for _ in range(depth):
        nxt = []
        for obj, path in frontier:
            for step, child in _children(obj):
                if id(child) in seen:
                    continue
                if child is target:
                    return path + (step,)
                seen.add(id(child))
                nxt.append((child, path + (step,)))
        frontier = nxt
    return None


def _resolve(obj, path):
    for kind, key in path:
        obj = getattr(obj, key) if kind == "A" else obj[key]
    return obj


def _family(player, path) -> Optional[tuple]:
    """``(sibling paths, own index)`` over ``path``'s last list index, else None."""
    for at in range(len(path) - 1, -1, -1):
        kind, key = path[at]
        if kind == "S" and isinstance(key, int):
            try:
                size = len(_resolve(player, path[:at]))
            except (AttributeError, LookupError, TypeError):
                return None
            if size < 2:
                return None
            head, tail = path[:at], path[at + 1 :]
            return tuple(head + (("S", j),) + tail for j in range(size)), key
    return None


def fit_family(siblings, own: int, ram, bases, span: int = 256) -> int:
    """The cell of sibling ``own`` in the strided RAM layout the most sibling frames fit.

    ``siblings`` ``(n, s)`` are a list's members' histories, ``ram`` ``(n, 65536)`` the
    6502's; every ``base`` cell (for ``own``) and stride up to ``span`` is scored.
    """
    bases = np.asarray(sorted(set(bases)), np.int64)
    offsets = np.arange(siblings.shape[1]) - own
    best = (-1, 0, bases[0])
    for stride in (*range(1, span + 1), *range(-span, 0)):
        cells = bases[:, None] + stride * offsets[None, :]
        ok = ((cells >= 0) & (cells < ram.shape[1])).all(1)
        if not ok.any():
            continue
        score = (ram[:, cells[ok]] == siblings[:, None, :]).sum((0, 2))
        k = int(np.argmax(score))
        if (score[k], -abs(stride)) > best[:2]:
            best = (int(score[k]), -abs(stride), bases[ok][k])
    return int(best[2])


def path_name(path) -> str:
    """``path`` as an expression on the player."""
    return "player" + "".join(
        f".{k}" if kind == "A" else f"[{k!r}]" for kind, k in path
    )


def _python_history(make_player, paths, frames: int) -> np.ndarray:
    """Each of ``paths``' integer value after each of ``frames`` play calls."""
    player, rows = make_player(), []
    for _ in range(frames):
        player.play_frame()
        row = []
        for path in paths:
            try:
                value = _resolve(player, path)
            except (AttributeError, LookupError, TypeError):
                value = None
            row.append(int(value) if isinstance(value, numbers.Integral) else _UNSET)
        rows.append(row)
    return np.asarray(rows, np.int64).reshape(frames, len(paths))


def _cpu_history(data, subtune: int, addrs, frames: int) -> np.ndarray:
    """Each of ``addrs``' byte after each of ``frames`` jennings play calls."""
    image = data if isinstance(data, SidImage) else SidImage.from_bytes(bytes(data))
    subject, mpu, mem, play = boot_flat(image, subtune)
    rows = np.empty((frames, len(addrs)), np.int64)
    for f in range(frames):
        run_to_rts(mpu, mem, play, 0, 8_000_000)
        rows[f] = [subject[a] for a in addrs]
    return rows


def _live_cells(steps) -> List[int]:
    """The memory cells a call read before writing them: the state it took in."""
    written, live = set(), {}
    for step in steps:
        for addr, _ in step.reads:
            if addr not in written:
                live.setdefault(addr, None)
        written.update(a for a, _ in step.writes)
    return list(live)


def first_break(x, y) -> int:
    """The first index from which ``x`` and ``y`` stop tracking each other (``len`` if never).

    They track as equals when they start equal, else under a one-to-one value map.
    """
    x, y = np.asarray(x), np.asarray(y)
    if x[0] == y[0]:
        bad = x != y
    else:
        _, fx, ix = np.unique(x, return_index=True, return_inverse=True)
        _, fy, iy = np.unique(y, return_index=True, return_inverse=True)
        bad = (y != y[fx[ix]]) | (x != x[fy[iy]])
    return int(np.argmax(bad)) if bad.any() else len(x)


def last_change(grid, frame: int, register: int) -> Optional[int]:
    """The last frame ``<= frame`` whose ``register`` differs from the frame before.

    None when no frame changed it: then nothing earlier explains its value.
    """
    col = np.asarray(grid)[: frame + 1, register]
    moved = np.flatnonzero(col[1:] != col[:-1])
    return int(moved[-1]) + 1 if moved.size else None


def python_side(
    make_player: Callable[[], object],
    frame: int,
    register: int,
    *,
    label: str = "python",
    cell: Callable = sid_cell,
    roots: Optional[Sequence] = None,
    sources: Optional[Dict[str, str]] = None,
    grid=None,
    progress: bool = False,
    traced: Optional[int] = None,
) -> Side:
    """The :class:`PyTrace` of ``make_player()``'s ``play_frame`` call ``frame`` (0-based).

    ``cell(player, register)`` names the ``(container, index)`` holding the register; its
    last store is the root. Without one, ``grid`` (this player's) retraces :func:`last_change`.
    """
    traced = frame if traced is None else traced
    player = make_player()
    steps = range(traced)
    if progress:
        from tqdm import tqdm

        steps = tqdm(steps, desc=f"{label} frames", unit="frame")
    for _ in steps:
        player.play_frame()
    write = getattr(type(player), "_wr", None)
    with PyTrace(
        roots or player_roots(player), [write.__code__] if write else [], sources
    ) as trace:
        player.play_frame()
    root = trace.store(*cell(player, register))
    back = None if grid is None else last_change(grid, traced, register)
    if root is None and back not in (None, traced):
        return python_side(
            make_player,
            frame,
            register,
            label=label,
            cell=cell,
            roots=roots,
            sources=sources,
            progress=progress,
            traced=back,
        )
    keep = trace.slice(root) if root is not None else []
    events = trace.events
    state: Dict[tuple, bool] = {}
    sliced = set(keep)
    for i, event in enumerate(events):
        for _, kind, (obj_id, key) in event.state:
            obj = trace.objects.get(obj_id)
            path = None if obj is None else object_path(player, obj)
            if path is not None:
                loc = path + ((kind, key),)
                state[loc] = state.get(loc, False) or i in sliced

    def retrace(at, path, column):
        return python_side(
            make_player,
            at,
            0,
            label=label,
            cell=lambda p, _: (_resolve(p, path[:-1]), path[-1][1]),
            roots=roots,
            sources=sources,
            grid=column,
        )

    return Side(
        label,
        frame,
        traced,
        events,
        [py_line(e) for e in events],
        root,
        keep,
        {t: v for i in keep for t, v in events[i].live},
        [(path_name(p), p, sliced, _family(player, p)) for p, sliced in state.items()],
        lambda paths, n: _python_history(make_player, paths, n),
        retrace,
    )


def cpu_side(
    data,
    frame: int,
    register: int,
    *,
    label: str = "6502",
    subtune: int = 0,
    grid=None,
    traced: Optional[int] = None,
    address: Optional[int] = None,
) -> Side:
    """The jennings :func:`cpu_history` of play call ``frame``, sliced from its last store
    to SID ``register`` (or RAM ``address``, ``grid`` then its column 0); without one,
    ``grid`` (the jennings grid) retraces :func:`last_change`.
    """
    traced = frame if traced is None else traced
    steps = cpu_history(data, traced, subtune=subtune)
    if address is None:
        hits = [
            i
            for i, s in enumerate(steps)
            if any(r == register for r, _ in s.sid_writes())
        ]
    else:
        hits = [
            i for i, s in enumerate(steps) if any(a == address for a, _ in s.writes)
        ]
    col = register if address is None else 0
    back = None if grid is None else last_change(grid, traced, col)
    if not hits and back not in (None, traced):
        return cpu_side(
            data,
            frame,
            register,
            label=label,
            subtune=subtune,
            traced=back,
            address=address,
        )
    root = hits[-1] if hits else None
    keep, live = cpu_slice(steps, root) if root is not None else ([], {})
    return Side(
        label,
        frame,
        traced,
        steps,
        [cpu_line(s) for s in steps],
        root,
        keep,
        {_loc(k): "entry" if v is None else f"${v:02X}" for k, v in live.items()},
        [(_loc(a), a, a in live, None) for a in _live_cells(steps)],
        lambda addrs, n: _cpu_history(data, subtune, addrs, n),
        lambda at, addr, column: cpu_side(
            data, at, 0, label=label, subtune=subtune, grid=column, address=addr
        ),
        lambda n: _cpu_history(data, subtune, range(0x10000), n),
    )


@dataclass
class Divergence:
    """A diverging cell and two sides' execution of the frame it is in.

    ``frame`` indexes the reference grid; ``values`` each grid's value at the cell, and
    ``notes`` how the sides' frames align to it.
    """

    name: str
    frame: int
    register: int
    values: Dict[str, int]
    notes: List[str]
    window: str
    left: Side
    right: Side
    cell: str = ""
    state: Optional["StateOnset"] = None

    def header(self) -> List[str]:
        """The cell, every grid's value at it, the frame alignment and the grid window."""
        values = ", ".join(f"{k} ${v & 0xFF:02X}" for k, v in self.values.items())
        cell = self.cell or (
            f"register ${self.register:02X} {gridcmp.REG_NAMES[self.register]}"
        )
        return [
            f"divergence: {self.name}, reference frame {self.frame}, {cell}",
            f"  {values}",
            *(f"  {n}" for n in self.notes),
            "",
            self.window,
        ]

    def text(self, width: int = 110) -> str:
        """The header, both sides' dataflow of the value, then both full histories."""
        a, b = self.left, self.right
        rule = f"{'-' * width}-+-{'-' * width}"
        title = f"{a.label.upper():<{width}} | {b.label.upper()}"
        rows = self.header() + [
            "",
            "WHY: the steps the diverging value came from",
            title,
            rule,
        ]
        rows += _columns(
            [a.lines[i] for i in a.keep] or ["(not stored on the traced frame)"],
            [b.lines[i] for i in b.keep] or ["(not stored on the traced frame)"],
            width,
            [len(a.keep) - 1] if a.keep else (),
            [len(b.keep) - 1] if b.keep else (),
        )
        rows += ["", "state read from earlier frames", rule]
        rows += _columns(
            [f"{t} = {v}" for t, v in a.live.items()],
            [f"{t} = {v}" for t, v in b.live.items()],
            width,
        )
        if self.state is not None:
            rows += ["", *self.state.lines(width)]
        rows += [
            "",
            f"FULL HISTORIES, meeting at the diverging store: {a.label} {len(a.lines)} "
            f"steps, {b.label} {len(b.lines)} steps; '>>' the dataflow above",
            title,
            rule,
        ]
        ar = len(a.lines) - 1 if a.root is None else a.root
        br = len(b.lines) - 1 if b.root is None else b.root
        for lo, hi, clo, chi, top in (
            (0, ar + 1, 0, br + 1, True),
            (ar + 1, len(a.lines), br + 1, len(b.lines), False),
        ):
            rows += _columns(
                a.lines[lo:hi],
                b.lines[clo:chi],
                width,
                [i - lo for i in a.keep if lo <= i < hi],
                [i - clo for i in b.keep if clo <= i < chi],
                top,
            )
        return "\n".join(rows)

    def summary(self) -> dict:
        """The machine-readable form for ``json.dumps``."""
        return dict(
            name=self.name,
            frame=self.frame,
            register=self.register,
            register_name=gridcmp.REG_NAMES[self.register],
            values=self.values,
            notes=self.notes,
            window=self.window,
            left=self.left.summary(),
            right=self.right.summary(),
            cell=self.cell,
            state=None if self.state is None else self.state.summary(),
        )


@dataclass
class StateOnset:
    """When the state a divergence read from earlier frames first disagreed.

    ``pairs`` are ``(left state, right state, frame)``: each left state's best-tracking
    right state and the left frame they part (None if they never do); ``why`` explains
    the earliest such frame, from the stores of that pair.
    """

    frames: Tuple[int, int]
    pairs: List[Tuple[str, str, Optional[int]]]
    why: Optional["Divergence"] = None

    def lines(self, width: int = 110) -> List[str]:
        """The pairing table, then the earliest parting explained."""
        lo, hi = self.frames
        rows = [
            f"STATE: the state read from earlier frames, tracked over frames {lo}-{hi}"
        ]
        rows += [
            f"  {a} ~ {b}: " + ("tracks" if f is None else f"parts at frame {f}")
            for a, b, f in self.pairs
        ]
        if self.why is not None:
            rows += ["", "STATE ONSET " + "=" * 60, self.why.text(width)]
        return rows

    def summary(self) -> dict:
        """The machine-readable form for ``json.dumps``."""
        return dict(
            frames=list(self.frames),
            pairs=[list(p) for p in self.pairs],
            why=None if self.why is None else self.why.summary(),
        )


def explain_state(why: "Divergence", depth: int = 4) -> Optional[StateOnset]:
    """Pair the earlier-frame state each side of ``why`` read and explain where it parts.

    Each state the left slice read is paired with a right one over the frames before the
    divergence: a list member against a 6502 by :func:`fit_family`, else the right
    frame's state that starts equal, tracks longest (:func:`first_break`), shares its
    name, agrees most, and is in the right slice. The earliest parting is explained from
    both stores, and chased again up to ``depth``.
    """
    a, b = why.left, why.right
    asked = [(name, loc, family) for name, loc, sliced, family in a.state if sliced]
    pool = b.state
    n = min(a.frame, b.frame)
    if not (asked and pool and a.history and b.history and n):
        return None
    lfull = a.history([loc for _, loc, _ in asked], a.frame)
    rfull = b.history([loc for _, loc, *_ in pool], b.frame)
    ram = b.ram(b.frame) if b.ram and any(f for *_, f in asked) else None
    lh, rh = lfull[a.frame - n :], rfull[b.frame - n :]
    pairs, best, partners = [], None, []
    for i, (name, _, family) in enumerate(asked):
        if family and ram is not None:
            sib = a.history(family[0], a.frame)[a.frame - n :]
            addr = fit_family(sib, family[1], ram[b.frame - n :], [p[1] for p in pool])
            partner = (_loc(addr), addr, ram[:, addr].astype(np.int64))
        else:
            same, agree = lh[0, i] == rh[0], (lh[:, [i]] == rh).sum(0)
            ranks = [
                (
                    same[j],
                    first_break(lh[:, i], rh[:, j]),
                    pool[j][0] == name,
                    agree[j],
                    pool[j][2],
                )
                for j in range(len(pool))
            ]
            j = max(range(len(pool)), key=ranks.__getitem__)
            partner = (pool[j][0], pool[j][1], rfull[:, j])
        partners.append(partner)
        cut = first_break(lh[:, i], partner[2][b.frame - n :])
        at = cut if cut < n else None
        pairs.append((name, partner[0], None if at is None else a.frame - n + at))
        if at is not None and (best is None or at < best[0]):
            best = (at, i)
    onset = StateOnset((a.frame - n, a.frame - 1), pairs)
    if best is None:
        return onset
    at, i = best
    pname, ploc, pcol = partners[i]
    theirs = pcol[b.frame - n :]
    lf, rf = a.frame - n + at, b.frame - n + at
    left = a.retrace(lf, asked[i][1], lfull[:, [i]])
    right = b.retrace(rf, ploc, pcol[:, None])
    window = "\n".join(
        f"{a.frame - n + f:6d} {a.label} {lh[f, i]}  {b.label} {theirs[f]}"
        + ("  ^^" if f >= at else "")
        for f in range(max(0, at - 3), min(n, at + 4))
    )
    onset.why = Divergence(
        name=why.name,
        frame=lf,
        register=why.register,
        values={a.label: int(lh[at, i]), b.label: int(theirs[at])},
        notes=[f"{a.label} frame {lf}, {b.label} frame {rf}"],
        window=window,
        left=left,
        right=right,
        cell=f"state {asked[i][0]} ~ {pname}",
    )
    if depth > 1:
        onset.why.state = explain_state(onset.why, depth - 1)
    return onset


def explain_cell(
    left: Side,
    right: Side,
    grids: Tuple,
    *,
    frame: int,
    register: int,
    name: str = "",
    notes: Sequence[str] = (),
) -> Divergence:
    """A :class:`Divergence` of two sides at reference cell ``(frame, register)``.

    ``grids`` are the ``(left, right)`` renders, the right one the reference, aligned at
    the sides' frames.
    """
    lg, rg = (np.asarray(g, np.int16) for g in grids)
    shift = left.frame - frame
    ref, lib = gridcmp.aligned(rg, lg, shift)
    return Divergence(
        name=name or "tune",
        frame=frame,
        register=register,
        values={
            left.label: int(lg[left.frame, register]),
            right.label: int(rg[right.frame, register]),
        },
        notes=[
            ", ".join(
                f"{s.label} frame {s.frame}"
                + (
                    ""
                    if s.traced == s.frame
                    else f" (traced {s.traced}: its last store)"
                )
                for s in (left, right)
            ),
            *notes,
        ],
        window=gridcmp.window_report(ref, lib, frame),
        left=left,
        right=right,
    )


def explain_divergence(
    data,
    make_player: Callable[[bytes], object],
    reference=None,
    *,
    song: int = 1,
    max_lead: int = 0,
    frames: int = 250,
    frame: Optional[int] = None,
    register: Optional[int] = None,
    name: str = "",
    label: str = "python",
    cell: Callable = sid_cell,
    roots: Optional[Sequence] = None,
    sources: Optional[Dict[str, str]] = None,
    progress: bool = False,
    state: bool = False,
) -> Optional[Divergence]:
    """The first divergence of ``make_player(data)``'s render from the 6502's, explained.

    ``reference`` is the expected grid (default ``frames`` of jennings'), compared under
    :func:`~pysidtracker.gridcmp.verdict_lead` up to ``max_lead``; ``frame`` and
    ``register`` pick a reference cell instead of the first mismatch. None on agreement.
    ``label``/``cell``/``roots``/``sources`` are :func:`python_side`'s; ``state`` adds
    :func:`explain_state`.
    """
    data = bytes(data)
    nref = len(reference) if reference is not None else max(frames, (frame or 0) + 1)
    jen = np.asarray(register_grid(data, nref + max_lead, subtune=song - 1), np.int16)
    ref = jen[:nref] if reference is None else np.asarray(reference, np.int16)
    lib = np.asarray(make_player(data).render_grid(len(ref) + max_lead), np.int16)
    found = gridcmp.verdict_lead(ref, lib, max_lead=max_lead)
    cpu = gridcmp.verdict_lead(ref, jen, max_lead=max_lead)
    if frame is None:
        if found.ok:
            return None
        frame, register = found.first, found.first_reg
    if register is None:
        bad = np.flatnonzero(ref[frame] != lib[frame + found.lead])
        register = int(bad[0]) if bad.size else 0
    agrees = cpu.ok or cpu.first > frame
    why = explain_cell(
        python_side(
            lambda: make_player(data),
            frame + found.lead,
            register,
            label=label,
            cell=cell,
            roots=roots,
            sources=sources,
            grid=lib,
            progress=progress,
        ),
        cpu_side(data, frame + cpu.lead, register, subtune=song - 1, grid=jen),
        (lib, jen),
        frame=frame,
        register=register,
        name=name,
        notes=[
            f"jennings {'agrees' if agrees else 'DOES NOT agree'} with the reference "
            "to here",
            gridcmp.lead_report(found),
        ],
    )
    why.values["reference"] = int(ref[frame, register])
    why.window = gridcmp.window_report(*gridcmp.aligned(ref, lib, found.lead), frame)
    if state:
        why.state = explain_state(why)
    return why


def main(argv=None) -> int:
    """``python -m pysidtracker.whydiff``: explain a render's first divergence.

    The player is ``module:callable(data, song) -> MemPlayer``, or an installed format's
    (``--format NAME``, or ``auto``; its parser's default song). The reference is a
    ``gridcmp`` grid file (``--reference``) or the jennings grid. Exit 0 on agreement.
    """
    parser = argparse.ArgumentParser(description="Explain a render's first divergence.")
    parser.add_argument("tune", type=Path)
    parser.add_argument(
        "player", nargs="?", help="module:callable(data, song) -> MemPlayer"
    )
    parser.add_argument("--format", help="an installed format's player, or auto")
    parser.add_argument(
        "--state",
        action="store_true",
        help="chase the state read back to where it parts",
    )
    parser.add_argument("--song", type=int, default=1)
    parser.add_argument("--frames", type=int, default=250)
    parser.add_argument("--reference", type=Path, help="a gridcmp grid file")
    parser.add_argument("--max-lead", type=int, default=0)
    parser.add_argument("--frame", type=int, help="reference frame to explain")
    parser.add_argument("--register", type=lambda s: int(s, 0), help="SID offset")
    parser.add_argument("--width", type=int, default=110)
    parser.add_argument("--json", dest="json_path", type=Path)
    args = parser.parse_args(argv)
    data = args.tune.read_bytes()
    if args.format:
        from .formats import discover_formats, recognize_format

        found = discover_formats()
        fmt = (
            recognize_format(found, data)
            if args.format == "auto"
            else next((f for f in found if f.name == args.format), None)
        )
        if fmt is None:
            parser.error(f"no installed format {args.format!r} plays {args.tune}")

        def make(d, _song):
            return fmt.player(fmt.parser.parse(d))

    elif args.player:
        module, _, attr = args.player.partition(":")
        make = getattr(importlib.import_module(module), attr)
    else:
        parser.error("give a player or --format")
    if args.reference:
        ref = gridcmp.read_grid(args.reference, args.frames)
    else:
        ref = register_grid(data, args.frames, subtune=args.song - 1)
    why = explain_divergence(
        data,
        lambda d: make(d, args.song),
        ref,
        song=args.song,
        max_lead=args.max_lead,
        frame=args.frame,
        register=args.register,
        name=args.tune.name,
        progress=True,
        state=args.state,
    )
    if why is None:
        print(f"verified: {args.frames} frames agree")
        return 0
    print(why.text(args.width))
    if args.json_path:
        args.json_path.write_text(json.dumps(why.summary()), encoding="utf-8")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
