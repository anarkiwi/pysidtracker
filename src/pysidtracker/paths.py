"""Per-frame code paths of a player, each side traced alone, for fault localisation.

:func:`cpu_paths` (jennings) and :func:`py_paths` (a Python player) return a :class:`PathTrace`
of the edges and store sites each play call ran, keyed stably across tunes;
:func:`cpu_universe` and :func:`py_universe` list the static edges keyed alike.
"""

from __future__ import annotations

import ast
import functools
import inspect
import linecache
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import registers as reg
from .gridcmp import REG_NAMES
from .image import SidImage
from .mos6502 import OP_LEN
from .player import MemPlayer
from .variants import _decode, _digest
from .whydiff import player_roots, sid_cell

_SID = reg.SID_BASE
_NREG = reg.SID_REG_COUNT
_CF_MNEMONICS = frozenset(
    "BPL BMI BVC BVS BCC BCS BNE BEQ JSR JMP RTS RTI BRK JAM".split()
)
_OUT = -1


@dataclass
class PathTrace:
    """The features one side executed, per play call.

    Attributes:
      side: the side's label (``"6502"``, or a :class:`PySubject`'s ``label``).
      names: feature id to its cross-tune key, ``edge:...`` or ``store:<reg>:...``.
      labels: feature id to its text in this tune (disassembly or source line).
      frames: per play call, the sorted unique ``int32`` feature ids it executed.
      order: per play call, aligned with ``frames``, each feature's ``int32``
        first-occurrence rank in that call (0: executed first); empty when not recorded.
      grid: ``(nframes, 25)`` registers after each play call.
      sites: ``(nframes, 25)`` ``store:`` id of each call's last write to a register, -1 none.
      reads: per play call, the sorted unique data addresses it read.
      roots: the files a Python side traced (for :func:`py_universe`).
      sources: the exec'd code a Python side traced (for :func:`py_universe`).
      starts: each voice's first-note frame (None: not begun), when the side gives them.
      code: a 6502 side's :class:`StaticCode`.
      edges: a 6502 side's ``(src, dst)`` of each ``edge:`` feature id (-1: out).
    """

    side: str
    names: List[str]
    labels: List[str]
    frames: List[np.ndarray]
    grid: np.ndarray
    sites: np.ndarray
    reads: List[np.ndarray]
    roots: List[str] = field(default_factory=list)
    sources: Dict[str, str] = field(default_factory=dict)
    starts: Optional[tuple] = None
    order: List[np.ndarray] = field(default_factory=list)
    code: Optional["StaticCode"] = None
    edges: List[Tuple[int, int]] = field(default_factory=list)

    def sequence(self, frame: int) -> np.ndarray:
        """Play call ``frame``'s feature ids in the order they first executed."""
        return self.frames[frame][np.argsort(self.order[frame], kind="stable")]

    def matrix(self, upto: Optional[int] = None):
        """``(frames[:upto], len(names))`` boolean CSR matrix of executed features."""
        from scipy.sparse import csr_matrix

        rows = self.frames[:upto]
        indptr = np.zeros(len(rows) + 1, np.int64)
        np.cumsum([len(r) for r in rows], out=indptr[1:])
        indices = np.concatenate(rows) if rows else np.zeros(0, np.int32)
        return csr_matrix(
            (np.ones(len(indices), bool), indices, indptr),
            shape=(len(rows), len(self.names)),
        )


def _split(keys: np.ndarray, nframes: int, width: int) -> List[np.ndarray]:
    """Per-frame sorted unique values of ``frame * width + value`` ``keys``."""
    keys = np.unique(keys)
    bounds = np.searchsorted(keys, np.arange(nframes + 1, dtype=np.int64) * width)
    vals = keys % width
    return [vals[bounds[f] : bounds[f + 1]] for f in range(nframes)]


def _first(frame, fid, when, nframes: int, width: int):
    """``(frames, order)``: per frame its sorted unique feature ids among events
    ``(frame, fid)`` at times ``when``, and each one's rank by first time in the frame.
    """
    key = np.asarray(frame, np.int64) * width + fid
    srt = np.lexsort((when, key))
    key, when = key[srt], np.asarray(when, np.int64)[srt]
    head = np.ones(len(key), bool)
    head[1:] = key[1:] != key[:-1]
    uk, first = key[head], when[head]
    ufr = uk // width
    bounds = np.searchsorted(ufr, np.arange(nframes + 1))
    by_time = np.lexsort((first, ufr))
    rank = np.empty(len(uk), np.int64)
    rank[by_time] = np.arange(len(uk))
    rank -= bounds[ufr]
    vals = (uk % width).astype(np.int32)
    rank = rank.astype(np.int32)
    return (
        [vals[bounds[f] : bounds[f + 1]] for f in range(nframes)],
        [rank[bounds[f] : bounds[f + 1]] for f in range(nframes)],
    )


# --- 6502 -------------------------------------------------------------------

_OPLEN = np.asarray(OP_LEN, np.int64)


@functools.lru_cache(maxsize=1)
def _opcf() -> np.ndarray:
    from jennings.opcodes import OPCODES

    return np.array([OPCODES[op][0] in _CF_MNEMONICS for op in range(256)])


class StaticCode:
    """The static control flow of a post-init memory snapshot's span ``[lo, hi)``.

    Descent follows fall-through, both branch arms, ``JMP abs`` and ``JSR`` (and its
    return site); an ``RTS`` returns to its routine's callers, or out for an entry.
    ``smc`` holds the span's addresses the program wrote: an operand among them is data,
    as is an immediate.
    """

    def __init__(self, mem: bytes, lo: int, hi: int, labels: Dict[str, int], smc=()):
        self.mem, self.lo, self.hi = mem, lo, hi
        self._labels = labels
        self.smc = set(smc)
        self._insn: Dict[int, Tuple[str, str, int, tuple]] = {}
        self.pred: Dict[int, int] = {}
        self.seen: set = set()
        self.calls: Dict[int, int] = {}
        self.edges: set = set()
        self._dis = None

    def inspan(self, pc: int) -> bool:
        return self.lo <= pc < self.hi

    def insn(self, pc: int) -> Tuple[str, str, int, tuple]:
        """``(mnemonic, mode, length, normalised form)`` of the instruction at ``pc``."""
        got = self._insn.get(pc)
        if got is None:
            length, mn, mode, operand = _decode(self.mem, pc, self.lo, self.lo, self.hi)
            if operand.startswith("rel:"):
                operand = "span"
            if mode == "imm" or self.smc.intersection(range(pc + 1, pc + length)):
                operand = "data"
            got = self._insn[pc] = (mn, mode, length, (mn, mode, operand))
        return got

    def _target(self, pc: int) -> int:
        """The target of the branch or absolute jump/call at ``pc``."""
        _, mode, length, _ = self.insn(pc)
        if mode == "rel":
            raw = self.mem[pc + 1]
            return (pc + length + (raw - 0x100 if raw & 0x80 else raw)) & 0xFFFF
        return self.mem[pc + 1] | self.mem[pc + 2] << 8

    def _succ(self, pc: int) -> List[int]:
        """Intra-routine successors (a call continues at its return site)."""
        mn, mode, length, _ = self.insn(pc)
        if mn not in _CF_MNEMONICS or mn == "JSR":
            return [pc + length]
        if mode == "rel":
            return [self._target(pc), pc + length]
        if mn == "JMP" and mode == "abs":
            return [self._target(pc)]
        return []

    def cfg(self, out) -> List[Tuple[int, object]]:
        """The decoded code's intra-routine edges; a routine's ``RTS``/``RTI`` (and a
        ``BRK``, ``JAM`` or indirect jump) leads to ``out``."""
        return [(pc, s) for pc in sorted(self.seen) for s in self._succ(pc) or [out]]

    def explore(self, seeds, entries) -> None:
        """Descend from ``seeds``, adding decoded runs and static edges.

        A later descent extends the runs only into code an earlier one had not seen.
        """
        seen = self.seen
        known = set(seen)
        stack = sorted(s for s in seeds if self.inspan(s))
        while stack:
            pc = stack.pop()
            while self.inspan(pc) and pc not in seen:
                seen.add(pc)
                mn, mode, length, _ = self.insn(pc)
                if mn not in _CF_MNEMONICS:
                    pc += length
                    continue
                targets = [self._target(pc)] if mode in ("rel", "abs") else []
                targets += [pc + length] if mode == "rel" else []
                if mn == "JSR" and self.inspan(pc + length):
                    self.calls[pc] = targets[0]
                    stack.append(pc + length)
                    if not self.inspan(targets[0]):
                        self.edges.add((_OUT, pc + length))
                for t in targets:
                    self.edges.add((pc, t if self.inspan(t) else _OUT))
                    stack.append(t)
                break
        for pc in sorted(seen - known):
            mn, _, length, _ = self.insn(pc)
            if mn not in _CF_MNEMONICS and pc + length in seen - known:
                self.pred.setdefault(pc + length, pc)
        self._returns(entries)

    def _returns(self, entries) -> None:
        """Add each routine's ``RTS`` edges to its callers' return sites."""
        rets: Dict[int, List[int]] = {}

        def ends(entry: int) -> List[int]:
            if entry not in rets:
                todo, done, out = [entry], set(), []
                while todo:
                    pc = todo.pop()
                    if pc in done or pc not in self.seen:
                        continue
                    done.add(pc)
                    if self.insn(pc)[0] == "RTS":
                        out.append(pc)
                    todo += self._succ(pc)
                rets[entry] = out
            return rets[entry]

        for site, target in self.calls.items():
            self.edges.update((r, site + 3) for r in ends(target))
        for entry in entries:
            self.edges.update((r, _OUT) for r in ends(entry))

    def prefix(self, pc: int) -> tuple:
        """Normalised straight-line run of decoded instructions falling into ``pc``."""
        run = [pc]
        while run[-1] in self.pred:
            run.append(self.pred[run[-1]])
        return tuple(self.insn(p)[3] for p in reversed(run))

    def head(self, pc: int) -> tuple:
        """Normalised straight-line run from ``pc`` through its first control flow."""
        run = [self.insn(pc)]
        while run[-1][0] not in _CF_MNEMONICS and self.inspan(pc + run[-1][2]):
            pc += run[-1][2]
            run.append(self.insn(pc))
        return tuple(i[3] for i in run)

    def edge_digest(self, src: int, dst: int) -> str:
        return _digest(
            [
                self.prefix(src) if src != _OUT else ("out",),
                self.head(dst) if dst != _OUT else ("out",),
            ]
        )

    def text(self, pc: int) -> str:
        """``$PC disassembly`` of the instruction at ``pc``."""
        if self._dis is None:
            from jennings.devices.mpu6502 import MPU
            from jennings.disassembler import Disassembler
            from jennings.utils.addressing import AddressParser

            self._dis = Disassembler(
                MPU(memory=bytearray(self.mem)),
                AddressParser(labels=dict(self._labels)),
            )
        return f"${pc:04X} {self._dis.instruction_at(pc)[1]}"

    def edge_names(self, edges=()) -> Dict[tuple, str]:
        """Each static edge and each of ``edges`` (``(src, dst)``, -1: out) to its
        :func:`cpu_paths` name."""
        return _ranked(
            {
                (s, d): "edge:" + self.edge_digest(s, d)
                for s, d in self.edges | set(edges)
            }
        )

    def edge_label(self, src: int, dst: int) -> str:
        """``$SRC disassembly -> $DST`` (``out`` for an end outside the span)."""
        left = self.text(src) if src != _OUT else "out"
        return f"{left} -> " + (f"${dst:04X}" if dst != _OUT else "out")


def _ranked(keys: Dict[tuple, str]) -> Dict[tuple, str]:
    """Each site's key, ``#k`` appended to the k-th (by site order) repeat of a key."""
    seen: Dict[str, int] = {}
    out = {}
    for site in sorted(keys):
        k = seen.get(keys[site], 0)
        seen[keys[site]] = k + 1
        out[site] = keys[site] + (f"#{k}" if k else "")
    return out


def _store_names(code: StaticCode, store_sites) -> List[str]:
    keys = {
        (r, pc): f"store:{r}:"
        + (_digest(["store", code.prefix(pc)]) if code.inspan(pc) else "out")
        for pc, r in store_sites
    }
    ranked = _ranked(keys)
    return [ranked[(r, pc)] for pc, r in store_sites]


def _booted(data, subtune: int, max_cycles: int):
    from .disasm import entry_labels
    from .emu import observe_data_access
    from .errors import SidParseError
    from .oracle import boot_flat

    image = data if isinstance(data, SidImage) else SidImage.from_bytes(bytes(data))
    if image.header is None:
        raise SidParseError("cannot trace paths: image has no SID header")
    lo, hi = image.load, image.end
    smc: set = set()

    def setup(mpu, _subject):
        observe_data_access(mpu, on_write=lambda a, _v: lo <= a < hi and smc.add(a))

    subject, mpu, mem, play = boot_flat(image, subtune, max_cycles, setup=setup)
    init = image.header.init_address or image.header.real_load_address
    code = StaticCode(bytes(subject), lo, hi, entry_labels(image), smc)
    return subject, mpu, mem, play, code, (init, play)


def cpu_universe(data, *, subtune: int = 0, max_cycles: int = 8_000_000) -> List[str]:
    """Names of every 6502 edge statically reachable from ``init`` and ``play``.

    Decoded from memory as ``init`` left it and keyed as :func:`cpu_paths` keys them;
    edges of indirect jumps, ``RTI``/``BRK`` and code reached only through them are
    absent, as is a ``#k`` repeat rank that a site found only dynamically would shift;
    an operand only ``play`` rewrites keeps its value.
    """
    return sorted(set(static_code(data, subtune, max_cycles).edge_names().values()))


def static_code(data, subtune: int = 0, max_cycles: int = 8_000_000) -> StaticCode:
    """The tune's :class:`StaticCode`, explored from ``init`` and ``play`` as ``init``
    left memory."""
    *_, code, entries = _booted(data, subtune, max_cycles)
    code.explore(entries, entries)
    return code


def cpu_paths(
    data,
    nframes: int,
    *,
    subtune: int = 0,
    max_cycles: int = 8_000_000,
    progress: bool = False,
) -> PathTrace:
    """The edges and SID store sites of each of ``nframes`` play calls on jennings.

    Booted as :func:`~pysidtracker.oracle.register_grid` is, so ``grid`` equals it.

    Notes:
      An edge is a step from a control-flow instruction, or a step that does not fall
      through, with an end in the loaded span; an end outside it is ``out``. Its name
      digests the normalised block ending at its source and the block starting at its
      destination, decoded from memory as ``init`` left it, absolute in-span operands
      folded to ``span`` so code inserted elsewhere leaves it unchanged; immediates and
      operands ``init`` or a play call wrote (self-modifying code) are folded to ``data``,
      as a packer or the tune patches its own values into them. Repeats of a
      name (voice-duplicated code) take ``#k`` by offset. A store site is keyed by
      its register and the normalised block up to the store. ``order`` ranks each
      feature by the step that first ran it in the call.
    """
    from tqdm import tqdm

    from .emu import observe_data_access, run_to_rts

    subject, mpu, mem, play, code, entries = _booted(data, subtune, max_cycles)
    pcs: List[int] = []
    reads: List[int] = []
    wsteps: List[int] = []
    waddrs: List[int] = []
    pc_bounds, read_bounds = [0], [0]
    grid = np.zeros((nframes, _NREG), np.int16)
    view = np.frombuffer(subject, np.uint8)
    mask = np.full(_NREG, 0xFF, np.int16)
    mask[list(reg.PW_HI_REGS)] = 0x0F

    def on_write(addr, _value):
        if _SID <= addr < _SID + _NREG:
            wsteps.append(len(pcs) - 1)
            waddrs.append(addr - _SID)
        elif code.lo <= addr < code.hi:
            code.smc.add(addr)

    observe_data_access(mpu, on_read=reads.append, on_write=on_write)
    step = mpu.step

    def traced(append=pcs.append):
        append(mpu.pc)
        step()

    mpu.step = traced
    try:
        for f in tqdm(range(nframes), unit="frame", disable=not progress):
            run_to_rts(mpu, mem, play, 0, max_cycles)
            pc_bounds.append(len(pcs))
            read_bounds.append(len(reads))
            grid[f] = view[_SID : _SID + _NREG] & mask
    finally:
        mpu.step = step
        mpu._vm._on_read = mpu._vm._on_write = None  # pylint: disable=protected-access

    p = np.asarray(pcs, np.int64)
    frame_of = np.repeat(np.arange(nframes), np.diff(pc_bounds))
    nxt = np.append(p[1:], _OUT)
    nxt[np.asarray(pc_bounds[1:], np.int64) - 1] = _OUT
    ins = (p >= code.lo) & (p < code.hi)
    nin = (nxt >= code.lo) & (nxt < code.hi)
    op = np.frombuffer(code.mem, np.uint8)[p & 0xFFFF]
    is_edge = np.where(ins, _opcf()[op] | (nxt != p + _OPLEN[op]), nin)
    pair = (np.where(ins, p, _OUT) + 1) * 0x10001 + np.where(nin, nxt, _OUT) + 1
    pairs, pair_id = np.unique(pair[is_edge], return_inverse=True)
    edges = [(int(x) // 0x10001 - 1, int(x) % 0x10001 - 1) for x in pairs]

    ws = np.asarray(wsteps, np.int64)
    wreg = np.asarray(waddrs, np.int64)
    stores, store_id = np.unique(p[ws] * 32 + wreg, return_inverse=True)
    store_sites = [(int(x) // 32, int(x) % 32) for x in stores]

    code.explore(entries, entries)
    code.explore({d for _, d in edges if d != _OUT}, entries)
    names_of = code.edge_names(edges)
    names = [names_of[e] for e in edges] + _store_names(code, store_sites)
    labels = [code.edge_label(*e) for e in edges] + [
        (code.text(pc) if code.inspan(pc) else f"${pc:04X}") + f" -> {REG_NAMES[r]}"
        for pc, r in store_sites
    ]

    store_fid = store_id + len(edges)
    width = len(names) + 1
    wframe = frame_of[ws]
    feats, order = _first(
        np.concatenate([frame_of[is_edge], wframe]),
        np.concatenate([pair_id, store_fid]),
        np.concatenate([np.flatnonzero(is_edge), ws]),
        nframes,
        width,
    )
    sites = np.full((nframes, _NREG), -1, np.int32)
    last = len(ws) - 1 - np.unique((wframe * 32 + wreg)[::-1], return_index=True)[1]
    sites[wframe[last], wreg[last]] = store_fid[last]
    rd = np.asarray(reads, np.int64)
    rframe = np.repeat(np.arange(nframes), np.diff(read_bounds))
    return PathTrace(
        "6502",
        names,
        labels,
        feats,
        grid,
        sites,
        [r.astype(np.uint16) for r in _split(rframe * 0x10000 + rd, nframes, 0x10000)],
        order=order,
        code=code,
        edges=edges,
    )


# --- Python -----------------------------------------------------------------

_NUMBER = re.compile(r"0[xX][0-9a-fA-F]+|\d+")
_EXIT = 0


def _generic(text: str) -> str:
    """``text`` with numbers (also inside generated names) as ``#``, spaces folded."""
    return " ".join(_NUMBER.sub("#", text).split())


@functools.lru_cache(maxsize=None)
def _relname(path: str) -> str:
    """``path`` relative to the parent of its outermost enclosing package."""
    top = os.path.dirname(path)
    while os.path.exists(os.path.join(top, "__init__.py")):
        top = os.path.dirname(top)
    return os.path.relpath(path, top)


def _qualnames(tree: ast.AST) -> Dict[int, str]:
    """Each line to the qualified name of the innermost def or class holding it."""
    out: Dict[int, str] = {}

    def visit(node, qual: str, local: bool) -> None:
        for child in ast.iter_child_nodes(node):
            inner, inner_local = qual, local
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                prefix = f"{qual}.<locals>." if local else (f"{qual}." if qual else "")
                inner = prefix + child.name
                inner_local = not isinstance(child, ast.ClassDef)
                for line in range(child.lineno, child.end_lineno + 1):
                    out[line] = inner
            visit(child, inner, inner_local)

    visit(tree, "", False)
    return out


def _subscripts(node) -> List[ast.Subscript]:
    if isinstance(node, (ast.Tuple, ast.List)):
        return [s for elt in node.elts for s in _subscripts(elt)]
    return [node] if isinstance(node, ast.Subscript) else []


def _stores(tree: ast.AST, name: str) -> Dict[int, list]:
    """Each line of a statement assigning to subscripts to their compiled
    ``(container, index)`` expressions."""
    stores: Dict[int, list] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            continue
        compiled = [
            tuple(compile(ast.Expression(e), name, "eval") for e in (s.value, s.slice))
            for t in targets
            for s in _subscripts(t)
        ]
        for line in range(node.lineno, node.end_lineno + 1) if compiled else ():
            stores.setdefault(line, []).extend(compiled)
    return stores


@dataclass(eq=False)
class _Source:
    """One traced file: its branch lines, line keys and subscript stores."""

    name: str
    generated: bool
    lines: List[str]
    first: Dict[int, int]
    branch: frozenset
    qual: Dict[int, str]
    keys: Dict[int, str]
    stores: Dict[int, list]
    arcs: frozenset
    flow: frozenset = frozenset()

    @functools.cached_property
    def file(self) -> str:
        return _generic(self.name) if self.generated else _relname(self.name)

    def text(self, line: int) -> str:
        return self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""

    def where(self, line: int) -> str:
        first = self.first.get(line, line)
        qual = self.qual.get(first, "<module>")
        qual = _generic(qual) if self.generated else qual
        return f"{self.file}:{qual}:{self.keys.get(first, first)}"

    def arc_key(self, src: int, dst: int) -> str:
        dst = self.first.get(dst, dst)
        to = self.keys.get(dst, dst) if dst > 0 else "exit"
        return f"edge:{self.where(src)}>{to}"


@functools.lru_cache(maxsize=256)
def _source(name: str, text: str, generated: bool) -> Optional[_Source]:
    """Parse one file (or exec'd ``text``) with coverage.py's static arc analysis."""
    from coverage.exceptions import NotPython
    from coverage.parser import PythonParser

    try:
        parser = PythonParser(text=text, filename=name)
        parser.parse_source()
        tree = ast.parse(text)
    except (NotPython, SyntaxError, ValueError):
        return None
    lines = text.splitlines()
    first = {n: parser.first_line(n) for n in range(1, len(lines) + 1)}
    exits = parser.exit_counts()
    qual = _qualnames(tree)
    keys = {n: str(n) for n in set(first.values())}
    if generated:
        groups: Dict[tuple, int] = {}
        for n in sorted(keys):
            base = _generic(lines[n - 1])
            k = groups.get((qual.get(n), base), 0)
            groups[(qual.get(n), base)] = k + 1
            keys[n] = base + (f"@{k}" if k else "")
    return _Source(
        name,
        generated,
        lines,
        first,
        frozenset(n for n, f in first.items() if exits.get(f, 0) > 1),
        qual,
        keys,
        _stores(tree, name),
        frozenset((a, b) for a, b in parser.arcs() if exits.get(a, 0) > 1),
        frozenset(parser.arcs()),
    )


def source_of(path: str, sources: Optional[Dict[str, str]] = None) -> Optional[_Source]:
    """The parsed file at ``path`` (an exec'd pseudo-filename in ``sources``: its text),
    None when it cannot be read or parsed."""
    if sources and path in sources:
        return _source(path, sources[path], True)
    real = os.path.realpath(path)
    try:
        with open(real, encoding="utf-8") as fh:
            text = fh.read()
    except (OSError, UnicodeDecodeError):
        return None
    return _source(real, text, False)


@dataclass
class PySubject:
    """A Python player to trace by :func:`py_paths`.

    Attributes:
      make: a fresh player, whose ``play_frame()`` plays a frame (None once ended).
      cell: ``(player, register)`` to the ``(container, index)`` holding it.
      sources: exec'd code's pseudo-filenames to their text.
      roots: files, packages or pseudo-filenames to trace (default the player's own
        packages, :func:`~pysidtracker.whydiff.player_roots`, and ``sources``).
      label: the trace's ``side``.
      starts: ``player`` to its voices' first-note frames after the frames played,
        for :func:`~pysidtracker.gridcmp.verdict_onset` (None: not given).
    """

    make: Callable[[], object]
    cell: Callable = sid_cell
    sources: Optional[Dict[str, str]] = None
    roots: Optional[Sequence[str]] = None
    label: str = "python"
    starts: Optional[Callable[[object], Sequence]] = None


class _Recorder:
    """Line arcs and subscript stores of the code under ``roots``, per frame.

    An arc steps from a line with more than one static exit (coverage.py's analysis)
    to the next line its frame runs, or out (``exit``); stores log as they start.
    ``arcs`` maps each arc to the event count at its first run, ``log`` entries end
    with theirs.
    """

    def __init__(self, roots: Sequence[str], sources: Dict[str, str]):
        self.sources = sources
        self.roots = tuple(os.path.realpath(r) for r in roots if r not in sources)
        self._skip = os.path.realpath(inspect.getsourcefile(MemPlayer))
        self._codes: Dict[object, Optional[_Source]] = {}
        self._stack: List[list] = []
        self.arcs: Dict[tuple, int] = {}
        self.log: List[tuple] = []
        self.seq = 0

    def arc(self, key: tuple) -> None:
        if key not in self.arcs:
            self.arcs[key] = self.seq
            self.seq += 1

    def store(self, held, at, src: _Source, line: int) -> None:
        self.log.append((held, at, src, line, self.seq))
        self.seq += 1

    def src(self, code) -> Optional[_Source]:
        if code in self._codes:
            return self._codes[code]
        path = code.co_filename
        got = None
        if code.co_name.startswith("<") and code.co_name != "<module>":
            pass
        elif path in self.sources:
            got = source_of(path, self.sources)
        else:
            real = os.path.realpath(path)
            if real != self._skip and any(
                real == r or real.startswith(r + os.sep) for r in self.roots
            ):
                got = source_of(real)
        self._codes[code] = got
        return got

    def start(self, code) -> bool:
        src = self.src(code)
        if src is not None:
            self._stack.append([code, src, -code.co_firstlineno])
        return src is not None

    def end(self, code) -> bool:
        stack = self._stack
        if stack and stack[-1][0] is code:
            _, src, last = stack.pop()
            if last in src.branch:
                self.arc((src, last, _EXIT))
        return self._codes.get(code) is not None

    def line(self, code, line: int, frame=None) -> bool:
        stack = self._stack
        if (not stack or stack[-1][0] is not code) and not self.start(code):
            return False
        top = stack[-1]
        src = top[1]
        if top[2] in src.branch:
            self.arc((src, top[2], line))
        top[2] = line
        exprs = src.stores.get(line)
        if exprs:
            frame = frame or self._frame(code)
            scope = (frame.f_globals, frame.f_locals)
            for base, index in exprs:
                try:
                    held = eval(base, *scope)  # pylint: disable=eval-used
                    at = eval(index, *scope)  # pylint: disable=eval-used
                except Exception:  # pylint: disable=broad-except
                    continue
                self.store(held, at, src, line)
        return True

    @staticmethod
    def _frame(code):
        frame = sys._getframe(1)  # pylint: disable=protected-access
        while frame.f_code is not code:
            frame = frame.f_back
        return frame

    def caller(self) -> Optional[tuple]:
        """``(source, line)`` of the innermost traced frame on the stack."""
        frame = sys._getframe(1)  # pylint: disable=protected-access
        while frame is not None:
            src = self.src(frame.f_code)
            if src is not None:
                return src, frame.f_lineno
            frame = frame.f_back
        return None


class _Monitor(_Recorder):
    """A :class:`_Recorder` on :mod:`sys.monitoring` (Python >= 3.12)."""

    def __enter__(self) -> "_Monitor":
        mon = sys.monitoring  # pylint: disable=no-member
        ev, disable = mon.events, mon.DISABLE
        self._tool = next(i for i in range(6) if mon.get_tool(i) is None)
        mon.use_tool_id(self._tool, "pysidtracker.paths")

        def start(code, _offset):
            return None if self.start(code) else disable

        def end(code, _offset, _value):
            return None if self.end(code) else disable

        def line(code, number):
            return None if self.line(code, number) else disable

        self._events = {
            ev.PY_START: start,
            ev.PY_RESUME: start,
            ev.PY_RETURN: end,
            ev.PY_YIELD: end,
            ev.PY_UNWIND: lambda code, _offset, _exc: self.end(code) and None,
            ev.LINE: line,
        }
        for event, callback in self._events.items():
            mon.register_callback(self._tool, event, callback)
        mon.restart_events()
        mon.set_events(self._tool, functools.reduce(int.__or__, self._events))
        return self

    def __exit__(self, *exc) -> None:
        mon = sys.monitoring  # pylint: disable=no-member
        mon.set_events(self._tool, 0)
        for event in self._events:
            mon.register_callback(self._tool, event, None)
        mon.free_tool_id(self._tool)


class _Settrace(_Recorder):
    """A :class:`_Recorder` on :func:`sys.settrace` (the previous tracer restored)."""

    def __enter__(self) -> "_Settrace":
        self._saved = sys.gettrace()

        def local(frame, event, _arg):
            if event == "line":
                self.line(frame.f_code, frame.f_lineno, frame)
            elif event == "return":
                self.end(frame.f_code)
            return local

        def call(frame, event, _arg):
            return local if event == "call" and self.start(frame.f_code) else None

        sys.settrace(call)
        return self

    def __exit__(self, *exc) -> None:
        sys.settrace(self._saved)


_BACKENDS = {"monitoring": _Monitor, "settrace": _Settrace}


def _wrap(player, recorder: _Recorder, reads: List[int]) -> None:
    """Log a :class:`MemPlayer`'s ``_wr`` stores at their traced callers, and its reads."""
    wr, rd = player._wr, player._rd  # pylint: disable=protected-access
    mem = player._mem  # pylint: disable=protected-access

    def write(addr, value):
        wr(addr, value)
        site = recorder.caller()
        if site is not None:
            recorder.store(mem, addr & 0xFFFF, *site)

    def read(addr):
        reads.append(addr & 0xFFFF)
        return rd(addr)

    player._wr, player._rd = write, read  # pylint: disable=protected-access


def _last_stores(log) -> Tuple[Dict[tuple, tuple], Dict[tuple, int]]:
    """``(id(container), index)`` to the ``(source, line)`` of its last logged store,
    and each ``(source, line)`` to the event count of its first."""
    last: Dict[tuple, tuple] = {}
    first: Dict[tuple, int] = {}
    for held, at, src, line, seq in log:
        first.setdefault((src, line), seq)
        try:
            idx = range(*at.indices(len(held))) if isinstance(at, slice) else (at,)
            last.update(((id(held), i), (src, line)) for i in idx)
        except TypeError:
            continue
    return last, first


def py_paths(
    subject: Union[PySubject, Callable[[], object]],
    nframes: int,
    *,
    backend: Optional[str] = None,
    progress: bool = False,
) -> PathTrace:
    """The line arcs and register store sites of each of ``nframes`` Python frames.

    ``subject`` is a :class:`PySubject` or a bare ``make``; ``grid`` is a ``MemPlayer``'s
    ``regs`` (its ``snapshot``), else reads ``cell``, after each frame, stopping at one whose ``play_frame`` returns None (a non-MemPlayer).

    Notes:
      Functions under the roots are traced (not lambdas, comprehensions or
      MemPlayer's scaffold) on ``sys.monitoring`` or ``sys.settrace`` (``backend``),
      with the same keys. A register's site is the last traced statement whose
      subscript target named its cell's container object and index as it began, or a
      ``MemPlayer._wr`` call's traced caller; a store by C code (through a view or a
      library call) or by a one-line loop body re-run without a new line event is not
      seen. ``reads`` are
      ``MemPlayer._rd`` addresses (none for other players). Keys are
      ``file:qualname:line``; in exec'd ``sources`` the line's text (numbers as ``#``,
      ``@k`` for its k-th repeat in the function) replaces the line number.
      ``order`` ranks an arc by its first run in the frame, a store site by its
      first logged store.
    """
    from tqdm import tqdm

    if not isinstance(subject, PySubject):
        subject = PySubject(subject)
    sources = dict(subject.sources or {})
    for name, text in sources.items():
        linecache.cache[name] = (len(text), None, text.splitlines(True), name)
    player = subject.make()
    roots = list(subject.roots or [*player_roots(player), *sources])
    mode = backend or ("monitoring" if hasattr(sys, "monitoring") else "settrace")
    recorder = _BACKENDS[mode](roots, sources)
    reads: List[int] = []
    if isinstance(player, MemPlayer):
        _wrap(player, recorder, reads)
    rows, arcs, stores, frame_reads = [], [], [], []
    with recorder:
        for _ in tqdm(range(nframes), unit="frame", disable=not progress):
            if player.play_frame() is None and not isinstance(player, MemPlayer):
                break
            cells = [subject.cell(player, r) for r in range(_NREG)]
            mem = isinstance(player, MemPlayer)
            rows.append(list(player.regs) if mem else [held[i] for held, i in cells])
            last, first = _last_stores(recorder.log)
            stores.append(
                {
                    r: (*last[(id(held), i)], first[last[(id(held), i)]])
                    for r, (held, i) in enumerate(cells)
                    if (id(held), i) in last
                }
            )
            arcs.append(recorder.arcs)
            frame_reads.append(np.unique(np.asarray(reads, np.int64)).astype(np.uint16))
            recorder.arcs, recorder.log = {}, []
            reads.clear()
    trace = _py_trace(subject.label, rows, arcs, stores, frame_reads, roots, sources)
    if subject.starts is not None:
        trace.starts = tuple(subject.starts(player))
    return trace


def _py_trace(side, rows, arcs, stores, frame_reads, roots, sources) -> PathTrace:
    ids: Dict[str, int] = {}
    labels: List[str] = []

    def fid(key: str, label: str) -> int:
        if key not in ids:
            ids[key] = len(ids)
            labels.append(label)
        return ids[key]

    nframes = len(rows)
    sites = np.full((nframes, _NREG), -1, np.int32)
    events: List[Tuple[int, int, int]] = []
    for f in range(nframes):
        for (src, a, b), seq in sorted(
            arcs[f].items(), key=lambda x: (x[0][0].name, x[0][1], x[0][2])
        ):
            to = str(b) if b != _EXIT else "exit"
            events.append(
                (
                    f,
                    fid(src.arc_key(a, b), f"{src.file}:{a} {src.text(a)} -> {to}"),
                    seq,
                )
            )
        for r, (src, line, seq) in sorted(stores[f].items()):
            sites[f, r] = fid(
                f"store:{r}:{src.where(line)}",
                f"{src.file}:{line} {src.text(line)} -> {REG_NAMES[r]}",
            )
            events.append((f, int(sites[f, r]), seq))
    ev = np.array(events, np.int64).reshape(-1, 3)
    frames, order = _first(ev[:, 0], ev[:, 1], ev[:, 2], nframes, len(ids) + 1)
    return PathTrace(
        side,
        list(ids),
        labels,
        frames,
        np.array(rows, np.int16).reshape(nframes, _NREG),
        sites,
        frame_reads,
        list(roots),
        dict(sources),
        order=order,
    )


def py_sources(roots: Sequence[str], sources: Optional[Dict[str, str]] = None):
    """The parsed files under ``roots`` (a root named in ``sources`` is that exec'd
    text), skipping any coverage.py cannot parse; each has ``arcs`` (its static branch
    arcs), ``arc_key(src, dst)`` and ``where(line)`` as :func:`py_paths` keys them."""
    sources = sources or {}
    for root in roots:
        if root in sources:
            files = [(root, sources[root], True)]
        else:
            real = os.path.realpath(root)
            walk = sorted(os.walk(real)) if os.path.isdir(real) else []
            names = [os.path.join(d, n) for d, _, ns in walk for n in sorted(ns)]
            paths = [n for n in names if n.endswith(".py")] if walk else [real]
            files = []
            for path in paths:
                with open(path, encoding="utf-8") as fh:
                    files.append((path, fh.read(), False))
        for name, text, generated in files:
            src = _source(name, text, generated)
            if src is not None:
                yield src


def py_universe(
    roots: Sequence[str], sources: Optional[Dict[str, str]] = None
) -> List[str]:
    """Names of every static branch arc in the files under ``roots``, keyed as
    :func:`py_paths` keys them; a root named in ``sources`` is that exec'd text."""
    return sorted(
        {
            src.arc_key(a, b if b > 0 else _EXIT)
            for src in py_sources(roots, sources)
            for a, b in src.arcs
        }
    )


__all__ = [
    "PathTrace",
    "PySubject",
    "StaticCode",
    "cpu_paths",
    "cpu_universe",
    "py_paths",
    "py_sources",
    "py_universe",
    "source_of",
    "static_code",
]
