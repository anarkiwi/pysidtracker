"""Generated inputs for fault localisation: coverage gaps, directed mutants, reproducers.

:func:`gaps`/:func:`distil` measure a corpus against :mod:`~pysidtracker.paths`'
universes; :func:`exercise` mutates data bytes toward unreached branch outcomes;
:func:`minimise` shrinks a divergent input. A patch is ``[[file_offset, byte], ...]``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import deque
from operator import itemgetter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .faultloc import min_cover
from .gridcmp import LEAD, REG_NAMES, verdict_lead
from .image import SidImage
from .oracle import register_grid
from .paths import (
    PathTrace,
    PySubject,
    _booted,
    _edge_label,
    _edge_names,
    _source,
    cpu_paths,
    py_paths,
)
from .player import MemPlayer

_CMP = frozenset(("CMP", "CPX", "CPY"))
_MASK = frozenset(("AND", "ORA", "EOR"))
SIDES = ("6502", "python")

# --- patches ----------------------------------------------------------------


def apply_patch(data: bytes, patch: Iterable[Sequence[int]]) -> bytes:
    """``data`` with each ``[offset, byte]`` of ``patch`` written."""
    out = bytearray(data)
    for off, val in patch:
        out[off] = val & 0xFF
    return bytes(out)


def make_patch(original: bytes, mutated: bytes) -> List[List[int]]:
    """The ``[offset, byte]`` pairs turning ``original`` into ``mutated`` (same length)."""
    a, b = np.frombuffer(original, np.uint8), np.frombuffer(mutated, np.uint8)
    if a.shape != b.shape:
        raise ValueError(f"lengths differ: {len(a)} and {len(b)}")
    return [[int(i), int(b[i])] for i in np.flatnonzero(a != b)]


def addr_to_offset(data: bytes, addr: int) -> Optional[int]:
    """The file offset holding C64 ``addr`` as loaded, None outside the image."""
    image = SidImage.from_bytes(bytes(data))
    inside = image.load <= addr < image.end
    return len(image.container) + addr - image.load if inside else None


# --- coverage ---------------------------------------------------------------


def covered(traces) -> set:
    """Feature names executed by a :class:`PathTrace`, a sequence of them, or names."""
    if isinstance(traces, PathTrace):
        hit = np.unique(np.concatenate(traces.frames or [np.zeros(0, np.int32)]))
        return {traces.names[i] for i in hit}
    if isinstance(traces, str):
        return {traces}
    return set().union(*(covered(t) for t in traces))


def gaps(traces, universe: Iterable[str]) -> List[str]:
    """The ``universe`` keys no trace covered, sorted."""
    return sorted(set(universe) - covered(traces))


def distil(traces_by_tune: Mapping) -> List:
    """A minimum subset of tunes covering every feature any tune covers (AFL cmin)."""
    hits: Dict[str, set] = {}
    for tune, traces in traces_by_tune.items():
        for name in covered(traces):
            hits.setdefault(name, set()).add(tune)
    return min_cover(list(hits.values()))


def py_arcs(roots: Sequence[str], sources: Optional[Dict[str, str]] = None):
    """Each static branch arc's :func:`~pysidtracker.paths.py_universe` key to its
    from-line's key."""
    sources = sources or {}
    files = []
    for root in roots:
        if root in sources:
            files.append((root, sources[root], True))
            continue
        real = os.path.realpath(root)
        walk = sorted(os.walk(real)) if os.path.isdir(real) else []
        names = [os.path.join(d, n) for d, _, ns in walk for n in sorted(ns)]
        for path in [n for n in names if n.endswith(".py")] if walk else [real]:
            with open(path, encoding="utf-8") as fh:
                files.append((path, fh.read(), False))
    out = {}
    for name, text, generated in files:
        src = _source(name, text, generated)
        for a, b in src.arcs if src is not None else ():
            out[src.arc_key(a, max(b, 0))] = src.where(a)
    return out


# --- both sides -------------------------------------------------------------


def as_subject(made) -> PySubject:
    """``make_right(data, song)``'s result as a :class:`PySubject`."""
    return made if isinstance(made, PySubject) else PySubject(lambda: made)


def right_grid(made, nframes: int) -> np.ndarray:
    """``(n, 25)`` registers of a player or :class:`PySubject`, untraced."""
    if not isinstance(made, PySubject):
        return np.asarray(made.render_grid(nframes), np.int16)
    player, rows = made.make(), []
    for _ in range(nframes):
        if player.play_frame() is None and not isinstance(player, MemPlayer):
            break
        rows.append([h[i] for h, i in (made.cell(player, r) for r in range(25))])
    return np.array(rows, np.int16).reshape(len(rows), 25)


def _verdict(left: np.ndarray, right: np.ndarray, max_lead: int) -> dict:
    try:
        v = verdict_lead(left, right, max_lead=max_lead)
    except ValueError as exc:
        return {"ok": False, "first": 0, "first_reg": None, "error": str(exc)}
    return {"ok": v.ok, "first": v.first, "first_reg": v.first_reg, "lead": v.lead}


@dataclass
class Run:
    """Both sides traced on one input; ``error`` is the right side's failure."""

    cpu: PathTrace
    py: Optional[PathTrace]
    verdict: Optional[dict]
    error: Optional[str] = None

    def trace(self, side: str) -> Optional[PathTrace]:
        return self.cpu if side == "6502" else self.py


def run_both(
    data: bytes, song: int, make_right: Callable, nframes: int, max_lead: int = LEAD
) -> Run:
    """:func:`cpu_paths` and :func:`py_paths` of ``nframes`` frames, and their verdict."""
    cpu = cpu_paths(data, nframes, subtune=song - 1)
    try:
        py = py_paths(as_subject(make_right(data, song)), nframes)
    except Exception as exc:  # the right side may reject mutated data
        return Run(cpu, None, None, f"{type(exc).__name__}: {exc}")
    return Run(cpu, py, _verdict(cpu.grid, py.grid, max_lead))


# --- baseline and targets ---------------------------------------------------


@dataclass
class Target:
    """An unreached branch outcome whose branch first ran in ``frame``.

    ``member`` flags the frames running the branch, ``outcomes`` per reached outcome
    the frames taking it; ``pc`` is a 6502 branch's address.
    """

    name: str
    side: str
    label: str
    frame: int
    member: np.ndarray
    outcomes: List[np.ndarray]
    pc: Optional[int] = None


def _target(name, side, label, matrix, fids, pc=None) -> Optional[Target]:
    if not fids:
        return None
    outcomes = [matrix[:, f].toarray().ravel() for f in fids]
    member = np.logical_or.reduce(outcomes)
    return Target(name, side, label, int(np.argmax(member)), member, outcomes, pc)


@dataclass
class Baseline:
    """A tune's traces over ``frames``, its static universes and its data bytes."""

    data: bytes
    song: int
    frames: int
    run: Run
    edges: Dict[Tuple[int, int], str]
    code: object
    arcs: Dict[str, str]
    data_mask: np.ndarray

    def departs(self, run: Run, n: int, max_lead: int = LEAD) -> bool:
        """Whether ``run`` (``n`` frames) fails or mismatches where this tune does not."""
        if run.error is not None or self.run.error is not None:
            return self.run.error is None
        mine = _verdict(self.run.cpu.grid[:n], self.run.py.grid[:n], max_lead)
        key = itemgetter("ok", "first", "first_reg")
        return not run.verdict["ok"] and key(run.verdict) != key(mine)

    def universe(self, side: str) -> List[str]:
        return sorted(self.edges.values() if side == "6502" else self.arcs)

    def targets(self, side: str) -> List[Target]:
        """The unreached outcomes of branches that ran, earliest frame first."""
        trace = self.run.trace(side)
        if trace is None:
            return []
        fid = {n: i for i, n in enumerate(trace.names)}
        if side == "6502":
            rows = [
                (n, e[0], _edge_label(self.code, *e), e[0])
                for e, n in self.edges.items()
                if e[0] >= 0 and self.code.insn(e[0])[1] == "rel"
            ]
        else:
            rows = [(n, where, n, None) for n, where in self.arcs.items()]
        sibs: Dict[object, list] = {}
        for n, key, *_ in rows:
            sibs.setdefault(key, []).extend([fid[n]] if n in fid else [])
        m = trace.matrix().tocsc()
        out = (
            _target(n, side, label, m, sibs[key], pc)
            for n, key, label, pc in rows
            if n not in fid
        )
        return sorted((t for t in out if t), key=lambda t: (t.frame, t.name))


def static_code(data: bytes, subtune: int):
    """The tune's static 6502 control flow from ``init`` and ``play`` (``paths``)."""
    *_, code, entries = _booted(data, subtune, 8_000_000)
    code.explore(entries, entries)
    return code


def data_mask(data: bytes, subtune: int, nframes: int, code=None) -> np.ndarray:
    """Per C64 address: loaded from the file, and neither executed in ``nframes``
    nor statically decoded as code."""
    from .mos6502 import OP_LEN
    from .variants import _executed_offsets

    image = SidImage.from_bytes(bytes(data))
    code = code or static_code(data, subtune)
    offs = _executed_offsets(image, subtune=subtune, nframes=nframes)
    starts = np.union1d(image.load + np.asarray(offs, int), sorted(code.seen))
    lens = np.asarray(OP_LEN, int)[np.frombuffer(bytes(image.mem), np.uint8)[starts]]
    spans = starts[:, None] + np.arange(3)
    mask = np.zeros(0x10000, bool)
    mask[image.load : image.end] = True
    mask[spans[np.arange(3) < lens[:, None]] & 0xFFFF] = False
    return mask


def baseline(
    data: bytes, song: int, make_right: Callable, frames: int, max_lead: int = LEAD
) -> Baseline:
    """Trace both sides of ``data`` over ``frames`` + ``max_lead`` and decode both
    universes."""
    run = run_both(data, song, make_right, frames + max_lead, max_lead)
    code = static_code(data, song - 1)
    arcs = py_arcs(run.py.roots, run.py.sources) if run.py is not None else {}
    mask = data_mask(data, song - 1, frames, code)
    return Baseline(data, song, frames, run, _edge_names(code, ()), code, arcs, mask)


# --- candidate bytes and values ---------------------------------------------


def branch_slice(data: bytes, subtune: int, frame: int, pc: int):
    """``(live, ops)`` of the 6502 branch at ``pc`` in play call ``frame``.

    ``live``: each byte its flags came from (``cpu_slice`` of every execution) to its
    value at entry; ``ops``: the ``(mnemonic, mode, operand)`` that set the flags.
    """
    from jennings.opcodes import OPCODES

    from .whydiff import cpu_history, cpu_slice

    steps = cpu_history(data, frame, subtune=subtune)
    live: Dict[int, int] = {}
    ops = []
    for i in (i for i, s in enumerate(steps) if s.pc == pc):
        keep, got = cpu_slice(steps, i, "PC")
        live.update((a, v) for a, v in got.items() if isinstance(a, int))
        prior = [k for k in keep if k < i]
        if prior:
            raw = steps[prior[-1]].raw
            ops.append((*OPCODES[raw[0]][:2], raw[1] if len(raw) > 1 else None))
    return live, list(dict.fromkeys(ops))


def values(v: int, ops: Sequence[tuple] = ()) -> List[int]:
    """Bytes to try for ``v``: the flag-setting ``ops``' boundaries (``CMP #i``:
    i-1..i+1, ``AND``/``ORA``/``EOR #m``: m's bits, ``BIT``: N/V), then bit flips,
    ±1, 0 and $FF."""
    out = []
    for mn, mode, arg in ops:
        if mode == "imm" and mn in _CMP:
            out += [arg - 1, arg, arg + 1]
        elif mode == "imm" and mn in _MASK:
            out += [v ^ arg] + [v ^ (1 << k) for k in range(8) if arg >> k & 1]
        elif mn == "BIT":
            out += [v ^ 0x80, v ^ 0x40]
    out += [v ^ (1 << k) for k in range(7, -1, -1)] + [v + 1, v - 1, 0, 0xFF]
    return [x for x in dict.fromkeys(x & 0xFF for x in out) if x != v]


def rank_reads(
    reads: Sequence[np.ndarray], target: Target, mask: np.ndarray, first=()
) -> np.ndarray:
    """Data addresses read up to the target's frame, most suspect first: in ``first``,
    then read differently by frames taking different reached outcomes, then read in
    the target's frame, then read more by frames running the branch than the rest."""
    f = target.frame
    cols = np.unique(np.concatenate([np.asarray(r, int) for r in reads[: f + 1]]))
    cols = cols[mask[cols]]
    n = min(len(reads), len(target.member))
    r = np.stack([np.isin(cols, reads[i]) for i in range(n)]).astype(float)

    def mean(rows):
        return r[rows].mean(0) if rows.any() else np.zeros(len(cols))

    member = target.member[:n]
    spec = mean(member) - mean(~member)
    contrast = np.zeros(len(cols))
    for took in target.outcomes:
        took, rest = took[:n], member & ~took[:n]
        if took.any() and rest.any():
            contrast = np.maximum(contrast, np.abs(mean(took) - mean(rest)))
    sliced = np.isin(cols, np.fromiter(first, int))
    return cols[np.lexsort((-spec, -r[min(f, n - 1)], -contrast, ~sliced))]


def candidates(base: Baseline, target: Target, rng, width: int = 16):
    """``(offset, byte)`` single-byte mutations for ``target``: the top ``width``
    addresses' :func:`values` first, then their other bytes in random order."""
    live, ops = {}, []
    if target.pc is not None:
        live, ops = branch_slice(base.data, base.song - 1, target.frame, target.pc)
    py = base.run.py
    reads = py.reads if py is not None and target.side != "6502" else None
    if reads is None or not any(len(x) for x in reads):
        reads = base.run.cpu.reads
    addrs = rank_reads(reads, target, base.data_mask, live)[:width]
    offs = [addr_to_offset(base.data, int(a)) for a in addrs]
    for off, a in zip(offs, addrs):
        yield from ((off, x) for x in values(base.data[off], ops if a in live else ()))
    for i in rng.permutation(len(offs) * 256):
        off, x = offs[i // 256], int(i % 256)
        if x != base.data[off]:
            yield off, x


# --- exercise ---------------------------------------------------------------


@dataclass
class Mutant:
    """A kept input: its patch, the features it reached first, and its verdict.

    ``new``/``labels``: per side the features no earlier input reached; ``hit``: it
    reached ``target``; ``diverges``: the right side departs (``verdict``) or fails.
    """

    patch: List[List[int]]
    target: str
    side: str
    frames: int
    hit: bool
    new: Dict[str, List[str]] = field(default_factory=dict)
    labels: Dict[str, List[str]] = field(default_factory=dict)
    diverges: bool = False
    verdict: Optional[dict] = None
    error: Optional[str] = None


def exercise(
    data: bytes,
    song: int,
    make_right: Callable,
    *,
    targets: Optional[Iterable[str]] = None,
    side: str = "6502",
    budget: int = 200,
    frames: int = 500,
    rng=None,
    margin: int = 2,
    width: int = 16,
    max_lead: int = LEAD,
    base: Optional[Baseline] = None,
    progress: bool = False,
) -> List[Mutant]:
    """Mutate data bytes toward ``side``'s unreached branch outcomes, ``budget`` tries.

    Targets take turns at their next :func:`candidates` mutation; each try runs both
    sides to the target's frame + ``margin`` + ``max_lead``, and is kept as a
    :class:`Mutant` when it reaches a feature no earlier input reached on either side.
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    rng = rng if rng is not None else np.random.default_rng(0)
    base = base or baseline(data, song, make_right, frames, max_lead)
    want = None if targets is None else set(targets)
    todo = deque(
        (t, candidates(base, t, rng, width))
        for t in base.targets(side)
        if want is None or t.name in want
    )
    seen = {s: covered(base.run.trace(s) or ()) for s in SIDES}
    tried: set = set()
    mutants: List[Mutant] = []
    with tqdm(total=budget, desc=f"exercise {side}", disable=not progress) as bar:
        while todo and len(tried) < budget:
            target, gen = todo.popleft()
            if target.name in seen[side]:
                continue
            pick = next((p for p in gen if p not in tried), None)
            if pick is None:
                continue
            tried.add(pick)
            todo.append((target, gen))
            n = min(target.frame + 1 + margin + max_lead, len(base.run.cpu.grid))
            patch = [list(pick)]
            run = run_both(apply_patch(data, patch), song, make_right, n, max_lead)
            bar.update()
            new, labels = {}, {}
            for s in SIDES:
                trace = run.trace(s)
                new[s] = sorted(covered(trace or ()) - seen[s])
                labels[s] = [trace.labels[trace.names.index(g)] for g in new[s]]
                seen[s].update(new[s])
            if any(new.values()):
                mutants.append(
                    Mutant(
                        patch,
                        target.name,
                        side,
                        n,
                        hit=target.name in new[side],
                        new=new,
                        labels=labels,
                        diverges=base.departs(run, n, max_lead),
                        verdict=run.verdict,
                        error=run.error,
                    )
                )
                bar.set_postfix(kept=len(mutants), targets=len(todo))
    return mutants


# --- minimise ---------------------------------------------------------------


def ddmin(changes: Sequence, test: Callable[[list], bool]) -> list:
    """A 1-minimal subset of ``changes`` passing ``test`` (Zeller's ddmin, memoised).

    ``test(changes)`` is assumed to hold.
    """
    memo: Dict[frozenset, bool] = {}

    def check(sub):
        key = frozenset(sub)
        if key not in memo:
            memo[key] = test([changes[i] for i in sorted(sub)])
        return memo[key]

    items, n = list(range(len(changes))), 2
    if check([]):
        return []
    while len(items) >= 2:
        chunks = [c.tolist() for c in np.array_split(items, min(n, len(items)))]
        pick = next((c for c in chunks if check(c)), None)
        if pick is not None:
            items, n = pick, 2
            continue
        rests = ([i for i in items if i not in set(c)] for c in chunks)
        pick = next((r for r in rests if check(r)), None) if n > 2 else None
        if pick is not None:
            items, n = pick, max(n - 1, 2)
        elif n >= len(items):
            break
        else:
            n = min(2 * n, len(items))
    return [changes[i] for i in items]


def first_true(test: Callable[[int], bool], hi: int) -> int:
    """The least ``k`` in ``1..hi`` with ``test(k)``, by bisection (monotone ``test``)."""
    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if test(mid) else (mid + 1, hi)
    return hi


def minimise(
    data: bytes,
    song: int,
    predicate: Callable[[bytes, int], bool],
    *,
    patch: Optional[Sequence[Sequence[int]]] = None,
    addrs: Optional[Iterable[int]] = None,
    frames: Optional[int] = None,
    neutral: int = 0,
    progress: bool = False,
) -> Tuple[List[List[int]], int]:
    """The least patch of ``data`` and frame count keeping ``predicate(bytes, frames)``.

    ddmin over ``patch``'s entries, or (no ``patch``) over which of ``addrs``' data
    bytes (default: the 6502's reads) to keep while the rest become ``neutral``;
    frames bisected before and after. Raises ``ValueError`` if the start fails.
    """
    frames = frames or 500
    if patch is not None:
        changes = [tuple(p) for p in patch]

        def build(kept):
            return apply_patch(data, kept)

    else:
        if addrs is None:
            mask = data_mask(data, song - 1, frames)
            reads = np.concatenate(cpu_paths(data, frames, subtune=song - 1).reads)
            addrs = [a for a in np.unique(reads) if mask[a]]
        offs = (addr_to_offset(data, int(a)) for a in addrs)
        changes = sorted({o for o in offs if o is not None and data[o] != neutral})

        def build(kept):
            keep = set(kept)
            return apply_patch(data, [(o, neutral) for o in changes if o not in keep])

    with tqdm(desc="minimise", unit="test", disable=not progress) as bar:

        def test(kept, n):
            bar.update()
            return predicate(build(kept), n)

        if not test(changes, frames):
            raise ValueError("the starting input does not satisfy the predicate")
        n = first_true(lambda k: test(changes, k), frames)
        kept = ddmin(changes, lambda c: test(c, n))
        n = first_true(lambda k: test(kept, k), n)
    return make_patch(data, build(kept)), n


def same_onset_predicate(
    make_right: Callable, register: int, song: int = 1, *, max_lead: int = LEAD
) -> Callable[[bytes, int], bool]:
    """``predicate(data, frames)``: within ``frames`` the right side first departs from
    the 6502 grid at ``register`` (a right side that fails counts as not)."""

    def predicate(data: bytes, frames: int) -> bool:
        try:
            left = register_grid(data, frames, subtune=song - 1)
            right = right_grid(make_right(data, song), frames + max_lead)
            v = verdict_lead(left, right, max_lead=max_lead)
        except Exception:  # the right side may reject reduced data
            return False
        return not v.ok and v.first_reg == register

    return predicate


# --- CLI --------------------------------------------------------------------


def load_patch(spec: str) -> List[List[int]]:
    """The patch at ``path.json#k`` (a mutants file's k-th) or ``path.json``."""
    path, _, k = spec.partition("#")
    got = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(got, dict) and "mutants" in got:
        return got["mutants"][int(k or 0)]["patch"]
    return got["patch"] if isinstance(got, dict) else got


def _gaps_cli(args, data, make_right) -> dict:
    base = baseline(data, args.song, make_right, args.frames)
    doc = {"error": base.run.error}
    for side in SIDES:
        universe = base.universe(side)
        missing = gaps(base.run.trace(side) or (), universe)
        reach = base.targets(side)
        doc[side] = {
            "universe": len(universe),
            "covered": len(universe) - len(missing),
            "gaps": missing,
            "targets": [[t.name, t.label, t.frame] for t in reach],
        }
        print(
            f"{side}: {len(universe) - len(missing)}/{len(universe)} covered, "
            f"{len(reach)} unreached outcomes of branches that ran"
        )
        for t in reach:
            print(f"  frame {t.frame:5d}  {t.label}")
    return doc


def _exercise_cli(args, data, make_right) -> dict:
    base = baseline(data, args.song, make_right, args.frames)
    mutants = exercise(
        data,
        args.song,
        make_right,
        side=args.side,
        budget=args.budget,
        frames=args.frames,
        rng=np.random.default_rng(args.seed),
        base=base,
        progress=True,
    )
    gain = {s: sum(len(m.new[s]) for m in mutants) for s in SIDES}
    print(f"{len(mutants)} mutants, {sum(m.hit for m in mutants)} hit, new {gain}")
    for k, m in enumerate(mutants):
        if m.diverges:
            v = m.verdict or {}
            reg = v.get("first_reg")
            at = f"frame {v.get('first')} {REG_NAMES[reg] if reg is not None else ''}"
            print(f"  #{k} {m.patch} diverges: {m.error or at}")
    return {
        "tune": str(args.tune),
        "song": args.song,
        "side": args.side,
        "frames": args.frames,
        "baseline": base.run.verdict,
        "gain": gain,
        "mutants": [asdict(m) for m in mutants],
    }


def _minimise_cli(args, data, make_right) -> dict:
    patch = load_patch(args.patch) if args.patch else None
    register = args.register
    if register is None:
        start = apply_patch(data, patch or [])
        left = register_grid(start, args.frames, subtune=args.song - 1)
        right = right_grid(make_right(start, args.song), args.frames + LEAD)
        register = _verdict(left, right, LEAD)["first_reg"]
        if register is None:
            raise SystemExit("the starting input does not diverge")
    got, n = minimise(
        data,
        args.song,
        same_onset_predicate(make_right, register, args.song),
        patch=patch,
        frames=args.frames,
        neutral=args.neutral,
        progress=True,
    )
    print(f"{len(got)} bytes, {n} frames, register {REG_NAMES[register]}: {got}")
    return {
        "tune": str(args.tune),
        "song": args.song,
        "register": register,
        "frames": n,
        "patch": got,
    }


def main(argv=None) -> int:
    """``python -m pysidtracker.pathfuzz {gaps,exercise,minimise} TUNE.sid --right SPEC``."""
    from .survey import load_render

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    cmds = {"gaps": _gaps_cli, "exercise": _exercise_cli, "minimise": _minimise_cli}
    for name in cmds:
        p = sub.add_parser(name)
        p.add_argument("tune", type=Path)
        p.add_argument("--right", required=True, help="module:fn or path.py:fn")
        p.add_argument("--song", type=int, default=1)
        p.add_argument("--frames", type=int, default=500)
        p.add_argument("--out", type=Path)
    ex, mn = sub.choices["exercise"], sub.choices["minimise"]
    ex.add_argument("--side", choices=SIDES, default="6502")
    ex.add_argument("--budget", type=int, default=200)
    ex.add_argument("--seed", type=int, default=0)
    mn.add_argument("--patch", help="mutants.json#k")
    mn.add_argument("--register", type=lambda s: int(s, 0))
    mn.add_argument("--neutral", type=lambda s: int(s, 0), default=0)
    args = parser.parse_args(argv)
    doc = cmds[args.cmd](args, args.tune.read_bytes(), load_render(args.right))
    if args.out is not None:
        args.out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
