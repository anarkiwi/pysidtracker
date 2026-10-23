"""Corpus-wide path differencing of two players, reduced to the fewest fixes.

``build`` records both sides' per-frame paths (:mod:`~pysidtracker.paths`) over every
selected survey subtune and the frame they first diverge; ``rank`` localises the
divergences over the corpus (:mod:`~pysidtracker.faultloc`) to a minimum cover of
fixes; ``xfails`` writes each fix's representative as a strict xfail case that
:func:`check_case` runs with the whydiff explanation (docs/pathdiff.md).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from tqdm import tqdm

from . import faultloc as fl
from .formats import find_format
from .gridcmp import GATE_COLS, LEAD, NREGS, verdict_lead, verdict_onset
from .jobs import deadline, imap
from .oracle import register_grid
from .paths import PathTrace, PySubject, cpu_paths, py_paths
from .player import MemPlayer
from .registers import PW_HI_REGS
from .relevance import cpu_relevant, py_relevant
from .survey import load_render, load_tunes, oracle_frames, read_results
from .testing import (
    ChecksumMismatch,
    TuneFetchError,
    default_cache,
    oracle_grid,
    resolve_tune,
)
from .whydiff import Divergence, explain_cell, explain_divergence, python_side

CPU = "6502"
FORMAT = 3
VERDICTS = ("differs", "exact")
COMPARE = ("lead", "onset")
_ROW_KEYS = ("indptr", "indices", "count", "fail", "onset", "lead", "register")
_ROW_KEYS += ("host", "frames", "rank", "relevant")
_PW_MASK = np.full(NREGS, 0xFF, np.int16)
_PW_MASK[list(PW_HI_REGS)] = 0x0F


# --- sides ------------------------------------------------------------------


def subject(spec: str, data: bytes, song: int) -> Optional[PySubject]:
    """The :class:`~pysidtracker.paths.PySubject` side ``spec`` plays for 1-based ``song``.

    ``spec`` is ``6502`` (None: jennings), ``format:NAME`` (an installed format's
    player, labelled ``NAME``) or a :func:`~pysidtracker.survey.load_render` spec
    ``fn(data, song)`` returning a ``MemPlayer`` (labelled ``python``) or a ``PySubject``.
    """
    if spec == CPU:
        return None
    if spec.startswith("format:"):
        fmt = find_format(data, spec.partition(":")[2])
        return PySubject(lambda: fmt.play(data, song - 1), label=fmt.name)
    return as_subject(load_render(spec), data, song)


def as_subject(fn: Callable, data: bytes, song: int) -> PySubject:
    """``fn(data, song)`` as a :class:`~pysidtracker.paths.PySubject`: its own, else
    one making that player first and fresh ones after."""
    got = fn(data, song)
    if isinstance(got, PySubject):
        return got
    first = [got]
    return PySubject(lambda: first.pop() if first else fn(data, song))


def trace(spec: str, data: bytes, song: int, nframes: int) -> PathTrace:
    """The :class:`~pysidtracker.paths.PathTrace` of side ``spec`` over ``nframes``."""
    subj = subject(spec, data, song)
    if subj is None:
        return cpu_paths(data, nframes, subtune=song - 1)
    return py_paths(subj, nframes)


def relevant(spec: str, tr: PathTrace, data: bytes, song: int, frame: int, register):
    """``tr``'s feature ids that side ``spec``'s ``register`` at ``frame`` depends on
    (:mod:`~pysidtracker.relevance`)."""
    subj = subject(spec, data, song)
    if subj is None:
        return cpu_relevant(tr, data, frame, register, song - 1)
    return py_relevant(tr, subj, frame, register)


def grid(spec: str, data: bytes, song: int, nframes: int):
    """``(grid, starts)`` of side ``spec`` over ``nframes`` (fewer where it ends), untraced."""
    subj = subject(spec, data, song)
    if subj is None:
        return (
            np.asarray(register_grid(data, nframes, subtune=song - 1), np.int16),
            None,
        )
    return play(subj, nframes)


def play(subj: PySubject, nframes: int):
    """``(grid, starts)`` of a fresh ``subj`` player over ``nframes`` (fewer where it
    ends), untraced: a ``MemPlayer``'s register file, else its ``cell``s."""
    player, rows = subj.make(), []
    for _ in range(nframes):
        if player.play_frame() is None and not isinstance(player, MemPlayer):
            break
        if isinstance(player, MemPlayer):
            rows.append(list(player.regs))
        else:
            rows.append(
                [held[i] for held, i in (subj.cell(player, r) for r in range(NREGS))]
            )
    starts = tuple(subj.starts(player)) if subj.starts is not None else None
    return np.array(rows, np.int16).reshape(-1, NREGS), starts


def prefixes(left: str, right: str) -> Tuple[str, str]:
    """The feature-name prefixes of two side labels, suffixed apart when equal."""
    if left == right:
        left, right = f"{left}-left", f"{right}-right"
    return f"{left}:", f"{right}:"


def slug(spec: str) -> str:
    """A side spec as a file-name component."""
    target, _, fn = spec.partition(":")
    name = Path(target).stem if target.endswith(".py") else target
    return re.sub(r"[^\w.]+", "_", ".".join(filter(None, (name, fn))))


def pair_dir(work, left: str, right: str) -> Path:
    """``work/pathdiff/<left>-<right>``: the pair's cache."""
    return Path(work) / "pathdiff" / f"{slug(left)}-{slug(right)}"


def apply_patch(data: bytes, patch: Optional[Sequence] = None) -> bytes:
    """``data`` with each ``[file_offset, byte]`` of ``patch`` written (the byte's low
    8 bits; the last entry for an offset wins)."""
    if not patch:
        return bytes(data)
    out = np.frombuffer(bytes(data), np.uint8).copy()
    at = np.asarray(patch, np.int64).reshape(-1, 2)[::-1]
    _, last = np.unique(at[:, 0], return_index=True)
    out[at[last, 0]] = at[last, 1] & 0xFF
    return out.tobytes()


# --- comparison -------------------------------------------------------------


def gate_starts(grid_) -> tuple:
    """Each voice's first gated frame in ``grid_`` (None: never)."""
    on = (np.asarray(grid_)[:, GATE_COLS] & 1).astype(bool)
    return tuple(int(c.argmax()) if c.any() else None for c in on.T)


class Cut(NamedTuple):
    """Where right grid first diverges from left: ``frame`` (aligned, None: never),
    ``register``, the alignment ``lead`` and the ``size`` of aligned frames compared."""

    lead: int
    size: int
    frame: Optional[int]
    register: Optional[int]

    def at(self, right: bool, frame: Optional[int] = None) -> int:
        """Aligned ``frame`` (default the onset) on the left or right side."""
        frame = self.frame if frame is None else frame
        return frame + max(self.lead if right else -self.lead, 0)


def compare(lg, rg, mode: str = "lead", starts=None) -> Cut:
    """The :class:`Cut` of right grid ``rg`` against reference ``lg``.

    Pulse-width highs are masked to their nibble, as the oracle grids are.
    ``lead``: :func:`~pysidtracker.gridcmp.verdict_lead` (``rg`` covers ``LEAD`` more);
    ``onset``: :func:`~pysidtracker.gridcmp.verdict_onset` from each voice's first note
    in ``starts`` (default its first gate in ``lg``), frames unshifted.
    """
    lg, rg = (np.asarray(g, np.int16).reshape(-1, NREGS) & _PW_MASK for g in (lg, rg))
    if mode == "lead":
        v = verdict_lead(lg, rg, max_lead=LEAD)
        return Cut(v.lead, v.n - max(-v.lead, 0), v.first, v.first_reg)
    n = min(len(lg), len(rg))
    starts = gate_starts(lg[:n]) if starts is None else starts
    v = verdict_onset(rg[:n], starts, lg[:n])
    return Cut(0, n, *(v.onset or (None, None)))


def _site(tr: PathTrace, frame: int, register: int, prefix: str) -> str:
    """The prefixed name of ``tr``'s last store to ``register`` up to ``frame``."""
    col = tr.sites[: frame + 1, register]
    hit = np.flatnonzero(col >= 0)
    return prefix + tr.names[col[hit[-1]]] if hit.size else ""


def _host(ocsv, cpu: PathTrace, cut: Cut, right: bool) -> Optional[int]:
    """Aligned frames before jennings first parts from the cached sidtrace oracle."""
    if ocsv is None or not Path(ocsv).exists() or len(cpu.grid) <= LEAD:
        return None
    _, ora = oracle_frames(ocsv, nframes=len(cpu.grid))
    v = verdict_lead(ora, cpu.grid, max_lead=LEAD)
    if v.ok:
        return None
    return v.first + max(v.lead, 0) - cut.at(right, 0)


def _unique_rows(mat: sparse.csr_matrix, fail: bool):
    """The distinct passing rows of ``mat`` with counts, then its failing last row."""
    npass = mat.shape[0] - int(fail)
    packed = np.packbits(mat[:npass].toarray(), axis=1)
    key = np.ascontiguousarray(np.column_stack([np.zeros(npass, np.uint8), packed]))
    _, first, count = np.unique(
        key.view(np.dtype((np.void, key.shape[1]))).ravel(),
        return_index=True,
        return_counts=True,
    )
    if fail:
        first, count = np.append(first, npass), np.append(count, 1)
    return mat[first], count


def record(data, song, left, right, mode="lead", nframes=250, ocsv=None) -> dict:
    """Both sides' paths over ``nframes`` aligned frames up to their first divergence.

    ``names``/``labels`` are prefixed feature names and texts; ``indptr``/``indices``
    the distinct rows over them (the failing onset row last when ``fail``), ``count``
    each row's frames. ``onset``/``lead``/``register`` place the divergence (``onset``
    -1: none), ``left``/``right`` are each side's last store to it; ``rank`` is each
    name's first-occurrence rank on its side in the onset frame (-1: not run there),
    ``relevant`` the indices of the names its register depends on (``relevance``);
    ``host`` marks a jennings side parting from the ``ocsv`` sidtrace oracle first,
    its rows dropped from there.
    """
    total = nframes + (LEAD if mode == "lead" else 0)
    tl, tr = (trace(s, data, song, total) for s in (left, right))
    cut = compare(tl.grid, tr.grid, mode, tr.starts or tl.starts)
    keep = cut.size if cut.frame is None else cut.frame + 1
    cpu = (tl, False) if left == CPU else (tr, True) if right == CPU else None
    host = None if cpu is None else _host(ocsv, cpu[0], cut, cpu[1])
    dropped = host is not None and host < keep
    if dropped:
        cut, keep = cut._replace(frame=None, register=None), max(host, 0)
    pl, pr = prefixes(tl.side, tr.side)
    lo, ro = cut.at(False, 0), cut.at(True, 0)
    mat = sparse.hstack(
        [tl.matrix()[lo : lo + keep], tr.matrix()[ro : ro + keep]], format="csr"
    )
    used = np.unique(mat.indices)
    rows, count = _unique_rows(mat[:, used], cut.frame is not None)
    names = [pl + n for n in tl.names] + [pr + n for n in tr.names]
    labels = tl.labels + tr.labels
    sites = ["", ""]
    rank = np.full(len(used), -1, np.int32)
    rel = np.zeros(0, np.int64)
    if cut.frame is not None:
        sites = [
            _site(t, cut.at(r), cut.register, p)
            for t, r, p in ((tl, False, pl), (tr, True, pr))
        ]
        at = [(tl, cut.at(False), 0), (tr, cut.at(True), len(tl.names))]
        if all(t.order for t, _, _ in at):
            cols = np.concatenate([t.frames[f] + off for t, f, off in at])
            rank[np.searchsorted(used, cols)] = np.concatenate(
                [t.order[f] for t, f, _ in at]
            )
        rel = np.concatenate(
            [
                relevant(s, t, data, song, f, cut.register) + off
                for s, (t, f, off) in zip((left, right), at)
            ]
        )
        rel = np.searchsorted(used, rel[np.isin(rel, used)])
    return dict(
        names=[names[i] for i in used],
        labels=[labels[i] for i in used],
        indptr=rows.indptr.astype(np.int64),
        indices=rows.indices.astype(np.int32),
        count=count.astype(np.int64),
        fail=cut.frame is not None,
        onset=-1 if cut.frame is None else cut.frame,
        lead=cut.lead,
        register=-1 if cut.register is None else cut.register,
        rank=rank,
        relevant=rel.astype(np.int32),
        left=sites[0],
        right=sites[1],
        host=dropped,
        frames=nframes,
        prefixes=[pl, pr],
        error="",
    )


def _err(exc) -> str:
    return f"{type(exc).__name__}: {exc}"


def _job(hvsc, row, left, right, mode, nframes, ocsv, timeout) -> dict:
    """:func:`record` of one ``results.csv`` row, or its ``error``."""
    late = dict(error=f"TimeoutError: exceeded {timeout}s")
    try:
        with deadline(timeout, late):
            data = (Path(hvsc) / row["path"]).read_bytes()
            return record(data, int(row["song"]), left, right, mode, nframes, ocsv)
    except Exception as exc:  # pylint: disable=broad-except
        return dict(error=_err(exc))


# --- the cache --------------------------------------------------------------


class _Vocab:
    """The pair's append-only feature names, ``vocab.txt``: line ``i`` names id ``i``."""

    def __init__(self, path: Path):
        self.path = path
        lines = path.read_text("utf-8").splitlines() if path.exists() else []
        self.ids = {n: i for i, n in enumerate(lines)}

    def __call__(self, names: Sequence[str]) -> np.ndarray:
        new = [n for n in dict.fromkeys(names) if n and n not in self.ids]
        if new:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write("".join(n + "\n" for n in new))
            self.ids.update(zip(new, range(len(self.ids), len(self.ids) + len(new))))
        return np.array([self.ids.get(n, -1) for n in names], np.int64)


def _save(path: Path, vocab: _Vocab, row: dict, rec: dict) -> None:
    meta = {k: row.get(k, "") for k in ("key", "path", "song", "verdict")}
    meta.update(error=rec.get("error", ""), prefixes=rec.get("prefixes", []))
    arrays = {k: np.asarray(rec[k]) for k in _ROW_KEYS if k in rec}
    if "names" in rec:
        arrays.update(
            feat=vocab(rec["names"]),
            labels=np.array(rec["labels"], str),
            site=vocab([rec["left"], rec["right"]]),
        )
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, path)


def _load(path: Path, names: np.ndarray, labels: bool = False) -> dict:
    """A cached tune as a :func:`record` dict plus its ``results.csv`` fields."""
    with np.load(path) as z:
        rec = json.loads(str(z["meta"]))
        if rec["error"]:
            return rec
        rec.update({k: z[k] for k in _ROW_KEYS if k in z.files})
        rec["names"] = names[z["feat"]].tolist()
        site = z["site"]
        rec["left"], rec["right"] = (str(names[i]) if i >= 0 else "" for i in site)
        if labels:
            rec["labels"] = z["labels"].tolist()
    rec["fail"] = bool(rec["fail"])
    return rec


def select(work, tunes_csv, verdicts=VERDICTS, limit=None) -> List[dict]:
    """``work/results.csv`` rows of ``verdicts``; without it, every song of ``tunes_csv``."""
    res = Path(work) / "results.csv"
    if res.exists():
        rows = [r for r in read_results(res).values() if r["verdict"] in verdicts]
    else:
        rows = [
            dict(key=f"{t['md5']}_{s}", path=t["path"], song=str(s), verdict="")
            for t in load_tunes(tunes_csv)
            for s in range(1, int(t["songs"]) + 1)
        ]
    return rows[:limit]


def frames_for(row: dict, max_frames: int) -> int:
    """Reference frames to record: the song's (``results.csv`` ``frames``), at most
    ``max_frames``."""
    return max(1, min(int(row.get("frames") or max_frames), max_frames))


def _meta(pair: Path, **want) -> dict:
    path = pair / "meta.json"
    meta = json.loads(path.read_text("utf-8")) if path.exists() else {}
    _check_format(pair, meta)
    for k in ("left", "right", "compare"):
        if meta.get(k, want[k]) != want[k]:
            raise ValueError(f"{pair} holds {k}={meta[k]!r}, not {want[k]!r}")
    meta.update(want)
    pair.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n", "utf-8")
    return meta


def _check_format(pair: Path, meta: dict) -> None:
    if meta and meta.get("format", 1) != FORMAT:
        raise ValueError(
            f"{pair} is cache format {meta.get('format', 1)}, not {FORMAT}: "
            "rebuild it with build --force"
        )


def build(
    hvsc,
    tunes_csv,
    work,
    left: str = CPU,
    right: str = "",
    *,
    mode: str = "lead",
    verdicts=VERDICTS,
    limit: Optional[int] = None,
    jobs: Optional[int] = None,
    max_frames: int = 3000,
    timeout: float = 600,
    force: bool = False,
) -> Counter:
    """Record each selected subtune not yet cached into ``pair_dir(work, left, right)``.

    Returns the tally of ``differs``, ``agrees``, ``host_mismatch`` and ``error``.
    """
    pair = pair_dir(work, left, right)
    if force:
        (pair / "meta.json").unlink(missing_ok=True)
    meta = _meta(
        pair, left=left, right=right, compare=mode, hvsc=str(hvsc), format=FORMAT
    )
    (pair / "tunes").mkdir(exist_ok=True)
    vocab = _Vocab(pair / "vocab.txt")
    rows = select(work, tunes_csv, verdicts, limit)
    todo = [r for r in rows if force or not _tune_path(pair, r["key"]).exists()]
    tasks = [
        (
            str(hvsc),
            r,
            left,
            right,
            mode,
            frames_for(r, max_frames),
            str(_survey_oracle(work, r["key"])),
            timeout,
        )
        for r in todo
    ]
    ended = (
        ((i, _job(*t), None) for i, t in enumerate(tasks))
        if jobs == 1
        else imap(_job, tasks, jobs)
    )
    tally: Counter = Counter()
    bar = tqdm(ended, total=len(tasks), unit="tune", desc="build")
    for i, rec, error in bar:
        rec = rec or dict(error=_err(error))
        if not rec["error"] and "prefixes" not in meta:
            meta = _meta(pair, **dict(meta, prefixes=rec["prefixes"]))
        _save(_tune_path(pair, todo[i]["key"]), vocab, todo[i], rec)
        tally[_outcome(rec)] += 1
        bar.set_postfix(tally)
    return tally


def _tune_path(pair: Path, key: str) -> Path:
    return pair / "tunes" / f"{key}.npz"


def _survey_oracle(work, key: str) -> Path:
    """The survey's cached sidtrace oracle trace of ``key``."""
    return Path(work) / "oracle" / f"{key}.csv.zst"


def _outcome(rec: dict) -> str:
    if rec.get("error"):
        return "error"
    if rec["fail"]:
        return "differs"
    return "host_mismatch" if rec["host"] else "agrees"


# --- ranking ----------------------------------------------------------------


def _matrix(rec: dict, width: int) -> Tuple[sparse.csr_matrix, Optional[int]]:
    """A record's frames (distinct rows repeated) and its onset row (None: none)."""
    count = np.asarray(rec["count"])
    rows = sparse.csr_matrix(
        (np.ones(len(rec["indices"]), bool), rec["indices"], rec["indptr"]),
        shape=(len(count), width),
    )
    onset = int(count[:-1].sum()) if rec["fail"] else None
    return rows[np.repeat(np.arange(len(count)), count)], onset


def _sample(key: str, rec: dict, extra=()) -> tuple:
    """A :func:`~pysidtracker.faultloc.build_spectrum` sample of a record."""
    mat, onset = _matrix(rec, len(rec["names"]))
    rank = rec.get("rank")
    if rank is not None:
        rank = np.append(rank, np.full(len(rec["names"]) - len(rank), -1))
    return key, mat, rec["names"], onset, extra, rank, rec.get("relevant")


def info_extras(work) -> Dict[str, Tuple[str, ...]]:
    """Per ``results.csv`` key, its ``info`` JSON as ``info:key=value`` features."""
    res = Path(work) / "results.csv"
    if not res.exists():
        return {}
    out = {}
    for key, row in read_results(res).items():
        try:
            info = json.loads(row.get("info") or "{}")
        except ValueError:
            continue
        if isinstance(info, dict):
            out[key] = tuple(f"info:{k}={v}" for k, v in sorted(info.items()))
    return out


def _pair(rec: dict) -> Optional[Tuple[str, str]]:
    sites = (rec["left"] or None, rec["right"] or None)
    return sites if any(sites) else None


@dataclass
class Analysis:
    """A pair's cached corpus localised: its spectrum, correspondence, per-failure
    partings and candidates, and ranked fixes."""

    pair: Path
    meta: dict
    records: Dict[str, dict]
    extras: Dict[str, Tuple[str, ...]]
    spec: fl.Spectrum
    corr: fl.Correspondence
    scores: np.ndarray
    parts: List[Optional[fl.Parting]]
    cands: List[set]
    fixes: List[fl.Fix]

    def tune(self, key: str, labels: bool = False) -> dict:
        """The cached record of ``key`` (``labels`` loaded too)."""
        if not labels:
            return self.records[key]
        return _load(_tune_path(self.pair, key), _names(self.pair), True)

    def context(self, fix: fl.Fix) -> Tuple[Tuple[str, str], ...]:
        """``(tag, name)`` of a ``first`` fix's representative: the departing side's
        feature before it (``after``) and the other side's departure (``vs``)."""
        if fix.key[0] != "first" or not fix.representative:
            return ()
        fails = self.spec.tune[self.spec.fail_rows]
        part = self.parts[
            int(np.argmax(fails == self.spec.tunes.index(fix.representative)))
        ]
        if part is None:
            return ()
        s = 0 if fix.key[1] else 1
        got = (("after", part.after[s]), ("vs", part[1 - s]))
        return tuple((t, n) for t, n in got if n)

    def label_fixes(self) -> None:
        """Name each fix's features by its representative's texts, else by the first
        tune that ran them; a ``first`` fix's labels end with its :meth:`context`."""
        own = {}
        for fix in self.fixes:
            if fix.representative:
                rep = self.tune(fix.representative, labels=True)
                own[fix.representative] = dict(zip(rep["names"], rep["labels"]))
        text: Dict[str, str] = {}
        want = {n for f in self.fixes for n in fl.key_names(f.key)}
        want |= {n for f in self.fixes for _, n in self.context(f)}
        want -= {n for f in self.fixes for n in own.get(f.representative, ())}
        for key, rec in self.records.items():
            if not want:
                break
            if want.intersection(rec["names"]):
                rep = self.tune(key, labels=True)
                text.update((n, t) for n, t in zip(rep["names"], rep["labels"]))
                want -= text.keys()
        for fix in self.fixes:
            mine = own.get(fix.representative, {})
            fix.labels = tuple(
                mine.get(n, text.get(n, n)) for n in fl.key_names(fix.key)
            ) + tuple(f"{t}: " + mine.get(n, n) for t, n in self.context(fix))

    def keys_of(self, rec: dict, key: str = "") -> set:
        """Candidate keys of a fresh :func:`record` under this corpus's correspondence."""
        if not rec["fail"]:
            return set()
        own = list(rec["names"])
        cols = [self.spec.index[n] for n in own if n in self.spec.index]
        more = {self.spec.names[p] for c in cols for p in self.corr.partners(c)}
        names = own + sorted(more - set(own))
        mini = fl.build_spectrum(
            [_sample(key, dict(rec, names=names), self.extras.get(key, ()))]
        )
        gid = np.array([self.spec.index.get(n, -1) for n in mini.names])
        rows = np.flatnonzero(gid >= 0)
        proj = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, gid[rows])),
            shape=(len(gid), len(self.spec.names)),
        )
        adj = (proj @ self.corr.adj.astype(np.float64) @ proj.T).tocsr() > 0
        side = np.array(
            [1 + p.startswith(self.corr.prefixes[1]) for p in mini.names], np.int8
        )
        side[[not n.startswith(self.corr.prefixes) for n in mini.names]] = 0
        corr = fl.Correspondence(mini.names, side, adj, self.corr.prefixes)
        scores = np.where(gid >= 0, self.scores[np.maximum(gid, 0)], 0.0)
        return fl.candidates(mini, corr, [_pair(rec)], scores)[0]


def _names(pair: Path) -> np.ndarray:
    return np.array((pair / "vocab.txt").read_text("utf-8").splitlines() or [""])


def analyse(work, left: str = CPU, right: str = "", progress: bool = True) -> Analysis:
    """Load a built pair, score its spectrum and rank the minimum cover of fixes."""
    pair = pair_dir(work, left, right)
    meta = json.loads((pair / "meta.json").read_text("utf-8"))
    _check_format(pair, meta)
    names = _names(pair)
    files = sorted((pair / "tunes").glob("*.npz"))
    recs = [_load(f, names) for f in tqdm(files, desc="load", disable=not progress)]
    records = {r["key"]: r for r in recs if not r["error"] and len(r["count"])}
    extras = info_extras(work)
    spec = fl.build_spectrum(
        (_sample(k, r, extras.get(k, ())) for k, r in records.items()), progress
    )
    corr = fl.correspond(spec, *meta.get("prefixes", prefixes(CPU, "python")))
    scores = fl.ochiai(spec)
    pairs = [_pair(r) for r in records.values() if r["fail"]]
    parts = fl.partings(spec, corr)
    cands = fl.candidates(spec, corr, pairs, scores, parts)
    fixes = fl.rank_fixes(spec, cands, scores)
    an = Analysis(pair, meta, records, extras, spec, corr, scores, parts, cands, fixes)
    an.label_fixes()
    return an


def fix_id(key) -> str:
    """A short stable digest of a fix key."""
    return hashlib.sha1(repr(key).encode()).hexdigest()[:12]


def explained(fixes: Sequence[fl.Fix], tops=(1, 5, 10)) -> Dict[int, int]:
    """Failing tunes the first ``n`` fixes are credited with, per ``n`` in ``tops``."""
    run = np.cumsum([0] + [f.n_tunes for f in fixes])
    return {n: int(run[min(n, len(fixes))]) for n in tops}


def write_fixes(fixes: Sequence[fl.Fix], out) -> None:
    """``fixes.csv``: :func:`~pysidtracker.faultloc.fixes_table` rows with their ``id``."""
    rows = [dict(r, id=fix_id(f.key)) for r, f in zip(fl.fixes_table(fixes), fixes)]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["rank", "id", *_FIELDS])
        w.writeheader()
        w.writerows(rows)


_FIELDS = ["kind", "key", "n_tunes", "score", "representative", "labels", "tunes"]


def _brief(labels: Sequence[str]) -> str:
    """A key's own label and its first partner's, with the other partners' count
    (``after:`` context labels left out)."""
    labels = [t for t in labels if not t.startswith("after: ")]
    more = len(labels) - 2
    return " || ".join(labels[:2]) + (f" (+{more} partners)" if more > 0 else "")


def table(fixes: Sequence[fl.Fix], records: Dict[str, dict], top: int = 20) -> str:
    """The compact fixes table: rank, tunes, score, kind, labels (a path key's own
    and its first partner's, with the partners' count), representative."""
    lines = [
        f"{'rank':>4} {'tunes':>5} {'score':>6} {'kind':<11} labels | representative"
    ]
    for i, f in enumerate(fixes[:top], 1):
        rep = records.get(f.representative, {})
        where = f"{rep.get('path', '')}:{rep.get('song', '')}" if rep else "-"
        text = _brief(f.labels) or "-"
        lines.append(
            f"{i:>4} {f.n_tunes:>5} {f.score:>6.3f} {f.key[0]:<11} {text} | {where}"
        )
    return "\n".join(lines)


# --- explanation and cases --------------------------------------------------


def diverge(
    data,
    song: int,
    left: str,
    right: str,
    mode: str = "lead",
    frames: int = 250,
    *,
    reference=None,
    name: str = "",
) -> Optional[Divergence]:
    """whydiff of the pair's first divergence over ``frames`` reference frames.

    With a 6502 side, :func:`~pysidtracker.whydiff.explain_divergence` of the other
    against ``reference`` (default jennings) under the lead criterion; else both
    Python sides' :func:`~pysidtracker.whydiff.python_side` under ``mode``. None on
    agreement.
    """
    lead = LEAD if mode == "lead" else 0
    subs = [subject(s, data, song) for s in (left, right)]
    if None in subs:
        subj = next(s for s in subs if s is not None)
        return explain_divergence(
            data,
            lambda _: subj.make(),
            reference,
            song=song,
            max_lead=lead,
            frames=frames,
            name=name,
            label=subj.label,
            cell=subj.cell,
            roots=subj.roots,
            sources=subj.sources,
            state=True,
        )
    (lg, lst), (rg, rst) = (grid(s, data, song, frames + lead) for s in (left, right))
    cut = compare(lg, rg, mode, rst or lst)
    if cut.frame is None:
        return None
    labels = [p[:-1] for p in prefixes(subs[0].label, subs[1].label)]

    def side(s, at, label, g):
        return python_side(
            s.make,
            at,
            cut.register,
            label=label,
            cell=s.cell,
            roots=s.roots,
            sources=s.sources,
            grid=g,
        )

    return explain_cell(
        side(subs[1], cut.at(True), labels[1], rg),
        side(subs[0], cut.at(False), labels[0], lg),
        (rg, lg),
        frame=cut.at(False),
        register=cut.register,
        name=name,
    )


def case_frames(rec: dict, mode: str) -> int:
    """The fewest reference frames that reproduce a record's onset verdict."""
    return int(rec["onset"]) + 1 + (LEAD if mode == "lead" else 0)


def _oracle_reference(oracle_cache) -> Callable:
    def reference(path, song, frames):
        return oracle_grid(
            path,
            oracle_cache=oracle_cache,
            frames=frames,
            song=song,
            seconds=frames // 50 + 2,
        )

    return reference


def check_case(
    entry: dict,
    *,
    hvsc_cache=None,
    oracle_cache=None,
    resolve: Callable = resolve_tune,
    reference: Optional[Callable] = None,
) -> None:
    """Run a manifest case; raise :class:`AssertionError` with whydiff's text on divergence.

    A 6502 pair compares the other side with ``reference(path, song, frames)``
    (default the sidtrace oracle, cached in ``oracle_cache``, default
    ``default_cache("csv")``); the tune comes from ``resolve`` (default
    :func:`~pysidtracker.testing.resolve_tune`, cached in ``hvsc_cache``, default
    the tune cache root); a Python pair compares
    both under the entry's ``compare``. ``patch`` bytes are applied first (the
    patched tune written beside a temporary copy for the oracle).
    """
    path = resolve(entry["relpath"], cache_dir=hvsc_cache)
    if path is None:
        raise TuneFetchError(f"{entry['relpath']} unavailable")
    raw = Path(path).read_bytes()
    if hashlib.md5(raw).hexdigest() != entry["md5"]:
        raise ChecksumMismatch(f"{entry['relpath']}: md5 differs from the case's")
    data = apply_patch(raw, entry.get("patch"))
    song, frames = int(entry["song"]), int(entry["frames"])
    ref = None
    if CPU in (entry["left"], entry["right"]):
        reference = reference or _oracle_reference(oracle_cache)
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(path)
            if entry.get("patch"):
                tune = Path(tmp) / f"{tune.stem}-{entry['id']}.sid"
                tune.write_bytes(data)
            ref = np.asarray(reference(tune, song, frames), np.int16)
    why = diverge(
        data,
        song,
        entry["left"],
        entry["right"],
        entry["compare"],
        frames,
        reference=ref,
        name=entry["relpath"],
    )
    if why is not None:
        raise AssertionError(f"{entry['relpath']}: {entry['reason']}\n{why.text()}")


def xfail_params(manifest, **match) -> list:
    """The manifest's cases (those whose fields equal ``match``) as strict-xfail
    ``pytest.param``s, each to be run by :func:`check_case`."""
    import pytest  # pylint: disable=import-outside-toplevel

    cases = load_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
    return [
        pytest.param(
            c,
            id=c["id"],
            marks=pytest.mark.xfail(
                strict=True, reason=c["reason"], raises=AssertionError
            ),
        )
        for c in cases
        if all(c.get(k) == v for k, v in match.items())
    ]


def load_manifest(path) -> List[dict]:
    """A manifest's cases (none when it does not exist)."""
    path = Path(path)
    return json.loads(path.read_text("utf-8")) if path.exists() else []


def save_manifest(path, cases: Sequence[dict]) -> None:
    """Write cases sorted by ``id``, keys sorted, one field per line."""
    out = sorted(cases, key=lambda c: c["id"])
    Path(path).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", "utf-8")


def make_case(
    an: Analysis, fix: fl.Fix, as_left: str = "", as_right: str = ""
) -> Optional[dict]:
    """The manifest case of a fix's representative, its reason from whydiff's headline;
    its sides are written as ``as_left``/``as_right`` (default the pair's specs)."""
    rec = an.tune(fix.representative)
    meta = an.meta
    data = (Path(meta["hvsc"]) / rec["path"]).read_bytes()
    entry = dict(
        id=fix_id(fix.key),
        key=repr(fix.key),
        kind=fix.key[0],
        relpath=rec["path"],
        md5=hashlib.md5(data).hexdigest(),
        song=int(rec["song"]),
        frames=case_frames(rec, meta["compare"]),
        left=as_left or meta["left"],
        right=as_right or meta["right"],
        compare=meta["compare"],
    )
    why = diverge(
        data,
        entry["song"],
        meta["left"],
        meta["right"],
        entry["compare"],
        entry["frames"],
        name=rec["path"],
    )
    if why is None:
        return None
    head = why.header()[0].partition(", ")[2]
    entry["reason"] = f"{_brief(fix.labels) or fix.key[0]}: {head}"
    return entry


def verify_case(an: Analysis, entry: dict, check: Callable[[dict], None]) -> bool:
    """Whether ``check`` fails the case and a fresh record's candidates hold its key."""
    try:
        check(entry)
        return False
    except AssertionError:
        pass
    data = apply_patch(
        (Path(an.meta["hvsc"]) / entry["relpath"]).read_bytes(), entry.get("patch")
    )
    rec = record(
        data,
        entry["song"],
        entry["left"],
        entry["right"],
        entry["compare"],
        entry["frames"],
    )
    key = f"{entry['md5']}_{entry['song']}"
    return entry["key"] in {repr(k) for k in an.keys_of(rec, key)}


def xfails(
    an: Analysis,
    manifest,
    *,
    top: int = 10,
    prune: bool = False,
    check: Optional[Callable[[dict], None]] = None,
    as_left: str = "",
    as_right: str = "",
) -> List[dict]:
    """Merge the top fixes' cases into ``manifest``; existing ids are kept as they are.

    With ``check`` (a :func:`check_case` partial) a new case is kept only when
    :func:`verify_case` holds; ``prune`` drops existing cases ``check`` passes. Cases
    name their sides ``as_left``/``as_right`` (default the pair's specs) and are
    checked with the pair's own.
    """
    meta = an.meta

    def real(case):
        alias = {as_left: meta["left"], as_right: meta["right"]}
        alias.pop("", None)
        return dict(case, **{s: alias.get(case[s], case[s]) for s in ("left", "right")})

    cases = {c["id"]: c for c in load_manifest(manifest)}
    if prune and check is not None:
        for cid, case in tqdm(list(cases.items()), desc="prune", unit="case"):
            try:
                check(real(case))
                del cases[cid]
            except AssertionError:
                pass
    fresh = [
        f for f in an.fixes[:top] if f.representative and fix_id(f.key) not in cases
    ]
    for fix in tqdm(fresh, desc="cases", unit="case"):
        entry = make_case(an, fix, as_left, as_right)
        if entry is not None and (check is None or verify_case(an, real(entry), check)):
            cases[entry["id"]] = entry
    out = list(cases.values())
    save_manifest(manifest, out)
    return out


# --- CLI --------------------------------------------------------------------


def _pair_args(p) -> None:
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--left", default=CPU, help="6502, format:NAME or a render spec")
    p.add_argument("--right", required=True, help="6502, format:NAME or a render spec")


def main(argv=None) -> int:
    """``python -m pysidtracker.pathdiff build|rank|xfails``."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build", help="record both sides' paths per subtune")
    p.add_argument("hvsc", type=Path)
    p.add_argument("tunes", type=Path)
    _pair_args(p)
    p.add_argument("--compare", choices=COMPARE, default="lead")
    p.add_argument("--verdict", action="append", help="results.csv verdicts")
    p.add_argument("--limit", type=int)
    p.add_argument("--jobs", type=int)
    p.add_argument("--max-frames", type=int, default=3000)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("rank", help="localise the corpus to the fewest fixes")
    _pair_args(p)
    p.add_argument("--out", type=Path)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--explain", type=int, default=0, help="whydiff the top N fixes")
    p = sub.add_parser("xfails", help="write the top fixes as strict xfail cases")
    _pair_args(p)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--prune", action="store_true")
    p.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--oracle-cache", type=Path, help="default: default_cache('csv')")
    p.add_argument("--as-left", default="", help="the left spec cases are written with")
    p.add_argument(
        "--as-right", default="", help="the right spec cases are written with"
    )
    args = ap.parse_args(argv)
    if args.command == "build":
        tally = build(
            args.hvsc,
            args.tunes,
            args.work,
            args.left,
            args.right,
            mode=args.compare,
            verdicts=tuple(args.verdict or VERDICTS),
            limit=args.limit,
            jobs=args.jobs,
            max_frames=args.max_frames,
            timeout=args.timeout,
            force=args.force,
        )
        print(dict(tally))
        return 0
    an = analyse(args.work, args.left, args.right)
    if args.command == "rank":
        write_fixes(an.fixes, args.out or an.pair / "fixes.csv")
        print(table(an.fixes, an.records, args.top))
        nfail = int(an.spec.fail.sum())
        cover = explained(an.fixes)
        print(
            f"{nfail} differing tunes; "
            + ", ".join(f"top {n}: {k}" for n, k in cover.items())
        )
        _explain(an, args.explain)
        return 0
    hvsc = Path(an.meta["hvsc"])
    check = None
    if args.verify or args.prune:
        oracle = args.oracle_cache or default_cache("csv")
        Path(oracle).mkdir(parents=True, exist_ok=True)

        def check(entry):
            check_case(
                entry,
                oracle_cache=oracle,
                resolve=lambda rel, cache_dir: hvsc / rel,
            )

    cases = xfails(
        an,
        args.manifest,
        top=args.top,
        prune=args.prune,
        check=check,
        as_left=args.as_left,
        as_right=args.as_right,
    )
    print(f"{len(cases)} cases in {args.manifest}")
    return 0


def _explain(an: Analysis, top: int) -> None:
    """``explain/<rank>.txt``: whydiff of each of the top fixes' representative."""
    out = an.pair / "explain"
    out.mkdir(exist_ok=True)
    meta = an.meta
    for rank, fix in enumerate(tqdm(an.fixes[:top], desc="explain"), 1):
        if not fix.representative:
            continue
        rec = an.tune(fix.representative)
        data = (Path(meta["hvsc"]) / rec["path"]).read_bytes()
        try:
            why = diverge(
                data,
                int(rec["song"]),
                meta["left"],
                meta["right"],
                meta["compare"],
                case_frames(rec, meta["compare"]),
                name=rec["path"],
            )
            text = "agrees" if why is None else why.text()
        except Exception as exc:  # pylint: disable=broad-except
            text = _err(exc)
        head = f"fix {rank} {fix_id(fix.key)} {fix.key!r}\n{' || '.join(fix.labels)}\n"
        (out / f"{rank}.txt").write_text(head + "\n" + text + "\n", "utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
