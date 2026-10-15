"""Whole-song HVSC survey of a format's renderer against the sidplayfp oracle.

``find`` lists tunes by SIDId signature, ``run`` compares every subtune's oracle
and ``module:function`` renderer grids, ``report`` tallies, ``diff`` shows frames,
``triage`` classifies first differences and renderer ``info`` exact rates,
``leads`` tabulates every alignment lead's mismatches.
"""

import argparse
import csv
import hashlib
import importlib
import io
import json
import math
import multiprocessing
import os
import re
import signal
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import zstandard
from tqdm import tqdm

from . import registers
from .disasm import entry_labels
from .errors import SidParseError
from .header import parse_sid_header
from .image import SidImage
from .oracle import (
    SidtraceUnavailable,
    cached_sidtrace,
    read_sidtrace,
    sidtrace_cadence,
    sidtrace_grid,
    sidtrace_image_id,
)
from .sidid import identify, sidid_signatures

LEAD = 8
NEAR = 3
NREGS = registers.SID_REG_COUNT
VOICE_ROLES = ("freq", "freq", "pw", "pw", "ctrl", "ad", "sr")
GATE_COLS = [off + registers.CTRL for off in registers.SID_VOICE_OFFSET]
ROLES = [n for _ in range(registers.SID_VOICES) for n in VOICE_ROLES] + [
    "fc",
    "fc",
    "res",
    "vol",
]
REG_NAMES = [
    f"v{v + 1}_{n}"
    for v in range(registers.SID_VOICES)
    for n in ("freq_lo", "freq_hi", "pw_lo", "pw_hi", "ctrl", "ad", "sr")
] + ["fc_lo", "fc_hi", "res_filt", "mode_vol"]
ORACLE_ARGS = ("-f8000",)
ORACLE_ROWS = ("raise", "entry", "rti")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
TRIAGE_FIELDS = ["key", "path", "player", "song", "first_diff", "when", "kind"]
TRIAGE_FIELDS += ["roles", "diffs", "min_lead", "min_bad", "min_first", "min_regs"]
LEAD_FIELDS = ["key", "path", "song", "frames", "lead", "first_diff", "bad", "regs"]
TUNE_FIELDS = ["path", "md5", "player", "songs", "clock", "lengths"]
RESULT_FIELDS = [
    "key",
    "path",
    "player",
    "song",
    "seconds",
    "frames",
    "verdict",
    "lead",
    "first_diff",
    "diff_frames",
    "diff_regs",
    "min_lead",
    "min_bad",
    "min_first",
    "min_regs",
    "info",
    "oracle_gated",
    "library_gated",
    "error",
]
_FIND = {}


def songlengths(hvsc_root):
    """``md5 -> [seconds per song]`` from HVSC's ``DOCUMENTS/Songlengths.md5``."""
    out = {}
    text = (Path(hvsc_root) / "DOCUMENTS" / "Songlengths.md5").read_text("latin-1")
    for line in text.splitlines():
        md5, sep, lens = line.partition("=")
        if sep and len(md5) == 32:
            out[md5] = [
                int(m) * 60 + int(s) + int((ms or "0").ljust(3, "0")) / 1000
                for m, s, ms in re.findall(r"(\d+):(\d+)(?:\.(\d+))?", lens)
            ]
    return out


def pool(jobs, initializer=None, initargs=()):
    """A spawn-context process pool whose workers start with one BLAS thread."""
    for var in THREAD_VARS:
        os.environ.setdefault(var, "1")
    return ProcessPoolExecutor(
        jobs,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initializer,
        initargs=initargs,
    )


def _find_init(cfg, player_regex):
    """Worker state for :func:`find`: the signatures and player filter."""
    _FIND.update(
        sigs=sidid_signatures(cfg), player=re.compile(player_regex or "", re.I)
    )


def _find_one(path):
    data = Path(path).read_bytes()
    name = identify(data, _FIND["sigs"])
    if name is None or not _FIND["player"].search(name):
        return None
    hdr = parse_sid_header(data)
    clock = "NTSC" if (hdr.flags >> 2) & 3 == 2 else "PAL"
    return hashlib.md5(data).hexdigest(), name, hdr.songs, clock


def find(hvsc, out_csv, cfg=None, player_regex=None, jobs=None):
    """Write the tunes SIDId names a ``player_regex`` player; return player tallies.

    ``cfg`` defaults to the bundled ``sidid.cfg`` (see :mod:`pysidtracker.sidid`).
    """
    hvsc = Path(hvsc)
    lengths = songlengths(hvsc)
    paths = sorted(hvsc.rglob("*.sid"))
    tally = Counter()
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with (
        pool(jobs, _find_init, (cfg, player_regex)) as ex,
        open(out_csv, "w", newline="", encoding="utf-8") as f,
    ):
        w = csv.writer(f)
        w.writerow(TUNE_FIELDS)
        hits = ex.map(_find_one, paths, chunksize=64)
        for path, hit in zip(paths, tqdm(hits, total=len(paths), unit="sid")):
            if hit:
                md5, name, songs, clock = hit
                tally[name] += 1
                lens = " ".join(f"{x:g}" for x in lengths.get(md5, []))
                w.writerow([path.relative_to(hvsc), md5, name, songs, clock, lens])
    return tally


def write_grid(grid, out):
    """A register grid as a zstd CSV of per-frame register changes."""
    grid = np.asarray(grid)
    change = np.ones(grid.shape, bool)
    change[1:] = grid[1:] != grid[:-1]
    f, r = np.nonzero(change)
    buf = io.StringIO()
    buf.write("frame,reg,value\n")
    np.savetxt(buf, np.column_stack([f, r, grid[f, r]]), fmt="%d", delimiter=",")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(zstandard.ZstdCompressor().compress(buf.getvalue().encode()))


def read_grid(path, nframes):
    """Inverse of :func:`write_grid`, forward-filled to ``nframes``."""
    text = zstandard.ZstdDecompressor().decompress(Path(path).read_bytes()).decode()
    rows = np.loadtxt(io.StringIO(text), np.int64, delimiter=",", skiprows=1, ndmin=2)
    f, r, v = rows[rows[:, 0] < nframes].T
    grid = np.zeros((nframes, NREGS), dtype=np.int16)
    src = np.zeros((nframes, NREGS), dtype=np.int64)
    grid[f, r], src[f, r] = v, f
    np.maximum.accumulate(src, axis=0, out=src)
    return np.take_along_axis(grid, src, 0)


def oracle_frames(ocsv, seconds=0.0, clock="PAL", nframes=None):
    """``(nframes, grid)`` of a sidtrace CSV (or its rows) over ``seconds`` at ``clock``.

    ``nframes`` overrides the frame count derived from ``seconds``.
    """
    rows = read_sidtrace(ocsv) if isinstance(ocsv, (str, Path)) else ocsv
    cpf = sidtrace_cadence(rows) or registers.PAL_CYCLES_PER_FRAME
    hz = registers.NTSC_CLOCK_HZ if clock == "NTSC" else registers.PAL_CLOCK_HZ
    nframes = int(seconds * hz / cpf) if nframes is None else nframes
    grid = np.asarray(sidtrace_grid(rows, cycles_per_frame=cpf), np.int16)
    grid = grid.reshape(-1, NREGS) if grid.size else np.zeros((1, NREGS), np.int16)
    return nframes, grid[np.minimum(np.arange(nframes), len(grid) - 1)]


def aligned(oracle, lib, lead):
    """Equal-length ``(oracle, lib)`` views, ``lib`` shifted by ``lead`` frames."""
    if lead >= 0:
        return oracle, lib[lead : lead + len(oracle)]
    return oracle[-lead:], lib[: len(oracle) + lead]


def scan_leads(oracle, lib):
    """Mismatches of ``lib`` against ``oracle`` at every lead, in ``|lead|`` order.

    Returns ``(leads, ne, size, first, nbad)``: ``ne[i, f, r]`` is whether register
    ``r`` differs on aligned frame ``f`` at ``leads[i]`` (False past that lead's
    ``size[i]`` aligned frames), ``first[i]`` the first bad frame (``size[i]``
    when none) and ``nbad[i]`` the count of bad frames.
    """
    n = len(oracle)
    leads = np.array(sorted(range(-min(LEAD, n - 1), LEAD + 1), key=abs))
    frames = np.arange(n)
    size = n - np.maximum(-leads, 0)
    o = np.minimum(frames + np.maximum(-leads, 0)[:, None], n - 1)
    l = frames + np.maximum(leads, 0)[:, None]
    ne = (oracle[o] != lib[l]) & (frames < size[:, None])[..., None]
    bad = ne.any(2)
    first = np.where(bad.any(1), bad.argmax(1), size)
    return leads, ne, size, first, bad.sum(1)


def reg_names(mask):
    """Space-separated :data:`REG_NAMES` of a boolean register mask."""
    return " ".join(REG_NAMES[r] for r in np.flatnonzero(mask))


def compare(oracle, lib):
    """``(lead, bad, first)``: the lead of ``lib`` with the latest first mismatch."""
    leads, ne, size, first, _ = scan_leads(oracle, lib)
    i = int(np.argmax(first))
    return int(leads[i]), ne[i, : size[i]].any(1), int(first[i])


def min_lead(scan):
    """``min_lead``/``min_bad``/``min_first``/``min_regs`` of a :func:`scan_leads`.

    The lead with the fewest bad frames (the smallest ``|lead|`` on ties), its
    bad-frame count and first bad frame, and the registers differing on them.
    """
    leads, ne, size, first, nbad = scan
    i = int(np.argmin(nbad))
    return dict(
        min_lead=int(leads[i]),
        min_bad=int(nbad[i]),
        min_first=int(first[i]) if nbad[i] else "",
        min_regs=reg_names(ne[i, : size[i]].any(0)),
    )


def verdict_fields(scan):
    """The ``results.csv`` comparison columns of a :func:`scan_leads`.

    ``lead`` has the latest first mismatch; ``min_lead`` the fewest bad frames.
    """
    leads, ne, size, first, nbad = scan
    i = int(np.argmax(first))
    return dict(
        verdict="differs" if nbad[i] else "exact",
        lead=int(leads[i]),
        first_diff=int(first[i]) if nbad[i] else "",
        diff_frames=int(nbad[i]),
        diff_regs=reg_names(ne[i, : size[i]].any(0)),
        **min_lead(scan),
    )


def gated(grid):
    """Whether any voice's gate bit is ever set."""
    return bool((np.asarray(grid)[:, GATE_COLS] & 1).any())


def _err(exc):
    return f"{type(exc).__name__}: {exc}"


def oracle_cache(work):
    """``work/oracle``, emptied when its traces came from another sidtrace setup.

    The stamp holds the image id, the arguments and the logged row kinds.
    """
    cache = Path(work) / "oracle"
    cache.mkdir(parents=True, exist_ok=True)
    image = sidtrace_image_id()
    stamp = cache / "STAMP"
    want = json.dumps({"image": image, "args": ORACLE_ARGS, "rows": ORACLE_ROWS})
    if image is None or (stamp.exists() and stamp.read_text("utf-8") == want):
        return cache
    stale = list(cache.glob("*.csv.zst"))
    if stale:
        print(f"oracle: re-rendering {len(stale)} stale traces", file=sys.stderr)
    for path in stale:
        path.unlink()
    stamp.write_text(want, "utf-8")
    return cache


def oracle_rows(sid_path, ocsv, seconds, song):
    """The sidtrace rows of one subtune, cached at ``ocsv``."""
    return cached_sidtrace(
        sid_path,
        ocsv,
        seconds=math.ceil(seconds) + 1,
        song=song,
        user=f"{os.getuid()}:{os.getgid()}",
        extra_args=ORACLE_ARGS,
    )


@contextmanager
def deadline(seconds):
    """Raise :class:`TimeoutError` in the main thread after ``seconds`` (0: never)."""

    def _expire(*_):
        raise TimeoutError(f"render exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def check(job):
    """One subtune's oracle-vs-renderer comparison row (see ``RESULT_FIELDS``)."""
    sid_path, tune, song, seconds, work, render, timeout = job
    key = f"{tune['md5']}_{song}"
    row = dict(key=key, path=tune["path"], player=tune["player"], song=song)
    row["seconds"] = seconds
    ocsv, lcsv = (Path(work) / d / f"{key}.csv.zst" for d in ("oracle", "library"))
    try:
        nframes, oracle = oracle_frames(
            oracle_rows(sid_path, ocsv, seconds, song), seconds, tune["clock"]
        )
        row["frames"] = nframes
    except (SidtraceUnavailable, OSError, ValueError) as e:
        return dict(row, verdict="oracle_error", error=_err(e))
    try:
        with deadline(timeout):
            lib, info = render(Path(sid_path).read_bytes(), song, nframes + LEAD)
        lib = np.asarray(lib, dtype=np.int16)
        if lib.ndim != 2 or lib.shape[1] != NREGS or len(lib) < nframes + LEAD:
            raise ValueError(f"render grid shape {lib.shape}, need {nframes + LEAD}")
        write_grid(lib, lcsv)
    except SidParseError as e:
        return dict(row, verdict="not_decompiled", error=str(e))
    except Exception as e:  # pylint: disable=broad-except
        return dict(row, verdict="library_error", error=_err(e))
    return dict(
        row,
        **verdict_fields(scan_leads(oracle, lib)),
        info=json.dumps(info, sort_keys=True, default=str),
        oracle_gated=gated(oracle),
        library_gated=gated(lib),
    )


def load_tunes(path):
    """The rows of a ``find`` CSV."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_render(spec):
    """The ``render`` callable named by ``module:function``."""
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


def jobs_for(hvsc, tunes, work, render, default_seconds=180, timeout=600):
    """One :func:`check` job per (tune, song), timed by its HVSC song length."""
    for tune in tunes:
        lens = [float(x) for x in tune["lengths"].split()]
        for song in range(1, int(tune["songs"]) + 1):
            secs = lens[song - 1] if song <= len(lens) else default_seconds
            yield Path(hvsc) / tune["path"], tune, song, secs, Path(
                work
            ), render, timeout


def run(
    hvsc,
    tunes_csv,
    work,
    render,
    *,
    player=None,
    limit=None,
    jobs=None,
    default_seconds=180,
    timeout=600,
):
    """Compare every subtune; write ``results.csv`` under ``work``, return tallies."""
    work = Path(work)
    oracle_cache(work)
    tunes = [
        t for t in load_tunes(tunes_csv) if re.search(player or "", t["player"], re.I)
    ]
    jobs_ = list(jobs_for(hvsc, tunes[:limit], work, render, default_seconds, timeout))
    tally = Counter()
    with (
        pool(jobs) as ex,
        open(work / "results.csv", "w", newline="", encoding="utf-8") as f,
    ):
        w = csv.DictWriter(f, RESULT_FIELDS)
        w.writeheader()
        futures = [ex.submit(check, job) for job in jobs_]
        bar = tqdm(as_completed(futures), total=len(jobs_), unit="song")
        for done in bar:
            row = done.result()
            w.writerow(row)
            tally[row["verdict"]] += 1
            bar.set_postfix(tally)
    return tally


def near_exact(rows, near=NEAR, top=20):
    """Print and return the ``differs`` rows with at most ``near`` bad frames at
    their ``min_lead`` (rows without ``min_bad``, from older surveys, are skipped).
    """
    hits = [
        r
        for r in rows
        if r["verdict"] == "differs" and r.get("min_bad") and int(r["min_bad"]) <= near
    ]
    print(f"near-exact (<= {near} bad frames at min_lead): {len(hits)}")
    groups = {}
    for r in hits:
        group = (r["min_bad"], r["min_first"], r["min_lead"], r["min_regs"])
        groups.setdefault(group, []).append(r["path"])
    for (nbad, first, lead, regs), paths in sorted(
        groups.items(), key=lambda kv: -len(kv[1])
    )[:top]:
        where = f"bad={nbad} first={first} lead={lead}"
        print(f"{len(paths):7d} {where:28} {regs:28} e.g. {', '.join(paths[:2])}")
    return hits


def report(results_csv, near=NEAR):
    """Print tallies by player/verdict, differing registers, near-exact subtunes
    and errors."""
    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    verdicts = Counter((r["player"], r["verdict"]) for r in rows)
    print(verdicts.most_common())
    differs = [r for r in rows if r["verdict"] == "differs"]
    print("diff regs:", Counter(r["diff_regs"] for r in differs).most_common(20))
    near_exact(rows, near)
    errs = Counter(r["error"][:70] for r in rows if r["error"])
    print("errors:", errs.most_common(20))
    return verdicts


def diff(hvsc, tunes_csv, key, work, render, context=3):
    """Print oracle (O) and renderer (L) rows around one subtune's first difference."""
    work = Path(work)
    oracle_cache(work)
    md5, song = key.split("_")
    tune = next(t for t in load_tunes(tunes_csv) if t["md5"] == md5)
    job = next(j for j in jobs_for(hvsc, [tune], work, render) if j[2] == int(song))
    row = check(job)
    print(row)
    if row["verdict"] not in ("exact", "differs"):
        return row
    _, oracle = oracle_frames(work / "oracle" / f"{key}.csv.zst", job[3], tune["clock"])
    lib = read_grid(work / "library" / f"{key}.csv.zst", len(oracle) + LEAD)
    oracle, lib = aligned(oracle, lib, row["lead"])
    first = row["first_diff"] or 0
    for f in range(max(0, first - context), min(len(lib), first + context + 1)):
        o, l = oracle[f], lib[f]
        print(f"{f:6d} O {' '.join(f'{x:02x}' for x in o)}")
        print(f"{'':6} L {' '.join(f'{x:02x}' for x in l)}")
        if (o != l).any():
            print(f"{'':8} " + " ".join("^^" if x != y else "  " for x, y in zip(o, l)))
    return row


def _hex_addr(value):
    """A hex C64 address: ``1000``, ``0x1000``, or ``$1000``."""
    text = value[1:] if value[:1] == "$" else value
    try:
        return int(text, 16)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a hex address, got {value!r}"
        ) from exc


def _render_value(render, sid_path, song, frame, reg):
    """The library renderer's value of ``reg`` at ``frame`` (the side-by-side compare)."""
    lib, _info = render(Path(sid_path).read_bytes(), song, frame + LEAD + 1)
    grid = np.asarray(lib, dtype=np.int16)
    if grid.size == 0 or grid.ndim < 2:
        return None
    return int(grid[min(frame, len(grid) - 1)][reg])


def explain(
    hvsc,
    tunes_csv,
    key,
    work,
    *,
    frame,
    reg=None,
    watch=(),
    render=None,
    depth=8,
):
    """Run the tune's own 6502 play routine to ``frame`` and explain its SID writes.

    For each write in frame ``frame`` (or just ``reg``), prints the writing
    instruction with its disassembly, the load chain that produced the value, and the
    frame/cycle; with ``--watch`` also prints the chosen RAM cells per frame. When
    ``--reg`` and a ``render`` hook are given, the library's rendered value for that
    ``(frame, reg)`` appears next to the real player's write.
    """
    work = Path(work)
    md5, song = key.split("_")
    tune = next(t for t in load_tunes(tunes_csv) if t["md5"] == md5)
    sid_path = Path(hvsc) / tune["path"]
    image = SidImage.from_bytes(sid_path.read_bytes())
    from .provenance import explain as _explain
    from .provenance import format_explain, format_watch, trace_frames

    labels = entry_labels(image)
    ntsc = tune.get("clock", "PAL") == "NTSC"
    pro = trace_frames(image, frame + 1, watch=list(watch), labels=labels, ntsc=ntsc)
    if watch:
        print(format_watch(pro, list(watch)))
        print()
    print(format_explain(pro, frame, reg=reg, depth=depth))
    if render is not None and reg is not None:
        real = next((w.value for w in _explain(pro, frame, reg=reg)), None)
        libv = _render_value(render, sid_path, int(song), frame, int(reg))
        if libv is not None or real is not None:
            real_note = "" if real is None else f"   real ${real:02X}"
            lib_note = "n/a" if libv is None else f"${libv:02X}"
            print(f"library (frame {frame}, reg {reg}): {lib_note}{real_note}")
    return pro


def cached_grids(work, key, nframes):
    """``(oracle, lib)`` grids of a subtune's cached ``work`` traces."""
    _, oracle = oracle_frames(Path(work) / "oracle" / f"{key}.csv.zst", nframes=nframes)
    lib = read_grid(Path(work) / "library" / f"{key}.csv.zst", nframes + LEAD)
    return oracle, lib


def _diffs_at(oracle, lib, lead, frame):
    o, l = (g[frame] for g in aligned(oracle, lib, lead))
    return [(int(r), int(o[r]), int(l[r])) for r in np.flatnonzero(o != l)]


def _lead_rows(job):
    """``(summary, rows)``: one subtune's :func:`verdict_fields` and per-lead rows.

    Reuses the cached oracle trace and library grid, rendering only what is missing.
    """
    sid_path, tune, song, seconds, work, render, _ = job
    key = f"{tune['md5']}_{song}"
    ocsv, lcsv = (work / d / f"{key}.csv.zst" for d in ("oracle", "library"))
    nframes, oracle = oracle_frames(
        oracle_rows(sid_path, ocsv, seconds, song), seconds, tune["clock"]
    )
    if lcsv.exists():
        lib = read_grid(lcsv, nframes + LEAD)
    elif render is None:
        raise ValueError(f"{lcsv} is not cached and no --render was given")
    else:
        lib = np.asarray(
            render(Path(sid_path).read_bytes(), song, nframes + LEAD)[0], np.int16
        )
        write_grid(lib, lcsv)
    scan = scan_leads(oracle, lib)
    leads, ne, _, first, nbad = scan
    at_first = ne[np.arange(len(leads)), np.minimum(first, nframes - 1)]
    row = dict(key=key, path=tune["path"], song=song, frames=nframes)
    rows = [
        dict(row, lead=int(leads[i]), first_diff=int(first[i]) if nbad[i] else "")
        | dict(bad=int(nbad[i]), regs=reg_names(at_first[i]))
        for i in np.argsort(leads)
    ]
    return dict(row, **verdict_fields(scan)), rows


def leads(hvsc, tunes_csv, work, render=None, *, keys=(), verdicts=(), **kw):
    """Tabulate every lead's first mismatch, bad frames and first-frame registers.

    One key prints its table. Otherwise the keys, plus the ``results.csv``
    subtunes with a verdict in ``verdicts``, are scanned with ``kw['jobs']``
    processes into ``kw['out']`` (default ``work/leads.csv``).
    """
    work = Path(work)
    keys = list(keys)
    if verdicts:
        results = load_tunes(work / "results.csv")
        keys += [r["key"] for r in results if r["verdict"] in verdicts]
    wanted = {tuple(k.split("_")) for k in keys}
    tunes = [t for t in load_tunes(tunes_csv) if t["md5"] in {m for m, _ in wanted}]
    jobs_ = [
        j
        for j in jobs_for(hvsc, tunes, work, render)
        if (j[1]["md5"], str(j[2])) in wanted
    ]
    if len(keys) == 1 and not verdicts:
        summary, rows = _lead_rows(jobs_[0])
        print(
            f"{summary['key']} {summary['path']} song {summary['song']}: "
            f"{summary['frames']} frames, {summary['verdict']}, "
            f"lead {summary['lead']} (*, latest first mismatch), "
            f"min_lead {summary['min_lead']} (<, fewest bad frames)"
        )
        print(f"{'lead':>6} {'first':>6} {'bad':>6}  regs at first bad frame")
        for r in rows:
            mark = ("*" if r["lead"] == summary["lead"] else " ") + (
                "<" if r["lead"] == summary["min_lead"] else " "
            )
            print(
                f"{r['lead']:4d}{mark} {r['first_diff']:>6} {r['bad']:6d}  {r['regs']}"
            )
        return [summary]
    out = Path(kw.get("out") or work / "leads.csv")
    summaries = []
    with (
        pool(kw.get("jobs")) as ex,
        open(out, "w", newline="", encoding="utf-8") as f,
    ):
        w = csv.DictWriter(f, LEAD_FIELDS)
        w.writeheader()
        for summary, rows in tqdm(
            ex.map(_lead_rows, jobs_), total=len(jobs_), unit="song"
        ):
            w.writerows(rows)
            summaries.append(summary)
    moved = Counter(
        (s["lead"], s["min_lead"]) for s in summaries if s["lead"] != s["min_lead"]
    )
    print(f"{len(summaries)} subtunes, lead != min_lead:", moved.most_common(20))
    near_exact(summaries, kw.get("near", NEAR))
    return summaries


def first_difference(work, row):
    """``(frame, [(reg, oracle, renderer)])`` at a ``differs`` row's first mismatch."""
    first = int(row["first_diff"])
    grids = cached_grids(work, row["key"], int(row["frames"]))
    return first, _diffs_at(*grids, int(row["lead"]), first)


def classify(frame, diffs, row):
    """``(when, kind, roles)`` of a first difference (see docs/survey.md)."""
    when = (
        "frame0"
        if frame == 0
        else "early" if frame < 10 else "mid" if frame < 250 else "late"
    )
    roles = " ".join(sorted({ROLES[r] for r, _, _ in diffs}))
    wrong = {r: (o, l) for r, o, l in diffs}
    deltas = {l - o for o, l in wrong.values()}
    if row.get("library_gated") == "False" and row.get("oracle_gated") == "True":
        kind = "silent"
    elif any(
        o and any(l2 == o for r2, (_, l2) in wrong.items() if r2 != r)
        for r, (o, _) in wrong.items()
    ):
        kind = "misplaced"
    elif len(wrong) > 1 and len(deltas) == 1:
        kind = f"offset{deltas.pop():+#x}"
    elif all(not l for _, l in wrong.values()):
        kind = "missing"
    elif all(not o for o, _ in wrong.values()):
        kind = "extra"
    else:
        kind = "value"
    return when, kind, roles


def _triage_one(args):
    work, row = args
    oracle, lib = cached_grids(work, row["key"], int(row["frames"]))
    frame = int(row["first_diff"])
    diffs = _diffs_at(oracle, lib, int(row["lead"]), frame)
    when, kind, roles = classify(frame, diffs, row)
    out = {k: row[k] for k in ("key", "path", "player", "song")}
    out.update(min_lead(scan_leads(oracle, lib)))
    return dict(out, first_diff=frame, when=when, kind=kind, roles=roles, diffs=diffs)


def info_rates(rows, max_values=20):
    """``{(key, value): (rendered, exact)}`` over each renderer ``info`` field."""
    tally = Counter()
    for r in rows:
        if r["verdict"] in ("exact", "differs"):
            for k, v in json.loads(r["info"] or "{}").items():
                tally[k, json.dumps(v), r["verdict"] == "exact"] += 1
    values = Counter(k for k, _ in {(k, v) for k, v, _ in tally})
    rates = {}
    for (k, v, exact), n in tally.items():
        if values[k] <= max_values:
            done, hit = rates.get((k, v), (0, 0))
            rates[k, v] = (done + n, hit + n * exact)
    return rates


def triage(work, jobs=None, top=15, max_values=20, near=NEAR):
    """Classify every ``differs`` first mismatch into ``triage.csv``; print tables."""
    work = Path(work)
    with open(work / "results.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    differs = [r for r in rows if r["verdict"] == "differs"]
    with (
        pool(jobs) as ex,
        open(work / "triage.csv", "w", newline="", encoding="utf-8") as f,
    ):
        w = csv.DictWriter(f, TRIAGE_FIELDS)
        w.writeheader()
        out = []
        hits = ex.map(_triage_one, ((work, r) for r in differs), chunksize=8)
        for t in tqdm(hits, total=len(differs), unit="song"):
            w.writerow(dict(t, diffs=json.dumps(t["diffs"])))
            out.append(t)
    print("verdicts:", Counter(r["verdict"] for r in rows).most_common())
    print("when:", Counter(t["when"] for t in out).most_common())
    print("kind:", Counter(t["kind"] for t in out).most_common(top))
    examples = {}
    for t in out:
        examples.setdefault((t["kind"], t["roles"]), []).append(t["path"])
    print("kind / roles at the first difference:")
    for (kind, roles), paths in sorted(examples.items(), key=lambda kv: -len(kv[1]))[
        :top
    ]:
        print(f"{len(paths):7d} {kind:14} {roles:28} e.g. {', '.join(paths[:2])}")
    near_exact([dict(t, verdict="differs") for t in out], near, top)
    rates = info_rates(rows, max_values)
    print("renderer info vs exact rate:")
    for (k, v), (n, hit) in sorted(rates.items(), key=lambda kv: -kv[1][0])[:top]:
        print(f"{n:7d} {100 * hit / n:5.1f}% exact  {k}={v}")
    return out, rates


def main(argv=None):
    """``python -m pysidtracker.survey {find,run,report,diff,triage,leads,explain}``."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("find", help=find.__doc__)
    p.add_argument("hvsc")
    p.add_argument("--sidid-cfg", help="sidid.cfg (default: the bundled one)")
    p.add_argument("--out", required=True)
    p.add_argument("--player", help="case-insensitive regex on the SIDId player name")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    for name, fn in (
        ("run", run),
        ("diff", diff),
        ("leads", leads),
        ("explain", explain),
    ):
        p = sub.add_parser(name, help=fn.__doc__)
        p.add_argument("hvsc")
        p.add_argument("tunes")
        p.add_argument("--work", required=True)
        p.add_argument(
            "--render",
            required=name not in ("leads", "explain"),
            help="module:function renderer (leads: only for uncached grids)",
        )
    sub.choices["diff"].add_argument("key")
    sub.choices["diff"].add_argument("--context", type=int, default=3)
    pe = sub.choices["explain"]
    pe.add_argument("key", help="<md5>_<song> subtune key")
    pe.add_argument("--frame", type=int, required=True, help="frame to explain")
    pe.add_argument("--reg", type=int, help="restrict to this SID register offset")
    pe.add_argument(
        "--watch",
        action="append",
        type=_hex_addr,
        default=[],
        help="a RAM cell ($XXXX) to watch per frame (repeatable)",
    )
    pe.add_argument("--depth", type=int, default=8, help="load-chain back-track depth")
    p = sub.choices["run"]
    p.add_argument("--player", help="case-insensitive regex on the SIDId player name")
    p.add_argument("--limit", type=int, help="only the first N tunes")
    p.add_argument("--default-seconds", type=float, default=180)
    p.add_argument("--timeout", type=float, default=600, help="render seconds")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    p = sub.choices["leads"]
    p.add_argument("keys", nargs="*", help="<md5>_<song> subtune keys")
    p.add_argument(
        "--all", action="store_true", help="every exact/differs subtune in results.csv"
    )
    p.add_argument(
        "--verdict", action="append", default=[], help="results.csv verdict to scan"
    )
    p.add_argument("--out", help="leads CSV (default: WORK/leads.csv)")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    p = sub.add_parser("report", help=report.__doc__)
    p.add_argument("results")
    p = sub.add_parser("triage", help=triage.__doc__)
    p.add_argument("work")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    p.add_argument("--top", type=int, default=15)
    for name in ("leads", "report", "triage"):
        sub.choices[name].add_argument(
            "--near", type=int, default=NEAR, help="near-exact bad-frame limit"
        )
    a = ap.parse_args(argv)
    if a.cmd == "leads" and not (a.keys or a.all or a.verdict):
        ap.error("leads needs keys, --all or --verdict")
    if a.cmd == "find":
        out = find(a.hvsc, a.out, a.sidid_cfg, a.player, a.jobs)
    elif a.cmd == "run":
        out = run(
            a.hvsc,
            a.tunes,
            a.work,
            load_render(a.render),
            player=a.player,
            limit=a.limit,
            jobs=a.jobs,
            default_seconds=a.default_seconds,
            timeout=a.timeout,
        )
    elif a.cmd == "diff":
        out = diff(a.hvsc, a.tunes, a.key, a.work, load_render(a.render), a.context)
    elif a.cmd == "explain":
        explain(
            a.hvsc,
            a.tunes,
            a.key,
            a.work,
            frame=a.frame,
            reg=a.reg,
            watch=a.watch,
            render=a.render and load_render(a.render),
            depth=a.depth,
        )
    elif a.cmd == "leads":
        leads(
            a.hvsc,
            a.tunes,
            a.work,
            a.render and load_render(a.render),
            keys=a.keys,
            verdicts=("exact", "differs") if a.all else a.verdict,
            out=a.out,
            jobs=a.jobs,
            near=a.near,
        )
    elif a.cmd == "triage":
        triage(a.work, a.jobs, a.top, near=a.near)
    else:
        out = report(a.results, a.near)
    if a.cmd in ("find", "run"):
        print(dict(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
