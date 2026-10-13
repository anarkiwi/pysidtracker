"""Whole-song HVSC survey of a format's renderer against the sidplayfp oracle.

``find`` lists tunes by SIDId signature, ``run`` compares every subtune's oracle
and ``module:function`` renderer grids, ``report`` tallies, ``diff`` shows frames,
``triage`` classifies first differences and renderer ``info`` exact rates.
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
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import zstandard
from tqdm import tqdm

from . import registers
from .errors import SidParseError
from .header import parse_sid_header
from .oracle import (
    SidtraceUnavailable,
    read_sidtrace,
    run_sidtrace,
    sidtrace_cadence,
    sidtrace_grid,
)
from .sidid import identify, sidid_signatures

LEAD = 8
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
ORACLE_ARGS = ("-f8000", "--sidlite")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
TRIAGE_FIELDS = ["key", "path", "player", "song", "first_diff", "when", "kind"]
TRIAGE_FIELDS += ["roles", "diffs"]
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
    """``(nframes, grid)`` of a cached sidtrace CSV over ``seconds`` at ``clock``.

    ``nframes`` overrides the frame count derived from ``seconds``.
    """
    rows = read_sidtrace(ocsv)
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


def compare(oracle, lib):
    """``(lead, bad, first)``: the lead of ``lib`` with the latest first mismatch."""
    best = None
    for lead in sorted(range(-min(LEAD, len(oracle) - 1), LEAD + 1), key=abs):
        bad = np.not_equal(*aligned(oracle, lib, lead)).any(1)
        first = int(np.argmax(bad)) if bad.any() else len(bad)
        if best is None or first > best[2]:
            best = (lead, bad, first)
    return best


def gated(grid):
    """Whether any voice's gate bit is ever set."""
    return bool((np.asarray(grid)[:, GATE_COLS] & 1).any())


def _err(exc):
    return f"{type(exc).__name__}: {exc}"


def check(job):
    """One subtune's oracle-vs-renderer comparison row (see ``RESULT_FIELDS``)."""
    sid_path, tune, song, seconds, work, render = job
    key = f"{tune['md5']}_{song}"
    row = dict(key=key, path=tune["path"], player=tune["player"], song=song)
    row["seconds"] = seconds
    ocsv, lcsv = (Path(work) / d / f"{key}.csv.zst" for d in ("oracle", "library"))
    try:
        if not ocsv.exists():
            run_sidtrace(
                sid_path,
                ocsv,
                seconds=math.ceil(seconds) + 1,
                song=song,
                user=f"{os.getuid()}:{os.getgid()}",
                extra_args=ORACLE_ARGS,
            )
        nframes, oracle = oracle_frames(ocsv, seconds, tune["clock"])
        row["frames"] = nframes
    except (SidtraceUnavailable, OSError, ValueError) as e:
        return dict(row, verdict="oracle_error", error=_err(e))
    try:
        lib, info = render(Path(sid_path).read_bytes(), song, nframes + LEAD)
        lib = np.asarray(lib, dtype=np.int16)
        if lib.ndim != 2 or lib.shape[1] != NREGS or len(lib) < nframes + LEAD:
            raise ValueError(f"render grid shape {lib.shape}, need {nframes + LEAD}")
        write_grid(lib, lcsv)
    except SidParseError as e:
        return dict(row, verdict="not_decompiled", error=str(e))
    except Exception as e:  # pylint: disable=broad-except
        return dict(row, verdict="library_error", error=_err(e))
    lead, bad, first = compare(oracle, lib)
    regs = np.not_equal(*aligned(oracle, lib, lead))[bad].any(0)
    return dict(
        row,
        verdict="differs" if bad.any() else "exact",
        lead=lead,
        first_diff=first if bad.any() else "",
        diff_frames=int(bad.sum()),
        diff_regs=" ".join(REG_NAMES[r] for r in np.flatnonzero(regs)),
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


def jobs_for(hvsc, tunes, work, render, default_seconds=180):
    """One :func:`check` job per (tune, song), timed by its HVSC song length."""
    for tune in tunes:
        lens = [float(x) for x in tune["lengths"].split()]
        for song in range(1, int(tune["songs"]) + 1):
            secs = lens[song - 1] if song <= len(lens) else default_seconds
            yield Path(hvsc) / tune["path"], tune, song, secs, Path(work), render


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
):
    """Compare every subtune; write ``results.csv`` under ``work``, return tallies."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    tunes = [
        t for t in load_tunes(tunes_csv) if re.search(player or "", t["player"], re.I)
    ]
    jobs_ = list(jobs_for(hvsc, tunes[:limit], work, render, default_seconds))
    tally = Counter()
    with (
        pool(jobs) as ex,
        open(work / "results.csv", "w", newline="", encoding="utf-8") as f,
    ):
        w = csv.DictWriter(f, RESULT_FIELDS)
        w.writeheader()
        bar = tqdm(ex.map(check, jobs_), total=len(jobs_), unit="song")
        for row in bar:
            w.writerow(row)
            tally[row["verdict"]] += 1
            bar.set_postfix(tally)
    return tally


def report(results_csv):
    """Print tallies by player/verdict, differing registers and errors."""
    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    verdicts = Counter((r["player"], r["verdict"]) for r in rows)
    print(verdicts.most_common())
    differs = [r for r in rows if r["verdict"] == "differs"]
    print("diff regs:", Counter(r["diff_regs"] for r in differs).most_common(20))
    errs = Counter(r["error"][:70] for r in rows if r["error"])
    print("errors:", errs.most_common(20))
    return verdicts


def diff(hvsc, tunes_csv, key, work, render, context=3):
    """Print oracle (O) and renderer (L) rows around one subtune's first difference."""
    work = Path(work)
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


def first_difference(work, row):
    """``(frame, [(reg, oracle, renderer)])`` at a ``differs`` row's first mismatch."""
    nframes, first = int(row["frames"]), int(row["first_diff"])
    _, oracle = oracle_frames(
        Path(work) / "oracle" / f"{row['key']}.csv.zst", nframes=nframes
    )
    lib = read_grid(Path(work) / "library" / f"{row['key']}.csv.zst", nframes + LEAD)
    o, l = (g[first] for g in aligned(oracle, lib, int(row["lead"])))
    return first, [(int(r), int(o[r]), int(l[r])) for r in np.flatnonzero(o != l)]


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
    frame, diffs = first_difference(work, row)
    when, kind, roles = classify(frame, diffs, row)
    out = {k: row[k] for k in ("key", "path", "player", "song")}
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


def triage(work, jobs=None, top=15, max_values=20):
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
    rates = info_rates(rows, max_values)
    print("renderer info vs exact rate:")
    for (k, v), (n, hit) in sorted(rates.items(), key=lambda kv: -kv[1][0])[:top]:
        print(f"{n:7d} {100 * hit / n:5.1f}% exact  {k}={v}")
    return out, rates


def main(argv=None):
    """``python -m pysidtracker.survey {find,run,report,diff,triage}``."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("find", help=find.__doc__)
    p.add_argument("hvsc")
    p.add_argument("--sidid-cfg", help="sidid.cfg (default: the bundled one)")
    p.add_argument("--out", required=True)
    p.add_argument("--player", help="case-insensitive regex on the SIDId player name")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    for name, fn in (("run", run), ("diff", diff)):
        p = sub.add_parser(name, help=fn.__doc__)
        p.add_argument("hvsc")
        p.add_argument("tunes")
        p.add_argument("--work", required=True)
        p.add_argument("--render", required=True, help="module:function renderer")
    sub.choices["diff"].add_argument("key")
    sub.choices["diff"].add_argument("--context", type=int, default=3)
    p = sub.choices["run"]
    p.add_argument("--player", help="case-insensitive regex on the SIDId player name")
    p.add_argument("--limit", type=int, help="only the first N tunes")
    p.add_argument("--default-seconds", type=float, default=180)
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    p = sub.add_parser("report", help=report.__doc__)
    p.add_argument("results")
    p = sub.add_parser("triage", help=triage.__doc__)
    p.add_argument("work")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    p.add_argument("--top", type=int, default=15)
    a = ap.parse_args(argv)
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
        )
    elif a.cmd == "diff":
        out = diff(a.hvsc, a.tunes, a.key, a.work, load_render(a.render), a.context)
    elif a.cmd == "triage":
        triage(a.work, a.jobs, a.top)
    else:
        out = report(a.results)
    if a.cmd in ("find", "run"):
        print(dict(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
