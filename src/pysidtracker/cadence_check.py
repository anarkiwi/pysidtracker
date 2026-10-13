"""Check :func:`~pysidtracker.cadence.playroutine_cadence` against sidplayfp interrupts.

``run`` traces subtune 1 of each tune in survey ``tunes.csv`` files under the
sidtrace oracle and compares the rate of the interrupt timing its SID writes to
a cadence function's prediction; ``report`` tallies results by player.
"""

import argparse
import csv
import os
import sys
from collections import Counter
from concurrent.futures import as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import registers as reg
from .header import parse_sid_header
from .oracle import SidtraceUnavailable, read_sidtrace, run_sidtrace
from .survey import _err, deadline, load_render, load_tunes, oracle_cache, pool

SOURCES = ("video_irq", "cia_irq", "nmi")
PREDICTED = {"pal_video": "video_irq", "ntsc_video": "video_irq"}
PREDICTED.update(cia_timer="cia_irq", nmi_timer="nmi")
FIELDS = ["key", "path", "player", "clock", "verdict", "pred_source", "pred_cpf"]
FIELDS += ["pred_cpc", "dynamic", "meas_source", "meas_cpf", "calls", "span"]
FIELDS += ["gaps", "raises", "error"]
DEFAULT_CADENCE = "pysidtracker.cadence:playroutine_cadence"


def interrupt_cadence(rows):
    """``(source, raises)``: the interrupt source whose raises time the SID writes.

    Each write is attributed to the most recent raise of any source; the source
    with the most writes wins. ``(None, [])`` if no write follows a raise.
    """
    times = {
        s: np.array([r.cycle for r in rows if r.value == s and r.chip is None])
        for s in SOURCES
    }
    writes = np.array([r.cycle for r in rows if r.chip is not None], dtype=np.int64)
    latest = np.full((len(SOURCES), len(writes)), -1, dtype=np.int64)
    for i, t in enumerate(times.values()):
        if t.size:
            idx = np.searchsorted(t, writes, side="right") - 1
            latest[i] = np.where(idx >= 0, t[idx.clip(0)], -1)
    owned = latest.max(0) >= 0
    if not owned.any():
        return None, np.array([], dtype=np.int64)
    source = SOURCES[np.bincount(latest.argmax(0)[owned]).argmax()]
    return source, times[source]


def measure(rows, frame):
    """The measured source timing the writes and its calls per ``frame`` cycles.

    The first raise is dropped: one raised during init is held until init ends.
    """
    source, raises = interrupt_cadence(rows)
    steady = raises[1:]
    out = {"meas_source": source or "", "raises": len(raises)}
    if len(steady) < 2:
        return out
    gaps = np.diff(steady)
    span = int(steady[-1] - steady[0])
    return dict(
        out,
        calls=len(gaps),
        span=span,
        meas_cpf=f"{frame * len(gaps) / span:.4f}",
        gaps=len(np.unique(gaps)),
    )


def verdict(pred_source, pred_cpf, meas, frame):
    """``agree`` if the source and the call count over the measured span match.

    The count may differ by one frame's calls, the boundary of the span.
    """
    if not meas.get("calls"):
        return "no_raises"
    if pred_source is None:
        return "unknown"
    if PREDICTED[pred_source] != meas["meas_source"]:
        return "source"
    expect = meas["span"] * pred_cpf / frame
    return "agree" if abs(expect - meas["calls"]) <= max(1.0, pred_cpf) else "rate"


def check(job):
    """One tune's comparison row (see ``FIELDS``)."""
    sid_path, tune, work, cadence, seconds, timeout = job
    data = Path(sid_path).read_bytes()
    ntsc = (parse_sid_header(data).flags >> 2) & 3 == 2
    frame = reg.NTSC_CYCLES_PER_FRAME if ntsc else reg.PAL_CYCLES_PER_FRAME
    row = {"key": f"{tune['md5']}_1", "path": tune["path"], "player": tune["player"]}
    row["clock"] = "NTSC" if ntsc else "PAL"
    ocsv = Path(work) / "oracle" / f"{row['key']}.csv.zst"
    try:
        if not ocsv.exists():
            run_sidtrace(
                sid_path,
                ocsv,
                seconds=seconds,
                song=1,
                user=f"{os.getuid()}:{os.getgid()}",
                extra_args=("-f8000",),
            )
        meas = measure(read_sidtrace(ocsv), frame)
    except (SidtraceUnavailable, OSError, ValueError) as e:
        return dict(row, verdict="oracle_error", error=_err(e))
    row.update(meas)
    try:
        with deadline(timeout):
            got = load_render(cadence)(data, subtune=0)
    except Exception as e:  # pylint: disable=broad-except
        return dict(row, verdict="cadence_error", error=_err(e))
    source = got.source.value if got.source is not None else None
    cpf = getattr(got, "calls_per_frame", frame / got.cycles_per_call)
    row.update(pred_source=source or "", pred_cpf=f"{cpf:.4f}")
    row.update(pred_cpc=got.cycles_per_call, dynamic=got.dynamic)
    return dict(row, verdict=verdict(source, cpf, meas, frame))


def run(hvsc, tunes_csvs, work, *, cadence=DEFAULT_CADENCE, out=None, **kw):
    """Check every distinct tune's subtune 1; write ``out``, return verdict tallies.

    ``kw``: ``seconds`` traced (10), ``jobs``, ``timeout`` per cadence call (300).
    """
    work = Path(work)
    oracle_cache(work)
    tunes = {t["md5"]: t for path in tunes_csvs for t in load_tunes(path)}
    jobs = [
        (Path(hvsc) / t["path"], t, work, cadence, kw.get("seconds", 10))
        + (kw.get("timeout", 300),)
        for t in tunes.values()
    ]
    tally = Counter()
    with (
        pool(kw.get("jobs")) as ex,
        open(out or work / "cadence.csv", "w", newline="", encoding="utf-8") as f,
    ):
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        bar = tqdm(as_completed([ex.submit(check, j) for j in jobs]), total=len(jobs))
        for done in bar:
            row = done.result()
            w.writerow(row)
            tally[row["verdict"]] += 1
            bar.set_postfix(tally)
    return tally


def report(results_csv):
    """Print and return ``{player: Counter(verdict)}`` with agreement rates."""
    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        out.setdefault(row["player"], Counter())[row["verdict"]] += 1
    for player, tally in sorted(out.items(), key=lambda kv: -kv[1].total()):
        rate = 100 * tally["agree"] / tally.total()
        print(f"{tally.total():6d} {rate:5.1f}% agree  {player}  {dict(tally)}")
    return out


def main(argv=None):
    """``python -m pysidtracker.cadence_check {run,report}``."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help=run.__doc__)
    p.add_argument("hvsc")
    p.add_argument("tunes", nargs="+", help="survey tunes.csv files")
    p.add_argument("--work", required=True)
    p.add_argument("--out", help="results CSV (default WORK/cadence.csv)")
    p.add_argument("--cadence", default=DEFAULT_CADENCE, help="module:function")
    p.add_argument("--seconds", type=int, default=10)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    p = sub.add_parser("report", help=report.__doc__)
    p.add_argument("results")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        kw = {"seconds": a.seconds, "timeout": a.timeout, "jobs": a.jobs}
        print(dict(run(a.hvsc, a.tunes, a.work, cadence=a.cadence, out=a.out, **kw)))
    else:
        report(a.results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
