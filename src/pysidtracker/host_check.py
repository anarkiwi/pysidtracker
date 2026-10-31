"""Check :class:`~pysidtracker.irqhost.InterruptHost` traces row-for-row against sidtrace.

``run`` traces each tune's subtunes under the oracle and the host and records
the first row that differs; ``report`` tallies; ``trace`` prints each instruction
the host fetches in a cycle window.
"""

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from .irqhost import InterruptHost
from .oracle import SidtraceUnavailable, cached_sidtrace
from .jobs import deadline, results
from .survey import ORACLE_ARGS, _err, load_tunes, oracle_cache

FIELDS = ["key", "path", "song", "verdict", "rows", "first", "cycle"]
FIELDS += ["oracle_row", "host_row", "error"]


def _row(r):
    return "" if r is None else f"{r.cycle} {r.chip} {r.reg} {r.value}"


def first_difference(want, got):
    """Index of the first differing row of two traces, ``None`` when equal."""
    n = min(len(want), len(got))
    first = next((i for i in range(n) if want[i] != got[i]), None)
    return n if first is None and len(want) != len(got) else first


def _job_row(job):
    """The identifying fields of a :func:`check` job's row."""
    _, tune, song, *_ = job
    return {"key": f"{tune['md5']}_{song}", "path": tune["path"], "song": song}


def _failed(job, error):
    """The row of a :func:`check` job whose worker raised or died."""
    return dict(_job_row(job), verdict="host_error", error=_err(error))


def check(job):
    """One subtune's comparison row (see ``FIELDS``)."""
    sid_path, _, song, work, seconds, timeout = job
    row = _job_row(job)
    key = row["key"]
    try:
        want = cached_sidtrace(
            sid_path,
            Path(work) / "oracle" / f"{key}.csv.zst",
            seconds=int(seconds) + 1,
            song=song,
            user=f"{os.getuid()}:{os.getgid()}",
            extra_args=ORACLE_ARGS,
        )
    except (SidtraceUnavailable, OSError, ValueError) as e:
        return dict(row, verdict="oracle_error", error=_err(e))
    try:
        with deadline(
            timeout,
            dict(row, verdict="host_error", error=f"TimeoutError: exceeded {timeout}s"),
        ):
            host = InterruptHost(Path(sid_path).read_bytes(), song - 1)
            end = host.start + int(seconds * host.clock_hz)
            host.run(end + 1 - host.cycle)
            got = [r for r in host.rows() if r.cycle <= end]
    except Exception as e:  # pylint: disable=broad-except
        return dict(row, verdict="host_error", error=_err(e))
    want = [r for r in want if r.cycle <= end]
    first = first_difference(want, got)
    row["rows"] = len(want)
    if first is None:
        return dict(row, verdict="match")
    pair = [rows[first] if first < len(rows) else None for rows in (want, got)]
    cycle = min(r.cycle for r in pair if r is not None)
    return dict(row, verdict="differs", first=first, cycle=cycle) | {
        "oracle_row": _row(pair[0]),
        "host_row": _row(pair[1]),
    }


def run(hvsc, tunes_csvs, work, *, seconds=10.0, jobs=None, timeout=3600, out=None):
    """Compare every subtune of every tune; write ``out``, return verdict tallies."""
    work = Path(work)
    oracle_cache(work)
    tunes = {t["md5"]: t for path in tunes_csvs for t in load_tunes(path)}
    work_items = [
        (Path(hvsc) / t["path"], t, song, work, seconds, timeout)
        for t in tunes.values()
        for song in range(1, int(t["songs"]) + 1)
    ]
    tally = Counter()
    with open(out or work / "host_check.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        rows = results(check, work_items, jobs, _failed)
        bar = tqdm(rows, total=len(work_items), unit="song")
        for result in bar:
            writer.writerow(result)
            tally[result["verdict"]] += 1
            bar.set_postfix(tally)
    return tally


def report(results_csv):
    """Print and return the verdict tally, then each differing subtune."""
    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tally = Counter(r["verdict"] for r in rows)
    print(dict(tally))
    for r in rows:
        if r["verdict"] != "match":
            print(r["verdict"], r["path"], r["song"], r["first"], r["error"])
            print("   oracle", r["oracle_row"], "| host", r["host_row"])
    return tally


def trace(data, start, end, song=0, out=None):
    """Print each instruction the host fetches from cycle ``start`` to ``end``."""
    from jennings.devices.mpu6502 import MPU
    from jennings.disassembler import Disassembler

    out = out or sys.stdout
    host = InterruptHost(data, song)
    machine = host.machine
    dis = Disassembler(MPU(memory=machine))

    def traced(*_):
        mpu = machine.mpu
        if start <= machine.t <= end:
            text = dis.instruction_at(mpu.pc)[1]
            print(
                f"{machine.t:>10} {mpu.pc:04X} {text:<16} A={mpu.a:02X} X={mpu.x:02X} "
                f"Y={mpu.y:02X} SP={mpu.sp:02X} P={mpu.p:02X} "
                f"$01={machine.port_data:02X}",
                file=out,
            )

    if hasattr(machine, "set_decode_hook"):
        machine.set_decode_hook(traced)
    else:
        decode = machine._decode  # pylint: disable=protected-access

        def decoded():
            traced()
            decode()

        machine._decode = decoded  # pylint: disable=protected-access
    host.run(max(0, end - host.cycle))
    return host


def main(argv=None):
    """``python -m pysidtracker.host_check {run,report,trace}``."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help=run.__doc__)
    p.add_argument("hvsc")
    p.add_argument("tunes", nargs="+", help="survey tunes.csv files")
    p.add_argument("--work", required=True)
    p.add_argument("--out", help="results CSV (default WORK/host_check.csv)")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    p = sub.add_parser("report", help=report.__doc__)
    p.add_argument("results")
    p = sub.add_parser("trace", help=trace.__doc__)
    p.add_argument("sid")
    p.add_argument("start", type=int)
    p.add_argument("end", type=int)
    p.add_argument("--song", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        kw = {"seconds": a.seconds, "jobs": a.jobs, "timeout": a.timeout}
        print(dict(run(a.hvsc, a.tunes, a.work, out=a.out, **kw)))
    elif a.cmd == "report":
        report(a.results)
    else:
        trace(Path(a.sid).read_bytes(), a.start, a.end, a.song - 1)
    return 0


__all__ = ["check", "first_difference", "main", "report", "run", "trace"]

if __name__ == "__main__":
    sys.exit(main())
