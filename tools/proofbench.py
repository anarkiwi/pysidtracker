"""Time the whole-call proofs of reordered tunes, one process per tune.

``python tools/proofbench.py TUNES.csv [--python PY] [--jobs N] [--budget S] [--only TEXT ...]``:
per tune the proof seconds and verdict from the report, the wall time, peak resident
memory and the cores used (CPU seconds over wall seconds) of the whole process.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from tqdm import tqdm


def run(
    python: str, tune: str, frames: int, budget: float, jobs: int, kill: int
) -> dict:
    """The row for one tune."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.json"
        cmd = [
            "timeout",
            str(kill),
            python,
            "-c",
            "from pysidtracker.maincli import main; main()",
        ]
        cmd += ["playspec", "reorder", tune, "--frames", str(frames)]
        cmd += ["--prove", "10000", "--budget", str(budget), "--out", str(out)]
        cmd += ["--proof-jobs", str(jobs)] if jobs > 1 else []
        start = time.monotonic()
        with subprocess.Popen(cmd, stdout=subprocess.DEVNULL) as proc:
            _pid, _status, usage = os.wait4(proc.pid, 0)
            proc.returncode = 0
        wall = time.monotonic() - start
        report = json.loads(out.read_text()) if out.exists() else {}
    programs = next(iter(report.values()), {}).get("programs", [])
    proofs = [p["proof"] for p in programs if "proof" in p]
    return {
        "tune": Path(tune).stem,
        "programs": len(proofs),
        "proved": sum(p["holds"] for p in proofs),
        "obligations": sum(sum(p["obligations"].values()) for p in proofs),
        "proof_s": round(sum(p["seconds"] for p in proofs), 1),
        "wall_s": round(wall, 1),
        "peak_mb": usage.ru_maxrss // 1024,
        "cores": round((usage.ru_utime + usage.ru_stime) / wall, 1),
    }


def main() -> None:
    """Run the tunes and print the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tunes")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--budget", type=float, default=300.0)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--kill", type=int, default=3600, help="seconds per tune")
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()
    with open(args.tunes, newline="", encoding="utf-8") as rows:
        paths = [r["path"] for r in csv.DictReader(rows)]
    paths = [p for p in paths if not args.only or any(o in p for o in args.only)]
    rows = [
        run(args.python, p, args.frames, args.budget, args.jobs, args.kill)
        for p in tqdm(paths, desc="tunes", unit="tune")
    ]
    print("\t".join(rows[0]))
    for row in rows:
        print("\t".join(map(str, row.values())))


if __name__ == "__main__":
    main()
