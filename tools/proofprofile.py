"""Profile the whole-call proof of a tune's reordered programs.

``python tools/proofprofile.py TUNE.sid [--timeout-ms N] [--budget S] [--cprofile OUT]``:
per program the obligations, the time inside z3 ``check`` against the rest, and the
distribution of time per obligation.
"""

import argparse
import cProfile
import pstats
import time
from pathlib import Path

import numpy as np
import z3

from pysidtracker.formats import find_format
from pysidtracker.playspec import outorder as O
from pysidtracker.playspec import outproof as P


def programs(path: str, order, frames: int):
    """The reordered programs of a tune: ``(program, memory, observed)`` each."""
    data = Path(path).read_bytes()
    fmt = find_format(data)
    model = fmt.load(data, None)
    _verdict, results = O.compare(
        lambda transform: fmt.spec_player(model, transform=transform), order, frames
    )
    return [r.source for r in results]


def timed_obligations():
    """Patch the checker to record each obligation's total and solver seconds."""
    records, solving = [], [0.0]
    check = z3.Solver.check

    def timed_check(self, *a):
        t = time.perf_counter()
        try:
            return check(self, *a)
        finally:
            solving[0] += time.perf_counter() - t

    obligation = P._Checker.obligation

    def timed_obligation(self, item):
        s0, t = solving[0], time.perf_counter()
        try:
            return obligation(self, item)
        finally:
            records.append((item[0], time.perf_counter() - t, solving[0] - s0))

    z3.Solver.check = timed_check
    P._Checker.obligation = timed_obligation
    return records


def summarize(records, wall: float) -> None:
    """Print the split and the per-obligation distribution."""
    total = np.array([r[1] for r in records])
    solve = np.array([r[2] for r in records])
    print(f"wall {wall:.1f}s obligations {len(records)}")
    print(f"in obligations {total.sum():.1f}s of which z3 check {solve.sum():.1f}s")
    for kind in sorted({r[0] for r in records}):
        sel = np.array([r[0] == kind for r in records])
        print(
            f"  {kind}: n={sel.sum()} total={total[sel].sum():.1f}s "
            f"solve={solve[sel].sum():.1f}s"
        )
    q = np.quantile(total, [0.5, 0.9, 0.99, 1.0])
    print("per-obligation s p50/p90/p99/max:", " ".join(f"{v:.4f}" for v in q))
    top = np.sort(total)[::-1][:10]
    print("top10 share of obligation time:", f"{top.sum() / total.sum():.2f}")


def main() -> None:
    """Run the proof of each program of a tune and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tune")
    parser.add_argument("--timeout-ms", type=int, default=10_000)
    parser.add_argument("--budget", type=float, default=120.0)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--cprofile")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    progs = programs(args.tune, O.ORDER, args.frames)
    records = timed_obligations()
    profiler = cProfile.Profile() if args.cprofile else None
    start = time.perf_counter()
    for program, memory, observed in progs:
        if profiler:
            profiler.enable()
        proof = P.prove(
            program,
            O.ORDER,
            memory=memory,
            observed=observed,
            timeout_ms=args.timeout_ms,
            budget_s=args.budget,
            progress=True,
            jobs=args.jobs,
        )
        if profiler:
            profiler.disable()
        print(
            f"holds={proof.holds} open={proof.open[:3]} "
            f"obligations={ {k: d.total for k, d in proof.passes.items()} } "
            f"{proof.seconds:.1f}s {proof.peak_mb}MB"
        )
    summarize(records, time.perf_counter() - start)
    if profiler:
        profiler.dump_stats(args.cprofile)
        pstats.Stats(args.cprofile).sort_stats("cumulative").print_stats(35)


if __name__ == "__main__":
    main()
