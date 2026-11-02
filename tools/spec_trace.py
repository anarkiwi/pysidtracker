"""Record a playspec's per-call SID output over tunes, or check another spec against a recording.

``spec_trace.py record|check REF.pkl.gz SPEC.play TUNES.csv [--frames N] [--jobs J]``; a check
compares what is audible (docs/spec-migration.md) and exits with the number of tunes that differ.
"""

import argparse
import csv
import dataclasses
import gzip
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

UNUSED = {3: 0x0F, 10: 0x0F, 17: 0x0F, 21: 0x07}


def trace(job):
    """``(path, [(writes, registers)] per call)``, or ``(path, refusal text)``."""
    path, spec, frames = job
    # pylint: disable=import-outside-toplevel
    from pysidtracker import find_format
    from pysidtracker.playspec import load

    try:
        raw = Path(path).read_bytes()
        fmt = find_format(raw)
        alt = load(spec)
        use = dataclasses.replace(
            fmt, spec=lambda m: dataclasses.replace(fmt.spec(m), spec=alt)
        )
        player = use.spec_player(fmt.parser.parse(raw), native=True)
        calls = []
        for _ in range(frames):
            player.play_frame()
            calls.append((tuple(player.writes), tuple(player.snapshot())))
        return path, calls
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return path, f"refused: {type(exc).__name__}: {str(exc)[:100]}"


def audible(calls) -> list:
    """Per call: the registers, bits the chip ignores masked, and per voice whether the gate
    was cleared and set again and whether TEST was set within the call."""
    out, ctrl = [], [0, 0, 0]
    for writes, regs in calls:
        flags = []
        for v in range(3):
            seen = [ctrl[v]] + [val for reg, val in writes if reg == 4 + 7 * v]
            end = regs[4 + 7 * v]
            flags.append(
                (
                    bool(end & 1) and any(not s & 1 for s in seen),
                    any(s & 8 for s in seen),
                )
            )
            ctrl[v] = end
        out.append(
            (tuple(r & UNUSED.get(i, 0xFF) for i, r in enumerate(regs)), tuple(flags))
        )
    return out


def differs(want, have):
    """None if two traces agree, else where they first differ."""
    if isinstance(want, str) or isinstance(have, str):
        if isinstance(want, str) == isinstance(have, str):
            return None
        return (
            "refusal",
            *(v if isinstance(v, str) else "plays" for v in (want, have)),
        )
    for call, (a, b) in enumerate(zip(audible(want), audible(have))):
        if a != b:
            return ("call", call, "registers" if a[0] != b[0] else "edges", a, b)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("mode", choices=("record", "check"))
    parser.add_argument("ref")
    parser.add_argument("spec")
    parser.add_argument("tunes")
    parser.add_argument("--frames", type=int, default=3000)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    with open(args.tunes, encoding="utf-8") as handle:
        paths = [row["path"] for row in csv.DictReader(handle)]
    jobs = [(p, args.spec, args.frames) for p in paths]
    with ProcessPoolExecutor(args.jobs) as pool:
        got = dict(tqdm(pool.map(trace, jobs), total=len(jobs), unit="tune"))
    if args.mode == "record":
        with gzip.open(args.ref, "wb") as handle:
            pickle.dump(got, handle)
        print(sum(not isinstance(v, str) for v in got.values()), "of", len(got), "play")
        return 0
    with gzip.open(args.ref, "rb") as handle:
        ref = pickle.load(handle)
    bad = [
        (p, d) for p, have in got.items() if (d := differs(ref[p], have)) is not None
    ]
    for path, where in bad:
        print(Path(path).name, *where)
    print("differ", len(bad), "of", len(got))
    return len(bad)


if __name__ == "__main__":
    sys.exit(main())
