"""Native against Python playspec execution over tunes: register grids and per-frame writes.

``python tools/spec_native_diff.py [TUNES.csv ...] [--census CENSUS.json] [--frames N]
[--write-frames N] [--jobs J] [--root DIR]`` where each CSV has a ``path`` column (default
``examples/output-order/tunes.csv``) and the census (``examples/output-order/census.json``) lists
emitted specs with the tunes they are bound to, which play those specs instead of the format's.
Each tune is played for ``--frames`` frames by ``SpecPlayer(native=False)`` and
``SpecPlayer(native=True)``; the grids must be identical, and for the first ``--write-frames``
frames so must ``SpecPlayer.writes`` of every frame. Prints one line per tune (microseconds per
frame of each, native bails) and, for a difference, the format, tune, frame and register of the
first. Exit status is the number of differing tunes.
"""

import argparse
import csv
import dataclasses
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

os.environ.setdefault("OMP_NUM_THREADS", "2")


def players(path: Path, example: str = ""):
    """``(format, python player, native player)`` of the tune at ``path``, playing the spec in
    the file ``example`` if given."""
    from pysidtracker import find_format  # pylint: disable=import-outside-toplevel
    from pysidtracker.playspec import load  # pylint: disable=import-outside-toplevel

    raw = path.read_bytes()
    fmt = find_format(raw)
    model = fmt.parser.parse(raw)
    if example:
        spec = load(example)
        binding = fmt.spec
        fmt = dataclasses.replace(
            fmt,
            spec=lambda m: dataclasses.replace(
                binding(m), spec=spec, bind=lambda view, entry: {}
            ),
        )
    made = [fmt.spec_player(model, notes=True, native=flag) for flag in (False, True)]
    return fmt.name, *made


def first_difference(python, native) -> str:
    """Where two register grids first differ, or the empty string."""
    for frame, (a, b) in enumerate(zip(python, native)):
        for reg, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return f"frame {frame} register {reg}: python {x} native {y}"
    return "" if len(python) == len(native) else f"lengths {len(python)} {len(native)}"


def check(args) -> dict:
    """The result row of one tune."""
    path, frames, write_frames, example = args
    row = {
        "path": str(path),
        "format": "",
        "diff": "",
        "py_us": 0.0,
        "native_us": 0.0,
        "bails": 0,
    }
    try:
        row["format"], python, native = players(path, example)
        start = time.perf_counter()
        grid_py = python.render_grid(frames)
        row["py_us"] = (time.perf_counter() - start) / frames * 1e6
        start = time.perf_counter()
        grid_nat = native.render_grid(frames)
        row["native_us"] = (time.perf_counter() - start) / frames * 1e6
        row["bails"] = native._bails  # pylint: disable=protected-access
        row["diff"] = first_difference(grid_py, grid_nat)
        if not row["diff"] and write_frames:
            python, native = players(path, example)[1:]
            for frame in range(write_frames):
                python.play_frame()
                native.play_frame()
                if python.writes != native.writes:
                    row["diff"] = (
                        f"frame {frame} writes: python {python.writes} native {native.writes}"
                    )
                    break
    except Exception as exc:  # pylint: disable=broad-exception-caught
        row["diff"] = f"error {type(exc).__name__}: {str(exc)[:100]}"
    return row


def main() -> int:
    """Run the comparison; the number of tunes that differ."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tunes", nargs="*")
    parser.add_argument("--frames", type=int, default=3000)
    parser.add_argument("--write-frames", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--root", default="/scratch/datasets/hvsc/C64Music")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--census", default="")
    args = parser.parse_args()
    paths = []
    for name in args.tunes or (
        [] if args.census else ["examples/output-order/tunes.csv"]
    ):
        with open(name, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                path = Path(row["path"])
                paths.append(path if path.is_absolute() else Path(args.root) / path)
    paths = paths[: args.limit or None]
    work = [(p, args.frames, args.write_frames, "") for p in paths]
    if args.census:
        for entry in json.loads(Path(args.census).read_text(encoding="utf-8")):
            work += [
                (Path(t), args.frames, args.write_frames, entry["file"])
                for t in entry["tunes"]
            ]
    bad = 0
    with ProcessPoolExecutor(args.jobs) as pool:
        for row in tqdm(pool.map(check, work), total=len(work), unit="tune"):
            bad += bool(row["diff"])
            print(
                f"{row['format']:15s} {Path(row['path']).name:52s} py {row['py_us']:6.1f}"
                f" native {row['native_us']:6.1f} us/frame bails {row['bails']} {row['diff']}",
                file=sys.stderr if row["diff"] else sys.stdout,
            )
    return bad


if __name__ == "__main__":
    sys.exit(main())
