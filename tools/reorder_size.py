"""Statement counts and seconds of ``outorder.reorder`` over the tunes of a format's playspec.

``python tools/reorder_size.py LIST.txt [--format MODULE[:ATTR]] [--spec F.play] [--prove]``
where LIST.txt holds one tune path per line (``--limit`` bounds the tunes taken).
"""

import argparse
import dataclasses
import statistics
import time
from importlib import import_module
from pathlib import Path

from tqdm import tqdm

from pysidtracker.formats import find_format
from pysidtracker.playspec import load
from pysidtracker.playspec import outorder as O
from pysidtracker.playspec import outproof


def measure(raw: bytes, fmt, spec, prove: bool) -> dict:
    """Counts of the first play entry of a tune and the seconds reorder (and prove) took."""
    if spec is not None:
        text = load(spec)
        fmt = dataclasses.replace(
            fmt, spec=lambda m, base=fmt.spec: dataclasses.replace(base(m), spec=text)
        )
    seen = {}

    def capture(program, memory):
        start = time.time()
        seen["result"] = (
            O.reorder(program, memory=memory)
            if program.outorder
            else O.reorder(program, O.ORDER, memory=memory)
        )
        seen["reorder_s"] = time.time() - start
        if prove:
            start = time.time()
            order = program.outorder or O.ORDER
            seen["proof"] = outproof.prove(program, order, memory=memory)
            seen["prove_s"] = time.time() - start
        return program

    fmt.spec_player(
        fmt.load(raw, None), notes=None if prove else True, transform=capture
    ).play_frame()
    result = seen["result"]
    every = list(O.statements(result.program.body)) + [
        s for p in result.program.procs.values() for s in O.statements(p)
    ]
    nodes = sum(1 for s in every for e in O.stmt_exprs(s) for _ in O.walk(e))
    return {
        "before": result.before.statements,
        "inlined": result.inlined.statements,
        "after": len(every),
        "nodes": nodes,
        "procs": len(result.program.procs),
        "staged": len(result.staged),
        "native": len(result.native),
        "reorder_s": seen["reorder_s"],
        "prove_s": seen.get("prove_s"),
        "holds": seen["proof"].holds if "proof" in seen else None,
    }


def main(argv=None) -> int:
    """Print one row per tune and the medians."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("names", type=Path)
    parser.add_argument(
        "--format", help="MODULE[:ATTR], else the format that detects each tune"
    )
    parser.add_argument("--spec")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--prove", action="store_true")
    args = parser.parse_args(argv)
    paths = [n.strip() for n in args.names.read_text().splitlines() if n.strip()]
    rows = []
    for path in tqdm(paths[: args.limit], unit="tune"):
        raw = Path(path).read_bytes()
        if args.format:
            module, _, attr = args.format.partition(":")
            fmt = getattr(import_module(module), attr or "FORMAT")
        else:
            fmt = find_format(raw)
        row = measure(raw, fmt, args.spec, args.prove)
        rows.append(row)
        print(
            Path(path).name,
            {k: round(v, 2) for k, v in row.items() if isinstance(v, (int, float))},
        )
    for key in rows[0]:
        vals = [r[key] for r in rows if r[key] is not None]
        if vals and not isinstance(vals[0], bool):
            print(
                f"{key:10s} median {statistics.median(vals):10.2f}  max {max(vals):10.2f}"
            )
        elif vals:
            print(f"{key:10s} {vals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
