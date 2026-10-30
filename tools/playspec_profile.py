"""Per-stage time and memory of a playspec: parse, elaborate, source, compile."""

import argparse
import json
import time
import tracemalloc
from pathlib import Path

from pysidtracker.playspec.lower import elaborate
from pysidtracker.playspec.pyexec import source
from pysidtracker.playspec.syntax import load


def stages(path, binding):
    """Yield ``(stage, seconds, peak_bytes, result)`` for each compile stage."""
    state = {}
    plan = (
        ("parse", lambda: load(path)),
        ("elaborate", lambda: elaborate(state["parse"], binding)),
        ("source", lambda: source(state["elaborate"])),
        ("compile", lambda: compile(state["source"], "<profile>", "exec")),
    )
    for name, fn in plan:
        tracemalloc.start()
        t0 = time.perf_counter()
        state[name] = fn()
        elapsed = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        yield name, elapsed, peak, state[name]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("spec")
    ap.add_argument("--binding", help="JSON layout binding")
    args = ap.parse_args()
    binding = json.loads(Path(args.binding).read_text()) if args.binding else None
    for name, sec, peak, res in stages(args.spec, binding):
        extra = f" lines={res.count(chr(10))}" if name == "source" else ""
        print(f"{name:10s} {sec:7.2f}s peak {peak / 2**20:8.1f} MiB{extra}")


if __name__ == "__main__":
    main()
