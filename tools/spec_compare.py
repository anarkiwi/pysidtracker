"""Play tunes with two playspecs (and optionally the tune's own 6502) call by call.

``python tools/spec_compare.py OLD.play [NEW.play] -- TUNES.csv|TUNE.sid ... [--player NAME]
[--frames N] [--python] [--oracle] [--bench] [--jobs J]``

Each tune is played by its format's binder over each spec (``NEW.play`` defaults to the
format's own). Per call the SID writes, registers and memory outside the stack page must be
equal; with ``--oracle`` the memory of the tune's own 6502 is compared too. ``--bench`` prints
microseconds per call after warm-up, native and Python, and the time to render the whole tune.
A CSV needs a ``path`` column; ``--player`` keeps the rows of that ``player`` column. Exit
status is the number of tunes that differ.
"""

import argparse
import csv
import dataclasses
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

STACK = range(0x100, 0x200)


def players(path: Path, specs, native: bool, oracle: bool):
    """``(players, play)``: one SpecPlayer per spec file (the format's own for ``""``) and,
    with ``oracle``, the tune's 6502, for the tune at ``path``."""
    from pysidtracker import find_format  # pylint: disable=import-outside-toplevel
    from pysidtracker.oracle import EmuPlayer  # pylint: disable=import-outside-toplevel
    from pysidtracker.playspec import load  # pylint: disable=import-outside-toplevel

    fmt = find_format(path.read_bytes())
    model = fmt.parser.parse(path.read_bytes())
    out = []
    for name in specs:
        use = fmt
        if name:
            alt = load(name)
            use = dataclasses.replace(
                fmt,
                spec=lambda m, alt=alt: dataclasses.replace(fmt.spec(m), spec=alt),
            )
        out.append(use.spec_player(model, native=native))
    if oracle:
        binding = fmt.spec(model)
        _n, _a, _r, base, init, play = fmt.model_metadata(model)
        out.append(EmuPlayer(binding.image, base, init, play, binding.subtune))
    return out


def differs(base, other, memory: bool) -> str:
    """What differs after a call between ``base`` and ``other``, or the empty string."""
    if hasattr(other, "writes") and base.writes != other.writes:
        return f"writes {base.writes[:6]} vs {other.writes[:6]}"
    if base.regs != other.regs:
        return "registers"
    if memory:
        mine = base.memory
        theirs = (
            other.memory if hasattr(other, "writes") else bytes(other._mem)
        )  # pylint: disable=protected-access
        for addr in range(0x10000):
            if mine[addr] != theirs[addr] and addr not in STACK:
                return f"memory ${addr:04x} {mine[addr]:02x} vs {theirs[addr]:02x}"
    return ""


def compare(job):
    """``(name, calls, microseconds per call by player, difference or None)`` of a tune."""
    path, specs, frames, native, oracle = job
    try:
        from pysidtracker.testing import (
            song_frames,
        )  # pylint: disable=import-outside-toplevel

        count = frames or min(song_frames(path.read_bytes(), 1), 6000)
        made = players(path, specs, native, oracle)
        spent = [0.0] * len(made)
        for call in range(count):
            for k, player in enumerate(made):
                start = time.perf_counter()
                player.play_frame()
                spent[k] += time.perf_counter() - start
            for k, player in enumerate(made[1:], 1):
                what = differs(made[0], player, True)
                if what:
                    return path.name, call, [], f"call {call}: player {k} {what}"
        return path.name, count, [1e6 * s / count for s in spent], None
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return path.name, 0, [], f"refused: {type(exc).__name__}: {str(exc)[:110]}"


def bench(job):
    """``(name, microseconds per call and render seconds by spec)`` for ``--bench``."""
    path, specs, frames, _native, _oracle = job
    from pysidtracker.testing import (
        song_frames,
    )  # pylint: disable=import-outside-toplevel

    whole = song_frames(path.read_bytes(), 1)
    rows = []
    for k in range(len(specs)):
        cells = []
        for native in (True, False):
            player = players(path, specs, native, False)[k]
            for _ in range(10):
                player.play_frame()
            start = time.perf_counter()
            for _ in range(frames):
                player.play_frame()
            cells.append(1e6 * (time.perf_counter() - start) / frames)
        player = players(path, specs, True, False)[k]
        start = time.perf_counter()
        for _ in range(whole):
            player.play_frame()
        rows.append((*cells, time.perf_counter() - start))
    return path.name, whole, rows


def tunes(items, player: str):
    """The tune paths named by ``items``."""
    for item in items:
        if item.suffix == ".csv":
            with open(item, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if not player or row.get("player") == player:
                        yield Path(row["path"])
        else:
            yield item


def main() -> int:
    """Parse the command line and run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", action="append", default=[], help="a .play file (repeat)"
    )
    parser.add_argument("tunes", nargs="+", type=Path)
    parser.add_argument("--player", default="")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--python", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    specs = args.spec or [""]
    if len(specs) == 1 and args.spec:
        specs.append("")
    jobs = [
        (t, specs, args.frames or 2000 * args.bench, not args.python, args.oracle)
        for t in tunes(args.tunes, args.player)
    ]
    bad = 0
    with ProcessPoolExecutor(args.jobs) as pool:
        for row in tqdm(
            pool.map(bench if args.bench else compare, jobs), total=len(jobs)
        ):
            if args.bench:
                name, whole, cells = row
                for spec, (nat, py, secs) in zip(specs, cells):
                    print(
                        f"{name[:30]:30} {Path(spec).name or 'format':16} native {nat:7.1f}us "
                        f"python {py:8.1f}us whole tune {whole} calls {secs:.2f}s"
                    )
                continue
            name, calls, micro, err = row
            bad += err is not None and not err.startswith("refused")
            print(
                f"{name[:40]:40} {calls:6} {' '.join(f'{u:.0f}us' for u in micro):24} {err or 'same'}"
            )
    return bad


if __name__ == "__main__":
    sys.exit(main())
