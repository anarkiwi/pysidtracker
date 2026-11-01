"""Microseconds per frame of the Python and the native playspec executor on tunes.

``python tools/spec_bench.py TUNE.sid ... [--frames N]``: for each tune, the time per frame
of ``render_grid`` with ``SpecPlayer(native=False)`` and ``SpecPlayer(native=True)`` (player
construction and the first frames excluded), and the native player's bail count.
"""

import argparse
import time
from pathlib import Path

from pysidtracker import find_format

WARM = 200


def per_frame(player, frames: int) -> float:
    """Microseconds per frame of ``player.render_grid(frames)`` after a warm-up."""
    player.render_grid(WARM)
    start = time.perf_counter()
    player.render_grid(frames)
    return (time.perf_counter() - start) / frames * 1e6


def main() -> None:
    """Print one line per tune."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tunes", nargs="+")
    parser.add_argument("--frames", type=int, default=20000)
    args = parser.parse_args()
    for name in args.tunes:
        raw = Path(name).read_bytes()
        fmt = find_format(raw)
        model = fmt.parser.parse(raw)
        python, native = (
            fmt.spec_player(model, notes=True, native=n) for n in (False, True)
        )
        py_us = per_frame(python, args.frames)
        nat_us = per_frame(native, args.frames)
        print(
            f"{fmt.name:15s} {Path(name).name:36s} python {py_us:6.2f} native {nat_us:6.2f}"
            f" us/frame ({py_us / nat_us:4.1f}x) bails {native._bails}"  # pylint: disable=protected-access
        )


if __name__ == "__main__":
    main()
