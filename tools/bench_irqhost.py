"""Emulated seconds per wall second of InterruptHost on a fixture RSID and PSID.

    python tools/bench_irqhost.py [--seconds 30]

Python backend against native; the shares of write handling and chip time come from
profiling the native run with cProfile.
"""

import argparse
import cProfile
import pstats
import sys
import time
from pathlib import Path

import vagrant_moses.machine as vmachine
from pysidtracker.irqhost import InterruptHost

TUNES = {"RSID": "raster_kernal", "PSID": "psid_cia"}


def build(name: str) -> bytes:
    """The fixture tune ``name`` as a PSID/RSID container."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tests import irqhost_tunes  # pylint: disable=import-outside-toplevel

    return irqhost_tunes.build(name)


def run(data: bytes, seconds: int, backend: str) -> float:
    """Wall seconds to emulate ``seconds`` of the tune."""
    host = InterruptHost(data, backend=backend)
    start = time.perf_counter()
    host.run(seconds * host.clock_hz)
    return time.perf_counter() - start


def shares(data: bytes, seconds: int) -> dict:
    """Fraction of a native run's time in each callee group."""
    host = InterruptHost(data, backend="native")
    profile = cProfile.Profile()
    profile.enable()
    host.run(seconds * host.clock_hz)
    profile.disable()
    stats = pstats.Stats(profile).stats
    total = sum(v[2] for v in stats.values())
    groups = {"_Sid.write": 0.0, "SidReads/pyresidfp": 0.0, "core run": 0.0}
    for (path, _, name), value in stats.items():
        if name == "write" and "irqhost" in path:
            groups["_Sid.write"] += value[2]
        elif "pyresidfp" in path or "emu.py" in path:
            groups["SidReads/pyresidfp"] += value[2]
        elif name == "run":
            groups["core run"] += value[2]
    return {k: v / total for k, v in groups.items()}


def main(argv=None) -> None:
    """Print x-realtime on both backends and the native time shares."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--seconds", type=int, default=30)
    args = parser.parse_args(argv)
    if not vmachine.NATIVE:
        parser.error("vagrant-moses was installed without its native core")
    for kind, name in TUNES.items():
        data = build(name)
        py, nat = (run(data, args.seconds, b) for b in ("python", "native"))
        print(
            f"{kind} {name}: python {args.seconds / py:.1f}x realtime, "
            f"native {args.seconds / nat:.1f}x realtime ({py / nat:.1f}x)"
        )
        print(
            "  native time shares:",
            {k: round(v, 3) for k, v in shares(data, args.seconds).items()},
        )


if __name__ == "__main__":
    main()
