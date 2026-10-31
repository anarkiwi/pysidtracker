"""Emulated seconds per wall second of InterruptHost, and where the time goes.

    python tools/bench_irqhost.py [--seconds 10] [--tunes a,b] [--backend native] [--repeat 3]
                                  [--baseline before.json] [--json out.json]

Wall time comes from an unprofiled run. Python time comes from a second, cProfile'd
run: its total less the time inside the machine's own ``run`` (the native cycle loop),
so it holds the merge of the logs into ``raw``, the callbacks into Python, pyresidfp and the glue. Native is wall less
Python. SID writes and ``$19-$1C`` reads are counted at the chip; reads of the other
registers are answered inside the native bus model and are not visible from Python.
"""

import argparse
import cProfile
import json
import pstats
import sys
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from pysidtracker import emu, irqhost
from pysidtracker.irqhost import InterruptHost

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# pylint: disable=wrong-import-position
from tests import irqhost_tunes

DEFAULT_TUNES = (
    "raster_kernal,psid_cia,psid_vbi,second_sid,osc3_reads,nmi_digi_oneshot,"
    "latency_stress,badline_loop,sprite_dma,banking"
)


def wall(data: bytes, seconds: int, backend) -> float:
    """Wall seconds to emulate ``seconds`` of the tune."""
    host = InterruptHost(data, backend=backend)
    start = time.perf_counter()
    host.run(seconds * host.clock_hz)
    assert host.raw is not None
    return time.perf_counter() - start


def profiled(data: bytes, seconds: int, backend) -> dict:
    """Python seconds, SID writes and SID reads by register class of a profiled run."""
    writes, reads = Counter(), Counter()
    write, read = emu.SidReads.write, emu.SidReads.read

    def counted_write(self, cycle, register, value):
        writes["fed"] += 1
        write(self, cycle, register, value)

    def counted_read(self, cycle, register):
        reads["$19-$1C"] += 1
        return read(self, cycle, register)

    emu.SidReads.write, emu.SidReads.read = counted_write, counted_read
    try:
        host = InterruptHost(data, backend=backend)
        profile = cProfile.Profile()
        profile.enable()
        host.run(seconds * host.clock_hz)
        assert host.raw is not None
        profile.disable()
        pending = sum(len(chip.drain_writes()) for chip in host._chips)
    finally:
        emu.SidReads.write, emu.SidReads.read = write, read
    stats = pstats.Stats(profile)
    core = sum(v[2] for k, v in stats.stats.items() if "'run'" in k[2])
    return {
        "pytime": stats.total_tt - core,
        "writes": writes["fed"] + pending,
        "reads_19_1c": reads["$19-$1C"],
    }


def measure(data: bytes, seconds: int, backend, repeat: int) -> dict:
    """One tune's best-of-``repeat`` wall, x-realtime, Python/native split and SID counts."""
    out = {"wall": min(wall(data, seconds, backend) for _ in range(repeat))}
    out.update(profiled(data, seconds, backend))
    out["xrt"] = seconds / out["wall"]
    out["native"] = max(0.0, out["wall"] - out["pytime"])
    return out


def main(argv=None) -> None:
    """Print the table for each tune, with the baseline's columns when given."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--tunes", default=DEFAULT_TUNES)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--backend", choices=["native", "python"], default="native")
    parser.add_argument("--baseline", help="JSON of an earlier --json run to compare")
    parser.add_argument("--json", help="write the results here")
    args = parser.parse_args(argv)
    irqhost.poweron_writes = lambda: None
    before = json.loads(Path(args.baseline).read_text()) if args.baseline else {}
    names = args.tunes.split(",")
    results = {
        name: measure(
            irqhost_tunes.build(name), args.seconds, args.backend, args.repeat
        )
        for name in tqdm(names, unit="tune")
    }
    head = f"{'tune':18}{'wall s':>8}{'xRT':>8}{'py s':>8}{'native s':>9}{'writes':>10}{'rd19-1C':>9}"
    print(head + (f"{'before xRT':>12}{'gain':>7}" if before else ""))
    for name, r in results.items():
        line = (
            f"{name:18}{r['wall']:8.3f}{r['xrt']:8.1f}{r['pytime']:8.3f}"
            f"{r['native']:9.3f}{r['writes']:10d}{r['reads_19_1c']:9d}"
        )
        if name in before:
            line += f"{before[name]['xrt']:12.1f}{r['xrt'] / before[name]['xrt']:6.1f}x"
        print(line)
    if args.json:
        Path(args.json).write_text(json.dumps(results))


if __name__ == "__main__":
    main()
