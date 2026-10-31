"""Serial, thread-pool and process-pool wall time of the batch workloads.

    python tools/bench_threads.py [--workers 8]

Shows which steps release the GIL (the native core, the native Machine) and which hold
it (pyresidfp, the per-write callbacks, Python glue).
"""

import argparse
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from pysidtracker import sidid
from pysidtracker.detect import detect_playroutine
from pysidtracker.image import SidImage
from pysidtracker.irqhost import InterruptHost

TASKS = 16


def tune() -> bytes:
    """A fixture PSID."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tests import irqhost_tunes  # pylint: disable=import-outside-toplevel

    return irqhost_tunes.build("psid_cia")


def detect(_):
    """Detection with init emulation."""
    detect_playroutine(SidImage.from_bytes(tune()), lambda _: None, init=True)


def interrupt_host(_):
    """Three emulated seconds on the native Machine."""
    InterruptHost(tune()).run(3_000_000)


def core(_):
    """A long native-core run with no callbacks."""
    from jennings.devices.mpu6502 import MPU  # pylint: disable=import-outside-toplevel
    from jennings.memory import NativeMemory  # pylint: disable=import-outside-toplevel

    mem = NativeMemory(bytearray(0x10000))
    mem.write(0x1000, bytes([0xE8, 0xD0, 0xFD, 0x4C, 0, 0x10]))
    mpu = MPU(memory=mem)
    mpu.pc = 0x1000
    mpu.run_until(max_cycles=100_000_000)


def chip(_):
    """pyresidfp clocking one voice."""
    from pyresidfp._pyresidfp import (  # pylint: disable=import-outside-toplevel
        SID,
        ChipModel,
        SamplingMethod,
    )

    sid = SID(ChipModel.MOS6581, SamplingMethod.DECIMATE, 985248.0, 8000.0)
    sid.write(0x18, 15)
    sid.write(0x04, 0x41)
    for _ in range(200):
        sid.clock(5000)


def scan(_):
    """SIDId signature matching."""
    for _ in range(20):
        sidid.identify(tune())


def timed(pool_cls, workers, fn):
    """Wall seconds for TASKS calls of ``fn`` on a pool."""
    start = time.perf_counter()
    with pool_cls(workers) as pool:
        list(pool.map(fn, range(TASKS)))
    return time.perf_counter() - start


def main(argv=None) -> None:
    """Print the speedup of each workload on threads and on processes."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    print(f"{'workload':16}{'serial s':>10}{'threads x':>11}{'processes x':>13}")
    for fn in (detect, interrupt_host, core, chip, scan):
        serial = timed(ThreadPoolExecutor, 1, fn)
        threads = timed(ThreadPoolExecutor, args.workers, fn)
        procs = timed(ProcessPoolExecutor, args.workers, fn)
        print(
            f"{fn.__name__:16}{serial:10.2f}{serial / threads:11.1f}{serial / procs:13.1f}"
        )
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    print(f"largest worker process: {peak:.0f} MiB resident")


if __name__ == "__main__":
    main()
