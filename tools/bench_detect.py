"""Time detection, cadence and one second of oracle play per tune, native vs Python.

    python tools/bench_detect.py [TUNE.sid ...] [--threads N]

Without arguments the synthetic tunes of ``tests/irqhost_tunes.py`` are used.
``--threads N`` also runs detection of N tunes in N threads against one tune alone.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

import jennings.memory as jmem
from pysidtracker import cadence, oracle
from pysidtracker.detect import detect_playroutine
from pysidtracker.image import SidImage

PLAY_SECOND_FRAMES = 50
CALL_CYCLES = 100_000


def timed(fn):
    """Seconds ``fn()`` takes."""
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def measure(data: bytes) -> dict:
    """Seconds per stage for one tune on the current engine."""
    return {
        "detect": timed(
            lambda: detect_playroutine(
                SidImage.from_bytes(data), lambda _: None, init=True
            )
        ),
        "cadence": timed(lambda: cadence.playroutine_cadence(data)),
        "oracle": timed(
            lambda: oracle.register_grid(
                data, PLAY_SECOND_FRAMES, max_cycles=CALL_CYCLES
            )
        ),
    }


def tunes(paths):
    """``(name, bytes)`` of each given file, else of each synthetic test tune."""
    if paths:
        return [(p.name, p.read_bytes()) for p in map(Path, paths)]
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tests import irqhost_tunes  # pylint: disable=import-outside-toplevel

    return [(n, irqhost_tunes.build(n)) for n in irqhost_tunes.TUNES]


def bench_one(job) -> tuple:
    """``(name, native, stage seconds)`` for one ``(name, bytes, native)``, in this process."""
    name, data, native = job
    jmem.NATIVE = native and jmem.NATIVE
    return name, native, measure(data)


def run(corpus, jobs: int) -> tuple:
    """``(python, native)``: ``{name: stage seconds}`` over the corpus, one tune a process.

    Each timing is one tune on one core; tunes run side by side only to finish sooner.
    """
    work = [(n, d, native) for n, d in corpus for native in (False, True)]
    out = {False: {}, True: {}}
    with ProcessPoolExecutor(jobs) as pool:
        for name, native, stages in tqdm(
            pool.map(bench_one, work), total=len(work), unit="run"
        ):
            out[native][name] = stages
    return out[False], out[True]


def thread_test(data: bytes, count: int) -> tuple:
    """``(one tune alone, count tunes in count threads)`` detection wall seconds."""

    def detect():
        detect_playroutine(SidImage.from_bytes(data), lambda _: None, init=True)

    alone = timed(detect)

    def parallel():
        with ThreadPoolExecutor(count) as pool:
            for future in [pool.submit(detect) for _ in range(count)]:
                future.result()

    return alone, timed(parallel)


def main(argv=None) -> None:
    """Print the per-tune and total timings on both engines."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("tunes", nargs="*")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args(argv)
    corpus = tunes(args.tunes)
    python, native = run(corpus, args.jobs)
    print(f"{'tune':28}{'python s':>10}{'native s':>10}{'speedup':>9}")
    total = [0.0, 0.0]
    for name, _ in corpus:
        py, nat = sum(python[name].values()), sum(native[name].values())
        total[0], total[1] = total[0] + py, total[1] + nat
        print(f"{name:28}{py:10.3f}{nat:10.3f}{py / nat:9.1f}x")
    print(f"{'total':28}{total[0]:10.3f}{total[1]:10.3f}{total[0] / total[1]:9.1f}x")
    if args.threads:
        alone, many = thread_test(corpus[0][1], args.threads)
        print(f"detect alone {alone:.3f}s, {args.threads} threads {many:.3f}s")


if __name__ == "__main__":
    main()
