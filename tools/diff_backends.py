"""Run every emulation entry point on jennings' native core and on its Python engine.

    python tools/diff_backends.py [TUNE.sid ...]

Without arguments the synthetic tunes of ``tests/irqhost_tunes.py`` are used.
Exit status is the number of differing (tune, result) pairs.
"""

import argparse
import contextlib
import dataclasses
import re
import sys
from pathlib import Path

from tqdm import tqdm

import jennings.memory as jmem
from pysidtracker import cadence, oracle, paths, trace, variants, wrap
from pysidtracker.detect import init_effects, run_init
from pysidtracker.image import SidImage

GRID_FRAMES = 16
MACHINE_CYCLES = 3_000_000
CALL_CYCLES = 100_000


@contextlib.contextmanager
def engine(native: bool):
    """Build every MPU inside the block on the native core or the Python engine."""
    saved = jmem.NATIVE
    jmem.NATIVE = native and saved
    try:
        yield
    finally:
        jmem.NATIVE = saved


def machine(image: SidImage, native: bool, stepped: bool):
    """The cadence machine's observable state after a run."""
    header = image.header
    ntsc = cadence._resolve_standard(image, None)  # pylint: disable=protected-access
    m = cadence._Machine(  # pylint: disable=protected-access
        image, ntsc, MACHINE_CYCLES, native
    )
    m.driver(header, 0, ntsc)
    end = (m.execute_stepped if stepped else m.execute)(8)
    return {
        "end": end,
        "cycles": m.mpu.processorCycles,
        "pc": m.mpu.pc,
        "events": [list(e) for e in m.events],
        "handlers": [h and list(h) for h in m.handlers],
        "frames": list(m.frames),
        "opaque": sorted(m.opaque),
        "ram": bytes(m.ram),
    }


def attempt(fn):
    """``fn()``, or the exception it raised, so both engines are compared on failures too."""
    try:
        return fn()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return type(exc).__name__, str(exc)


def booted(image: SidImage):
    """The state :func:`pysidtracker.wrap.boot` leaves: RAM, registers and cycles."""
    ram, mpu, _, play = wrap.boot(image, 0, CALL_CYCLES)
    return bytes(ram), mpu.pc, mpu.sp, mpu.a, mpu.processorCycles, play


def results(data: bytes, native: bool) -> dict:
    """Every result compared, for one engine."""
    image = SidImage.from_bytes(data)
    out = {}
    with engine(native):
        out["cadence"] = dataclasses.astuple(cadence.playroutine_cadence(data))
        out["machine"] = machine(image, native, stepped=not native)
        out["grid"] = oracle.register_grid(data, GRID_FRAMES, max_cycles=CALL_CYCLES)
        out["trace"] = trace.trace_init(image, play_calls=4, max_cycles=CALL_CYCLES)
        effects = init_effects(image, 0, max_cycles=CALL_CYCLES)
        out["effects"] = (effects.writes, effects.reads, effects.jumps)
        ran = SidImage.from_bytes(data)
        run_init(ran, max_cycles=CALL_CYCLES)
        out["init_mem"] = bytes(ran.mem)
        out["offsets"] = attempt(
            lambda: variants.executed_offsets(image, nframes=4, max_cycles=CALL_CYCLES)
        )
        out["paths"] = attempt(
            lambda: re.sub(
                r" at 0x[0-9a-f]+",
                "",
                repr(paths.cpu_paths(data, 4, max_cycles=CALL_CYCLES)),
            )
        )
        out["boot"] = attempt(lambda: booted(SidImage.from_bytes(data)))
    return out


def tunes(paths):
    """``(name, bytes)`` of each given file, else of each synthetic test tune."""
    if paths:
        yield from ((p.name, p.read_bytes()) for p in map(Path, paths))
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tests import irqhost_tunes  # pylint: disable=import-outside-toplevel

    yield from ((n, irqhost_tunes.build(n)) for n in irqhost_tunes.TUNES)


def main(argv=None) -> int:
    """Diff the engines over the tunes; print each difference; return their count."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("tunes", nargs="*")
    args = parser.parse_args(argv)
    if not jmem.NATIVE:
        parser.error("jennings was installed without its native core")
    corpus = list(tunes(args.tunes))
    differing = 0
    for name, data in tqdm(corpus, unit="tune"):
        native, python = results(data, True), results(data, False)
        for key in native:
            if native[key] != python[key]:
                differing += 1
                tqdm.write(f"{name}: {key} differs")
    print(f"{len(corpus)} tunes, {differing} differences")
    return differing


if __name__ == "__main__":
    sys.exit(main())
