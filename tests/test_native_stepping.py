"""The traced native paths give what stepping the Python engine gives."""

import random

import pytest

from pysidtracker.image import SidImage
from pysidtracker.playspec import tv
from pysidtracker.provenance import trace_frames
from pysidtracker.whydiff import cpu_history

from .irqhost_tunes import build

jmem = pytest.importorskip("jennings.memory")
TUNES = ("init_writes", "psid_cia", "raster_kernal", "osc3_reads")


@pytest.fixture
def engines(monkeypatch):
    """``run(fn)``: ``fn()``'s result on the native core and on the Python engine."""
    if not jmem.NATIVE:
        pytest.skip("jennings was installed without its native core")

    def run(fn):
        out = []
        for native in (True, False):
            monkeypatch.setattr(jmem, "NATIVE", native)
            out.append(fn())
        return out

    return run


@pytest.mark.parametrize("name", TUNES)
def test_cpu_history_matches(name, engines):
    native, python = engines(lambda: cpu_history(build(name), 1, max_cycles=20000))
    assert native == python and native


@pytest.mark.parametrize("name", TUNES)
def test_provenance_matches(name, engines):
    def trace():
        result = trace_frames(SidImage.from_bytes(build(name)), 2, max_cycles=20000)
        return result.frames, result.writes

    native, python = engines(trace)
    assert native == python and native[0][0]


def test_replay_routine_matches(engines):
    rng = random.Random(7)
    ops = (0x20, 0x60, 0xEA, 0xE6, 0x8D, 0xA9, 0x48, 0x68, 0xD0, 0x4C, 0xE8, 0x9D)
    for _ in range(40):
        image = bytearray(rng.getrandbits(8) for _ in range(0x10000))
        image[0x1000:0x1400] = bytes(rng.choice(ops) for _ in range(0x400))

        def run():
            memory = bytearray(image)
            cpu = tv._cpu(memory)  # pylint: disable=protected-access
            cpu.sp, cpu.pc = 0xFF, 0x1000
            tv._run_routine(cpu, memory)  # pylint: disable=protected-access
            return bytes(memory), cpu.pc, cpu.sp, cpu.processorCycles

        native, python = engines(run)
        assert native == python
