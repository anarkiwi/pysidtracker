"""Tests for the shared jennings host: NMOS illegal opcodes and the C64 read model."""

import pytest

from pysidtracker import observe_data_access, wire_mpu
from pysidtracker.emu import (
    live_ram,
    run_until,
    step_to_rts,
    sync_ram,
)
from pysidtracker.emu import SidReads
from pysidtracker.errors import EmulatorUnavailable

jennings = pytest.importorskip("jennings")

CODE = 0x1000
TARGET = 0x0080  # zero-page operand for the read-modify-write illegals


def _run(code, *, a=0, x=0, y=0, sp=0xFF, p=0x20, mem_at=None, illegals=True):
    """Assemble ``code`` at $1000, step it once, return ``(mpu, subject)``."""
    subject = bytearray(0x10000)
    subject[CODE : CODE + len(code)] = bytes(code)
    if mem_at:
        for addr, val in mem_at.items():
            subject[addr] = val
    mpu, _ = wire_mpu(subject, illegal_opcodes=illegals)
    mpu.a, mpu.x, mpu.y, mpu.sp, mpu.p, mpu.pc = a, x, y, sp, p, CODE
    mpu.step()
    sync_ram(mpu)
    return mpu, subject


# LDA #$FF / STA $D412,$D40E,$D40F / LDA #$20 / STA $D412 / LDA $D41B / RTS: the
# common SID-model probe, reading osc3 four cycles after releasing the test bit.
_MODEL_PROBE = bytes(
    [0xA9, 0xFF, 0x8D, 0x12, 0xD4, 0x8D, 0x0E, 0xD4, 0x8D, 0x0F, 0xD4]
    + [0xA9, 0x20, 0x8D, 0x12, 0xD4, 0xAD, 0x1B, 0xD4, 0x60]
)


def test_sid_reads_never_clock_backwards():
    chip = SidReads()
    chip.write(100, 0x12, 0x11)
    chip.flush(500)
    chip.write(200, 0x0E, 0x80)
    chip.read(300, 0x1B)
    assert chip.pending() == 0
    chip.flush(100)


def test_observe_data_access_sees_data_reads():
    subject = bytearray(0x10000)
    subject[CODE : CODE + 4] = bytes([0xAD, 0x42, 0x00])  # LDA $0042
    subject[0x42] = 0x77
    mpu, _mem = wire_mpu(subject)
    observe_data_access(mpu)  # no callbacks: nothing is wired
    superseded, current = [], []
    observe_data_access(mpu, on_read=superseded.append)
    observe_data_access(mpu, on_read=current.append)  # the last wiring stands
    mpu.pc = CODE
    mpu.step()
    assert superseded == []
    assert current == [0x42]
    assert mpu.a == 0x77


def test_wire_mpu_reports_missing_jennings(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _no_jennings(name, *args, **kwargs):
        if name.startswith("jennings"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_jennings)
    with pytest.raises(EmulatorUnavailable):
        wire_mpu(bytearray(0x10000))


def test_sid_reads_flush_queued_writes():
    chip = SidReads("8580")
    chip.write(10, 0x12, 0x81)
    assert chip.pending() == 1
    chip.flush(20)
    assert chip.pending() == 0
    assert chip.read(30, 0x19) == chip.read(30, 0x19)


def _routine(code):
    subject = bytearray(0x10000)
    subject[CODE : CODE + len(code)] = bytes(code)
    mpu, mem = wire_mpu(subject)
    return subject, mpu, mem


def test_step_to_rts_runs_the_routine_from_python():
    _, mpu, mem = _routine([0xE8, 0xE8, 0x60])
    step_to_rts(mpu, mem, CODE, 0, 1000)
    assert mpu.x == 2


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"stop_pcs": {CODE + 2}}, "pc"),
        ({"max_cycles": 3}, "cycles"),
        ({"sp_above": 0x80}, "sp"),
    ],
)
def test_run_until_reasons(kwargs, reason):
    _, mpu, _ = _routine([0xEA, 0xEA, 0xEA, 0xEA])
    mpu.pc, mpu.sp = CODE, 0xFF
    assert run_until(mpu, **kwargs) == reason
    expected = {"pc": CODE + 2, "cycles": CODE + 2, "sp": CODE}[reason]
    assert mpu.pc == expected


def _run_both(prog, steps):
    from vagrant_moses.cpu import flat_mpu

    out = {}
    for f in (flat_mpu, wire_mpu):
        s = bytearray(65536)
        s[0x1000 : 0x1000 + len(prog)] = prog
        mpu, _ = f(s)
        mpu.pc = 0x1000
        for _ in range(steps):
            mpu.step()
        out[f.__name__] = bytes(live_ram(mpu, s))
    return out


# lda #$ff / sta $d40f / lda #$81 / sta $d412 / lda #$37 / sta $01 / ldx #$ff
# / dex / bne -3 (spin) / lda $d41b / sta $2000 / lda $d43b / sta $2001 / jmp *
_VOICE3 = bytes.fromhex(
    "a9ff8d0fd4a9818d12d4"  # voice 3 setup (prefixed per case below)
)
_SPIN_READ = bytes.fromhex("a2ffcad0fd" "ad1bd48d0020" "ad3bd48d0120" "4c1e10")


def _voice3(port_first):
    prefix = bytes([0xA9, port_first, 0x85, 0x01]) if port_first else b""
    return prefix + _VOICE3 + bytes.fromhex("a9378501") + _SPIN_READ


def _read_osc3(prog):
    s = bytearray(65536)
    s[0x1000 : 0x1000 + len(prog)] = prog
    mpu, _ = wire_mpu(s)
    mpu.pc = 0x1000
    for _ in range(800):
        mpu.step()
    sync_ram(mpu)
    return s[0x2000], s[0x2001]
