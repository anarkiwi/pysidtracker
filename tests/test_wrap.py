"""Native-tick play wrappers over synthetic 6502 images."""

import pytest

from pysidtracker.asm6502 import assemble
from pysidtracker.emu import SidWriteLog, call_play
from pysidtracker.errors import SidError
from pysidtracker.image import SidImage
from pysidtracker.registers import PW_HI_REGS, SID_REG_COUNT
from pysidtracker.wrap import (
    Wrap,
    Wrapped,
    boot,
    count_entries,
    idle_init,
    measure_wrap,
)

from .helpers import build_psid

ORG = 0x1000

FLUSH = """
       LDX #$18
flush: LDA ghost,X
       STA $D400,X
       DEX
       BPL flush
"""

GHOST_PLAYER = """
entry: LDX #$18
bump:  LDA ghost,X
       CLC
       ADC #$01
       STA ghost,X
       DEX
       BPL bump
       RTS
ghost: .byte """ + ",".join(["$00"] * SID_REG_COUNT)

SID_PLAYER = """
entry: INC count
       LDA count
       STA $D401
       RTS
count: .byte $00
"""

TUNES = {
    "jsr_then_flush": "init: RTS\nplay: JSR entry\n" + FLUSH + "RTS\n" + GHOST_PLAYER,
    "flush_then_jmp": "init: RTS\nplay:" + FLUSH + "JMP entry\n" + GHOST_PLAYER,
    "twice": "init: RTS\nplay: JSR entry\nJSR entry\nRTS\n" + SID_PLAYER,
    "spin": "init: CLI\nidle: JMP idle\nplay: RTS\n",
    "hang": "init: RTS\nplay: JMP play\n",
    "busy": "init: LDX #$00\nloop:" + "NOP\n" * 10 + "DEX\nBNE loop\nRTS\nplay: RTS\n",
}


def tune(name, play=True):
    """``(image, labels)`` of a synthetic tune."""
    code, labels = assemble(TUNES[name], ORG)
    sid = build_psid(code, ORG, labels["init"], labels["play"] if play else 0)
    return SidImage.from_bytes(sid), labels


def regs(frame):
    """A distinct register file for ``frame``."""
    return [(frame * 7 + r) & 0xFF for r in range(SID_REG_COUNT)]


def masked(values):
    """``values`` as :class:`Wrapped` snapshots them."""
    return [v & 0x0F if r in PW_HI_REGS else v for r, v in enumerate(values)]


def test_entry_called_twice_writes_the_sid():
    image, lab = tune("twice")
    entries = {lab["entry"]}
    wrap = measure_wrap(image, 0, lab["play"], entries)
    assert wrap == Wrap(lab["play"], frozenset(entries), 0, None)
    assert count_entries(image, 0, lab["play"], entries) == 2
    assert count_entries(image, 0, lab["play"], {lab["count"]}) == 1
    calls = []

    def tick():
        calls.append(None)
        return {0xD400: regs(len(calls))}

    player = Wrapped(tick, image, wrap)
    player.play_frame()
    assert len(calls) == 2 and player.regs == masked(regs(2))


@pytest.mark.parametrize(
    "play,entries",
    [(None, {0x1000}), (0x1001, set()), (0x1001, {0x1001, 0x1005})],
)
def test_no_wrap(play, entries):
    image, _ = tune("twice")
    assert measure_wrap(image, 0, play, entries) is None


def test_sink_of_writes_outside_any_entry():
    image, lab = tune("twice")
    assert measure_wrap(image, 0, lab["play"], {lab["count"]}).sink == lab["count"] - 1
    image, lab = tune("jsr_then_flush")
    assert measure_wrap(image, 0, lab["init"], {lab["entry"]}).sink is None


def test_boot_idles_a_play_less_init():
    image, lab = tune("spin", play=False)
    _, mpu, _, play = boot(image)
    assert play == lab["init"] and mpu.pc in (lab["idle"], lab["idle"] + 3)
    assert mpu.processorCycles < 100_000


@pytest.mark.parametrize("name", ["twice", "busy"])
def test_idle_init_runs_a_returning_init_to_its_rts(name):
    image, lab = tune(name)
    _, mpu, mem, _ = boot(image)
    depth = mpu.sp
    idle_init(mpu, mem, lab["init"], 0, 100_000)
    assert mpu.sp == depth and mpu.x == 0


def test_write_log_and_call_play():
    image, lab = tune("twice")
    _, mpu, mem, _ = boot(image)
    log = SidWriteLog(mpu)
    assert not log.calls
    writes = log.call(mem, lab["play"])
    pcs = [pc for _, _, pc, _ in writes]
    assert writes == [(1, lab["count"], pcs[0], 1), (1, lab["count"], pcs[1], 2)]
    assert pcs[0] == pcs[1] == lab["entry"] + 6
    assert log.call(mem, lab["play"])[-1][3] == 4 and len(log.calls) == 2
    mirrored, exact = SidWriteLog(mpu), SidWriteLog(mpu, mirrors=False)
    assert mirrored.registers[0xD421] == 1 and 0xD421 not in exact.registers
    assert 0xD400 + SID_REG_COUNT not in mirrored.registers


def test_call_play_raises_when_play_hangs():
    image, lab = tune("hang")
    _, mpu, mem, _ = boot(image)
    with pytest.raises(SidError, match="did not return"):
        call_play(mpu, mem, lab["play"], 2000)

    class Miss(Exception):
        """A caller's own error."""

    with pytest.raises(Miss):
        call_play(mpu, mem, lab["play"], 2000, Miss)
