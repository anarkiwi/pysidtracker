"""Tests for the MemPlayer scaffold and the reglog player helper."""

from pysidtracker import MemPlayer, register_writes_from_player
from pysidtracker.registers import PW_HI_REGS, SID_BASE, SID_REG_COUNT
from pysidtracker.testing import whole_song


class _CountPlayer(MemPlayer):
    """A tiny player: init seeds two registers; each frame bumps register 0."""

    def _init(self, subtune):
        self._wr(SID_BASE + 0, 0x10 + subtune)
        self._wr(SID_BASE + 0x18, 0x0F)

    def _frame(self):
        value = self._rd(SID_BASE)
        self._wr(SID_BASE, (value + 1) & 0xFF)


def test_post_init_snapshot():
    player = _CountPlayer(b"", 0x1000, subtune=1)
    assert player.regs[0] == 0x11
    assert player.regs[0x18] == 0x0F
    assert len(player.regs) == SID_REG_COUNT


def test_first_frame_emits_all_then_diffs():
    player = _CountPlayer(b"", 0x1000)
    first = player.play_frame()
    assert len(first) == SID_REG_COUNT
    assert dict(first)[0] == 0x11  # 0x10 seeded, bumped once
    second = player.play_frame()
    assert second == [(0, 0x12)]


def test_render_grid():
    player = _CountPlayer(b"", 0x1000)
    grid = player.render_grid(whole_song(3))
    assert [row[0] for row in grid] == [0x11, 0x12, 0x13]
    assert all(len(row) == SID_REG_COUNT for row in grid)


def test_iter_frames_bounded():
    player = _CountPlayer(b"", 0x1000)
    frames = list(player.iter_frames(max_frames=whole_song(2)))
    assert len(frames) == 2


def test_register_writes_from_player():
    player = _CountPlayer(b"", 0x1000)
    writes = list(
        register_writes_from_player(player, max_frames=2, cycles_per_frame=19656)
    )
    # Baseline: all registers at clock 0.
    baseline = [w for w in writes if w.clock < 19656]
    assert len(baseline) == SID_REG_COUNT
    assert baseline[0].clock == 0
    assert dict((w.reg, w.val) for w in baseline)[0] == 0x10
    # Frames start at start_frame=1.
    frame_clocks = {w.clock // 19656 for w in writes if w.clock >= 19656}
    assert frame_clocks == {1, 2}


class _PwPlayer(_CountPlayer):
    """Writes a full byte to every pulse-width-high register at init."""

    def _init(self, subtune):
        for register in PW_HI_REGS:
            self._wr(SID_BASE + register, 0xF0 | register)


class _MaskedPwPlayer(_PwPlayer):
    MASK_PW_HI = True


def test_snapshot_masks_pw_hi_only_when_asked():
    raw, masked = _PwPlayer(b"", 0x1000).regs, _MaskedPwPlayer(b"", 0x1000).regs
    assert [raw[r] for r in PW_HI_REGS] == [0xF3, 0xFA, 0xF1]
    assert [masked[r] for r in PW_HI_REGS] == [0x03, 0x0A, 0x01]
    others = [r for r in range(SID_REG_COUNT) if r not in PW_HI_REGS]
    assert [raw[r] for r in others] == [masked[r] for r in others]
