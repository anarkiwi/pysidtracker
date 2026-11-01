"""Tests for the MemPlayer scaffold and the reglog player helper."""

from pysidtracker import MemPlayer
from pysidtracker.registers import PW_HI_REGS, SID_BASE
from pysidtracker.testing import whole_song


class _CountPlayer(MemPlayer):
    """A tiny player: init seeds two registers; each frame bumps register 0."""

    def _init(self, subtune):
        self._wr(SID_BASE + 0, 0x10 + subtune)
        self._wr(SID_BASE + 0x18, 0x0F)

    def _frame(self):
        value = self._rd(SID_BASE)
        self._wr(SID_BASE, (value + 1) & 0xFF)


def test_iter_frames_bounded():
    player = _CountPlayer(b"", 0x1000)
    frames = list(player.iter_frames(max_frames=whole_song(2)))
    assert len(frames) == 2


class _PwPlayer(_CountPlayer):
    """Writes a full byte to every pulse-width-high register at init."""

    def _init(self, subtune):
        for register in PW_HI_REGS:
            self._wr(SID_BASE + register, 0xF0 | register)


class _MaskedPwPlayer(_PwPlayer):
    MASK_PW_HI = True
