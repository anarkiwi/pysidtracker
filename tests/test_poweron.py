"""The shared libsidplayfp power-on RAM image that the players and emulator boot from."""

import numpy as np

from pysidtracker import MemPlayer, poweron


def test_banks_follow_the_driver_iomap():
    """Every address banks as ``psiddrv::iomap``'s $A000/$D000/$E000 windows."""
    addresses = np.arange(0x10000)
    windows = np.searchsorted([0xA000, 0xD000, 0xE000], addresses, side="right")
    expected = np.array([0x37, 0x36, 0x34, 0x35])[windows].tolist()
    assert [poweron.psid_init_bank(a) for a in range(0x10000)] == expected
    assert [poweron.psid_play_bank(a) for a in range(0x10000)] == expected


class _ReadPlayer(MemPlayer):
    """A do-nothing player used only to inspect the boot memory image."""

    def _init(self, subtune):
        pass

    def _frame(self):
        pass
