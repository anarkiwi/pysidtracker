"""The shared libsidplayfp power-on RAM image that the players and emulator boot from."""

from pysidtracker import MemPlayer, poweron


def test_banks_follow_the_driver_iomap():
    """The init and play bank maps are libsidplayfp's ``psiddrv::iomap``."""
    # address windows: below BASIC, BASIC ($A000-$CFFF), I/O page, above $DFFF
    addresses = (0x9FF0, 0xC000, 0xD000, 0xE000)
    expected = (0x37, 0x36, 0x34, 0x35)
    assert tuple(poweron.psid_init_bank(a) for a in addresses) == expected
    assert tuple(poweron.psid_play_bank(a) for a in addresses) == expected


class _ReadPlayer(MemPlayer):
    """A do-nothing player used only to inspect the boot memory image."""

    def _init(self, subtune):
        pass

    def _frame(self):
        pass
