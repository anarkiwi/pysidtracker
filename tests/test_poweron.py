"""The shared libsidplayfp power-on RAM image that the players and emulator boot from."""

import numpy as np

from vagrant_moses import c64
from vagrant_moses import memmap as mm

from pysidtracker import MemPlayer, poweron


def test_ram_pattern_is_four_byte_runs_flipped_per_16k_bank():
    ram = poweron.ram_pattern()
    assert len(ram) == 0x10000
    assert ram[:16] == bytes.fromhex("0000ffffffff00000000ffffffff0000")
    assert ram[0x4000:0x4008] == bytes.fromhex("ffff00000000ffff")
    assert ram[0x8000:0x8010] == ram[:16]
    assert ram[0xC000:0xC008] == ram[0x4000:0x4008]


def test_power_on_image_is_the_driver_cleared_pattern_with_the_sid_at_rest():
    pattern = poweron.power_on_ram()
    base = poweron.ram_pattern()
    # psiddrv::install zeroes $0000-$03FE (0x3FF bytes).
    assert pattern[: poweron.PSID_DRIVER_CLEARED] == bytes(poweron.PSID_DRIVER_CLEARED)
    assert pattern[0x03FE] == 0
    assert pattern[0x03FF] == base[0x03FF]  # the clear stops before $03FF
    # the uninitialised SID register file reads as 0 (reSID reset)
    assert not any(pattern[mm.SID_BASE : mm.SID_LAST + 1])
    # ... everything else is the power-on pattern, untouched
    assert pattern[0x3FF : mm.SID_BASE] == base[0x3FF : mm.SID_BASE]
    assert pattern[mm.SID_LAST + 1 :] == base[mm.SID_LAST + 1 :]


def test_power_on_image_matches_the_reference_formula():
    addr = np.arange(0x10000)
    run = (addr & 7 >= 2) & (addr & 7 < 6)
    want = np.where(run ^ ((addr >> 14) & 1).astype(bool), 0xFF, 0).astype(np.uint8)
    want[:0x3FF] = 0
    want[0xD400:0xD800] = 0  # the SID register file at rest
    assert bytes(poweron.power_on_ram()) == want.tobytes()


def test_ram_pattern_is_vagrant_moses_power_on_ram():
    assert poweron.ram_pattern() == c64.poweron_ram()


def test_banks_follow_the_driver_iomap():
    """The init and play bank maps are libsidplayfp's ``psiddrv::iomap``."""
    # address windows: below BASIC, BASIC ($A000-$CFFF), I/O page, above $DFFF
    addresses = (0x9FF0, 0xC000, 0xD000, 0xE000)
    expected = (0x37, 0x36, 0x34, 0x35)
    assert tuple(poweron.psid_init_bank(a) for a in addresses) == expected
    assert tuple(poweron.psid_play_bank(a) for a in addresses) == expected


def test_play_bank_switches_the_windows_a_play_reads():
    """A play in BASIC reads the RAM behind the window, not the ROM (into_orbit)."""
    address = 0xC000  # Into Orbit's play address, init at $9FF0
    init_bank = poweron.psid_init_bank(0x9FF0)
    play_bank = poweron.psid_play_bank(address)
    assert init_bank == 0x37  # ROMs in for init
    assert play_bank == 0x36  # RAM behind BASIC, KERNAL and I/O for play


class _ReadPlayer(MemPlayer):
    """A do-nothing player used only to inspect the boot memory image."""

    def _init(self, subtune):
        pass

    def _frame(self):
        pass


def test_memplayer_boots_from_the_poweron_image():
    player = _ReadPlayer(b"", 0x8000)
    want = poweron.power_on_ram()
    want[mm.SID_BASE + 0x18] = 0x0F  # the PSID driver seeds maximum volume
    assert bytes(player._mem) == want


def test_memplayer_reads_poweron_ram_not_zeros():
    player = _ReadPlayer(b"", 0x8000)
    # Uninitialised RAM a tune reads is the C64 power-on pattern, not zeros.
    assert player._rd(0x0400) == 0x00
    assert player._rd(0x0402) == 0xFF
    assert player._rd(0x8402) == 0xFF  # bank $8000-$BFFF: offset 2-5 run is $FF
    # ... the PSID driver cleared $0000-$03FE:
    assert player._rd(0x0200) == 0x00
    # ... and the SID maximum-volume seed survived:
    assert player._rd(mm.SID_BASE + 0x18) == 0x0F
