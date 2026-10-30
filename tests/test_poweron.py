"""The shared libsidplayfp power-on RAM image that the players and emulator boot from."""

import numpy as np

from vagrant_moses import c64
from vagrant_moses import memmap as mm

from pysidtracker import MemPlayer, poweron
from pysidtracker.testing import whole_song


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
    poweron.place_tune(want, 0x8000, b"")
    assert bytes(player._mem) == want


def test_place_tune_leaves_the_basic_load_pointers():
    ram = poweron.power_on_ram()
    poweron.place_tune(ram, 0x1000, b"\x01\x02\x03")
    assert ram[0x1000:0x1003] == b"\x01\x02\x03"
    assert [ram[a] | ram[a + 1] << 8 for a in (0x2D, 0x2F, 0x31, 0xAE)] == [0x1003] * 4
    assert ram[0xAC] | ram[0xAD] << 8 == 0x1000


def test_place_tune_image_covers_the_pointers():
    ram = poweron.power_on_ram()
    poweron.place_tune(ram, 0x0020, bytes(range(0x20, 0x40)))
    assert ram[0x2D] == 0x2D and ram[0x31] == 0x31  # the image's own bytes
    assert ram[0xAC:0xB0] == bytes((0x20, 0x00, 0x40, 0x00))


def test_flat_players_see_the_basic_load_pointers():
    """A tune reading ``$2D``/``$AC`` sees its bounds on every flat path, as sidplayfp."""
    from pysidtracker.asm6502 import assemble
    from pysidtracker.header import write_psid
    from pysidtracker.oracle import EmuPlayer, register_grid

    code, labels = assemble(
        """init: RTS
        play: LDA $2D
        STA $D400
        LDA $2E
        STA $D401
        LDA $AC
        STA $D405
        LDA $AD
        STA $D406
        RTS""",
        0x1000,
    )
    end = 0x1000 + len(code)
    want = (end & 0xFF, end >> 8, 0x00, 0x10)
    psid = write_psid(load=0x1000, init=0x1000, play=labels["play"], image=code)
    row = register_grid(psid, whole_song(1))[0]
    assert (row[0], row[1], row[5], row[6]) == want
    emu = EmuPlayer(code, 0x1000, 0x1000, labels["play"]).render_grid(whole_song(1))[0]
    assert (emu[0], emu[1], emu[5], emu[6]) == want


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
