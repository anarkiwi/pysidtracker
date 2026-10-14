"""The shared libsidplayfp power-on RAM image that the players and emulator boot from."""

import numpy as np

from pysidtracker import MemPlayer, poweron
from pysidtracker import registers as reg
from pysidtracker import irqhost


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
    assert pattern[
        poweron.SID_PORT : poweron.SID_PORT + poweron.SID_PORT_SIZE
    ] == bytes(poweron.SID_PORT_SIZE)
    # ... everything else is the power-on pattern, untouched
    assert pattern[0x3FF : poweron.SID_PORT] == base[0x3FF : poweron.SID_PORT]
    assert (
        pattern[poweron.SID_PORT + poweron.SID_PORT_SIZE :]
        == base[poweron.SID_PORT + poweron.SID_PORT_SIZE :]
    )


def test_power_on_image_matches_the_reference_formula():
    addr = np.arange(0x10000)
    run = (addr & 7 >= 2) & (addr & 7 < 6)
    want = np.where(run ^ ((addr >> 14) & 1).astype(bool), 0xFF, 0).astype(np.uint8)
    want[:0x3FF] = 0
    want[0xD400:0xD800] = 0  # the SID register file at rest
    assert bytes(poweron.power_on_ram()) == want.tobytes()


def test_irqhost_shares_the_single_source_of_truth():
    assert poweron.ram_pattern() == irqhost.ram_pattern()
    base = poweron.ram_pattern()
    base[: poweron.PSID_DRIVER_CLEARED] = bytes(poweron.PSID_DRIVER_CLEARED)
    base[poweron.SID_PORT : poweron.SID_PORT + poweron.SID_PORT_SIZE] = bytes(
        poweron.SID_PORT_SIZE
    )
    assert base == poweron.power_on_ram()


class _ReadPlayer(MemPlayer):
    """A do-nothing player used only to inspect the boot memory image."""

    def _init(self, subtune):
        pass

    def _frame(self):
        pass


def test_memplayer_boots_from_the_poweron_image():
    player = _ReadPlayer(b"", 0x8000)
    want = poweron.power_on_ram()
    want[reg.SID_BASE + 0x18] = 0x0F  # the PSID driver seeds maximum volume
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
    assert player._rd(reg.SID_BASE + 0x18) == 0x0F
