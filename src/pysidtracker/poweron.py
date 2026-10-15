"""libsidplayfp's power-on C64 RAM: the machine a PSID or RSID boots onto.

The fill is vagrant-moses' ``c64.poweron_ram`` (``ram_pattern`` wraps it
mutable); ``power_on_ram`` layers the PSID driver's ``$0000-$03FE`` clear and
zeroed SID port on it. RSID tunes boot from the fetched ``roms.poweron_writes``.
"""

from __future__ import annotations

from vagrant_moses import c64  # core dependency

#: ``psiddrv::install`` clears ``$0000-$03FE`` (``0x3FF`` bytes) before a PSID runs.
PSID_DRIVER_CLEARED = 0x3FF
# The SID occupies ``$D400-$D7FF``; before the tune sets it the register file reads as 0.
SID_PORT = 0xD400
SID_PORT_SIZE = 0x400


def ram_pattern() -> bytearray:
    """libsidplayfp's power-up RAM (vagrant-moses ``c64.poweron_ram``), mutable."""
    return bytearray(c64.poweron_ram())


def power_on_ram() -> bytearray:
    """The PSID start-up image a player or the flat emulator boots from.

    The power-on RAM (:func:`ram_pattern`) with the PSID driver's ``$0000-$03FE``
    clear (``psiddrv::install``) and the uninitialised SID register file at 0
    (the reSID's reset state; only ``$D418`` is then seeded by the driver). A tune
    reading uninitialised RAM reads the power-on byte, but a register it never
    writes reads the oracle's 0 baseline, as libsidplayfp plays it.
    """
    ram = ram_pattern()
    ram[:PSID_DRIVER_CLEARED] = bytes(PSID_DRIVER_CLEARED)
    ram[SID_PORT : SID_PORT + SID_PORT_SIZE] = bytes(SID_PORT_SIZE)
    return ram


def psid_init_bank(init_address: int) -> int:
    """The driver's ``$01`` bank register before running ``init`` at ``init_address``.

    ``psiddrv`` banks the init's own ROM space in (so a tune that reads
    KERNAL/BASIC sees the oracle's ROMs) and the rest of the map out: ``$37`` for
    init below BASIC, else the map for the ROM the init lives in.
    """
    if init_address < 0xA000:
        return 0x37
    if init_address < 0xD000:
        return 0x36
    return 0x35 if init_address >= 0xE000 else 0x34


__all__ = [
    "PSID_DRIVER_CLEARED",
    "SID_PORT",
    "power_on_ram",
    "psid_init_bank",
    "ram_pattern",
]
