"""libsidplayfp's power-on C64 RAM: the machine a PSID or RSID boots onto.

A tune reads uninitialised memory as the C64 had it at power-on, not as zeros.
libsidplayfp models that RAM analytically (``SystemRAMBank::reset``,
:func:`ram_pattern`) and its PSID driver clears ``$0000-$03FE`` before running
init (``psiddrv::install``, :func:`power_on_ram`). That start-up image describes
the oracle's machine, not any one tracker, so it lives here as the single source
the flat :class:`~pysidtracker.player.MemPlayer` path and the
:func:`~pysidtracker.emu.wire_mpu` wiring build on. RSID (real-C64) tunes boot
from the fetched power-on image instead -- :func:`~pysidtracker.roms.poweron_writes`,
falling back to the KERNAL vectors in :mod:`~pysidtracker.irqhost`.
"""

from __future__ import annotations

#: ``psiddrv::install`` clears ``$0000-$03FE`` (``0x3FF`` bytes) before a PSID runs.
PSID_DRIVER_CLEARED = 0x3FF
# The SID occupies ``$D400-$D7FF``; before the tune sets it the register file reads as 0.
SID_PORT = 0xD400
SID_PORT_SIZE = 0x400


def ram_pattern() -> bytearray:
    """libsidplayfp's power-up RAM: 4-byte runs of ``$00``/``$FF``, flipped per 16K bank."""
    ram = bytearray()
    byte = 0
    for _ in range(4):
        block = bytearray([byte]) * 0x4000
        for i in range(2, 0x4000, 8):
            block[i : i + 4] = bytes([byte ^ 0xFF]) * 4
        ram += block
        byte ^= 0xFF
    return ram


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


__all__ = [
    "PSID_DRIVER_CLEARED",
    "SID_PORT",
    "power_on_ram",
    "ram_pattern",
]
