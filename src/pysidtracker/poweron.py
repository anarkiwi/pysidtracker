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
    """The PSID start-up image: :func:`ram_pattern` with ``$0000-$03FE`` cleared.

    ``psiddrv::install`` zeroes :data:`PSID_DRIVER_CLEARED` bytes before running
    init, so a tune reading uninitialised memory there reads zeros as libsidplayfp plays it.
    """
    ram = ram_pattern()
    ram[:PSID_DRIVER_CLEARED] = bytes(PSID_DRIVER_CLEARED)
    return ram


__all__ = ["PSID_DRIVER_CLEARED", "power_on_ram", "ram_pattern"]
