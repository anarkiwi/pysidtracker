"""libsidplayfp's power-on C64 RAM: the machine a PSID or RSID boots onto.

The fill is vagrant-moses' ``c64.poweron_ram`` (``ram_pattern`` wraps it
mutable); ``power_on_ram`` layers the PSID driver's ``$0000-$03FE`` clear and
zeroed SID port on it. RSID tunes boot from the fetched ``roms.poweron_writes``.
"""

from __future__ import annotations

from vagrant_moses import c64
from vagrant_moses.memmap import SID_BASE, SID_LAST

#: ``psiddrv::install`` clears ``$0000-$03FE`` (``0x3FF`` bytes) before a PSID runs.
PSID_DRIVER_CLEARED = 0x3FF


def ram_pattern() -> bytearray:
    """libsidplayfp's power-up RAM (vagrant-moses ``c64.poweron_ram``), mutable."""
    return bytearray(c64.poweron_ram())


def power_on_ram() -> bytearray:
    """The PSID start-up image a player or the flat emulator boots from.

    The power-on RAM (:func:`ram_pattern`) with the PSID driver's ``$0000-$03FE``
    clear and the uninitialised SID register file at 0, as libsidplayfp plays it.
    """
    ram = ram_pattern()
    ram[:PSID_DRIVER_CLEARED] = bytes(PSID_DRIVER_CLEARED)
    ram[SID_BASE : SID_LAST + 1] = bytes(SID_LAST + 1 - SID_BASE)
    return ram


def _iomap(address: int) -> int:
    """libsidplayfp ``psiddrv::iomap``: the ``$01`` map for code at ``address``.

    ``$37`` (BASIC + KERNAL ROM, I/O) below BASIC, ``$36`` (KERNAL ROM, I/O) in
    ``$A000-$CFFF``, ``$35`` (I/O only) above ``$DFFF``, ``$34`` (RAM only) in
    ``$D000-$DFFF``.
    """
    if address < 0xA000:
        return 0x37
    if address < 0xD000:
        return 0x36
    return 0x35 if address >= 0xE000 else 0x34


def psid_init_bank(init_address: int) -> int:
    """The driver's ``$01`` bank register before running ``init`` (init's ROM in)."""
    return _iomap(init_address)


def psid_play_bank(play_address: int) -> int:
    """The driver's ``$01`` bank register around the play calls (play's ROM in).

    ``psiddrv`` sets this around every play call, so a play in ``$A000-$CFFF``
    reads the RAM behind the BASIC window (the tune's loaded data), not ROM.
    """
    return _iomap(play_address)


__all__ = [
    "PSID_DRIVER_CLEARED",
    "power_on_ram",
    "psid_init_bank",
    "psid_play_bank",
    "ram_pattern",
]
