"""libsidplayfp's power-on C64 RAM: the machine a PSID or RSID boots onto.

The fill is vagrant-moses' ``c64.poweron_ram`` (``ram_pattern`` wraps it
mutable); ``power_on_ram`` layers the PSID driver's ``$0000-$03FE`` clear and
zeroed SID port on it, and ``place_tune`` loads a tune with the BASIC ``LOAD``
pointers. RSID tunes boot from the fetched ``roms.poweron_writes``.
"""

from __future__ import annotations

from vagrant_moses import c64
from vagrant_moses.memmap import SID_BASE, SID_LAST

#: ``psiddrv::install`` clears ``$0000-$03FE`` (``0x3FF`` bytes) before a PSID runs.
PSID_DRIVER_CLEARED = 0x3FF
#: The BASIC LOAD pointers ``SidTuneBase::placeSidTuneInC64mem`` sets: the load end
#: in ``$2D``/``$2F``/``$31`` (variables / arrays / strings start) and ``$AE``, the
#: load start in ``$AC``.
BASIC_END_POINTERS = (0x2D, 0x2F, 0x31, 0xAE)
BASIC_START_POINTER = 0xAC


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


def place_tune(ram, load: int, image) -> None:
    """Load ``image`` at ``load`` into ``ram`` as libsidplayfp places a tune.

    The zero-page pointers a BASIC ``LOAD`` leaves (:data:`BASIC_END_POINTERS`,
    :data:`BASIC_START_POINTER`) are set first, so a tune reading them sees its own
    bounds, then the image, which wins where it covers them.
    """
    load &= 0xFFFF
    end = (load + len(image)) & 0xFFFF
    for addr in BASIC_END_POINTERS:
        ram[addr : addr + 2] = end.to_bytes(2, "little")
    ram[BASIC_START_POINTER : BASIC_START_POINTER + 2] = load.to_bytes(2, "little")
    ram[load : load + len(image)] = image


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
    "BASIC_END_POINTERS",
    "BASIC_START_POINTER",
    "PSID_DRIVER_CLEARED",
    "place_tune",
    "power_on_ram",
    "psid_init_bank",
    "psid_play_bank",
    "ram_pattern",
]
