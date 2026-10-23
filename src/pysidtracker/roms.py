"""C64 ROMs from :mod:`vagrant_moses.roms`, cached beside the HVSC tunes.

The fetch, verify and mapping logic is vagrant-moses'; these wrappers only default
``cache_dir`` to :func:`default_rom_cache` and raise :class:`RomFetchError` as a
:class:`~pysidtracker.errors.SidError`. Nothing is committed (docs/c64-roms.md).
"""

from __future__ import annotations

import functools
from pathlib import Path

from vagrant_moses import roms as _roms
from vagrant_moses.roms import (
    ASSETS,
    BASIC_BOOT,
    BASIC_RUN,
    DEFAULT_ROM_MIRROR,
    DEFAULT_ROMS,
    KERNAL_BOOT,
    ROMS,
    Rom,
)

from .errors import SidError
from .testing import default_cache


class RomFetchError(SidError, _roms.RomFetchError):
    """A C64 ROM is unavailable, or the fetched bytes failed their checksum."""


def default_rom_cache(root=None) -> Path:
    """``default_cache("roms")``: one CI cache with the tunes."""
    return default_cache("roms", root)


def _cached(func):
    @functools.wraps(func)
    def wrapper(*args, cache_dir=None, **kwargs):
        try:
            return func(*args, cache_dir=cache_dir or default_rom_cache(), **kwargs)
        except _roms.RomFetchError as exc:
            raise RomFetchError(*exc.args) from exc

    return wrapper


fetch_rom = _cached(_roms.fetch_rom)
resolve_rom = _cached(_roms.resolve_rom)
load_roms = _cached(_roms.load_roms)
apply_roms = _cached(_roms.apply_roms)
rom_image = _cached(_roms.rom_image)
roms_available = _cached(_roms.roms_available)
poweron_writes = _cached(_roms.poweron_writes)

__all__ = [
    "ASSETS",
    "BASIC_BOOT",
    "BASIC_RUN",
    "DEFAULT_ROMS",
    "DEFAULT_ROM_MIRROR",
    "KERNAL_BOOT",
    "ROMS",
    "Rom",
    "RomFetchError",
    "apply_roms",
    "default_rom_cache",
    "fetch_rom",
    "load_roms",
    "poweron_writes",
    "resolve_rom",
    "rom_image",
    "roms_available",
]
