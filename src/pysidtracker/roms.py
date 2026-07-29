"""C64 KERNAL/BASIC/CHARGEN ROM fetch/cache and 64 KiB memory mapping.

BASIC tunes and unclean rips need the real ROMs. Being Commodore copyright works
they are never committed: they are fetched on demand into the gitignored fixture
cache and SHA-256 verified, as the HVSC ``.sid`` fixtures are (docs/c64-roms.md).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Iterable, NamedTuple, Optional

from .errors import SidError
from .image import MEM_SIZE
from .testing import TuneFetchError, _atomic_write, _download, default_tune_cache

# VICE's own C64 ROM set at an immutable mirror tag; ``$C64_ROM_MIRROR`` overrides.
DEFAULT_ROM_MIRROR = (
    "https://raw.githubusercontent.com/VICE-Team/svn-mirror/r46192/vice/data/C64"
)

# $00/$01 CPU port power-on state: BASIC + KERNAL + I/O visible, CHARGEN hidden.
CPU_DDR, CPU_PORT = 0x0000, 0x0001
DDR_DEFAULT, BANK_DEFAULT = 0x2F, 0x37


class Rom(NamedTuple):
    """One C64 ROM: mirror filename, mapped base address, size and SHA-256."""

    filename: str
    base: int
    size: int
    sha256: str


ROMS: Dict[str, Rom] = {
    "kernal": Rom(
        "kernal-901227-03.bin",
        0xE000,
        0x2000,
        "83c60d47047d7beab8e5b7bf6f67f80daa088b7a6a27de0d7e016f6484042721",
    ),
    "basic": Rom(
        "basic-901226-01.bin",
        0xA000,
        0x2000,
        "89878cea0a268734696de11c4bae593eaaa506465d2029d619c0e0cbccdfa62d",
    ),
    "chargen": Rom(
        "chargen-901225-01.bin",
        0xD000,
        0x1000,
        "fd0d53b8480e86163ac98998976c72cc58d5dd8eb824ed7b829774e74213b420",
    ),
}

# CHARGEN excluded: at $D000 it aliases the I/O bank a flat image must keep.
DEFAULT_ROMS = ("kernal", "basic")

KERNAL_BOOT = (0xFD15, 0xFDA3, 0xFF5B)  # RESTOR, IOINIT, CINT (docs/c64-roms.md)
BASIC_BOOT = (0xE453, 0xE3BF)  # BASIC vectors, BASIC RAM
BASIC_RUN = 0xA7B1  # BASIC interpreter main loop, where RUN enters


class RomFetchError(SidError):
    """A C64 ROM is unavailable, or the fetched bytes failed their checksum."""


def default_rom_cache(root=None) -> Path:
    """The ROM-cache directory: ``$PYSID_ROMCACHE`` or ``<tune cache>/roms``.

    Nested inside :func:`~pysidtracker.testing.default_tune_cache` so one
    gitignore entry and one CI ``actions/cache`` step cover both.
    """
    env = os.environ.get("PYSID_ROMCACHE")
    if env:
        return Path(env)
    return default_tune_cache(root) / "roms"


def _spec(name: str) -> Rom:
    try:
        return ROMS[name]
    except KeyError:
        raise RomFetchError(
            f"unknown ROM {name!r} (known: {', '.join(sorted(ROMS))})"
        ) from None


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _matches(spec: Rom, path: Path) -> bool:
    data = path.read_bytes()
    return len(data) == spec.size and _digest(data) == spec.sha256


def fetch_rom(
    name: str,
    *,
    cache_dir=None,
    mirror: str = DEFAULT_ROM_MIRROR,
    force: bool = False,
) -> Path:
    """Fetch the ``name`` ROM into ``cache_dir``; return its verified path.

    A cached copy whose SHA-256 matches is returned untouched, so a warm cache
    never hits the network; otherwise ``mirror``/``filename`` is downloaded
    (``$C64_ROM_MIRROR`` overrides), checked against :data:`ROMS`, written atomically.

    Raises:
      RomFetchError: the mirror is unreachable or the bytes fail their checksum.
    """
    spec = _spec(name)
    root = default_rom_cache() if cache_dir is None else Path(cache_dir)
    dest = root / spec.filename
    if dest.exists() and not force and _matches(spec, dest):
        return dest
    base = os.environ.get("C64_ROM_MIRROR", mirror).rstrip("/")
    try:
        data = _download(f"{base}/{spec.filename}")
    except TuneFetchError as exc:
        raise RomFetchError(f"{name}: {exc}") from exc
    if len(data) != spec.size or _digest(data) != spec.sha256:
        raise RomFetchError(
            f"{name}: checksum mismatch ({len(data)} bytes, sha256 {_digest(data)})"
        )
    _atomic_write(dest, data)
    return dest


def resolve_rom(name: str, *, cache_dir=None, local_env: str = "C64_ROMS"):
    """Path to the ``name`` ROM, or ``None`` when it is genuinely unavailable.

    Checks a local ROM directory (``$C64_ROMS``, holding either the dated VICE
    filename or the plain ``kernal``/``basic``/``chargen`` name), then the cache,
    then the mirror; ``None`` only when uncached and offline.
    """
    spec = _spec(name)
    local = os.environ.get(local_env)
    if local:
        for cand in (Path(local) / spec.filename, Path(local) / name):
            if cand.exists() and _matches(spec, cand):
                return cand
    try:
        return fetch_rom(name, cache_dir=cache_dir)
    except RomFetchError:
        return None


def apply_roms(
    mem: bytearray,
    names: Iterable[str] = DEFAULT_ROMS,
    *,
    cache_dir=None,
    local_env: str = "C64_ROMS",
    bank: Optional[int] = BANK_DEFAULT,
) -> bytearray:
    """Map the named ROMs into the 64 KiB image ``mem``; return ``mem``.

    Bases come from :data:`ROMS`; ``bank`` (``None`` to skip) is stored to ``$01``
    with :data:`DDR_DEFAULT` in ``$00``. A flat image has no shadow RAM, so a tune
    writing under a ROM or re-banking ``$01`` itself needs a paging host.
    """
    if len(mem) != MEM_SIZE:
        raise RomFetchError(f"image is {len(mem)} bytes, expected {MEM_SIZE}")
    for name in names:
        spec = _spec(name)
        path = resolve_rom(name, cache_dir=cache_dir, local_env=local_env)
        if path is None:
            raise RomFetchError(f"{name}: unavailable (offline, not cached)")
        mem[spec.base : spec.base + spec.size] = path.read_bytes()
    if bank is not None:
        mem[CPU_DDR], mem[CPU_PORT] = DDR_DEFAULT, bank
    return mem


def rom_image(names: Iterable[str] = DEFAULT_ROMS, **kwargs) -> bytearray:
    """A ``bytearray(0x10000)`` with the named ROMs mapped (see :func:`apply_roms`)."""
    return apply_roms(bytearray(MEM_SIZE), names, **kwargs)


def roms_available(
    names: Iterable[str] = DEFAULT_ROMS, *, cache_dir=None, local_env: str = "C64_ROMS"
) -> bool:
    """True when every named ROM resolves (local, cached, or fetchable)."""
    return all(
        resolve_rom(n, cache_dir=cache_dir, local_env=local_env) is not None
        for n in names
    )


__all__ = [
    "BANK_DEFAULT",
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
    "resolve_rom",
    "rom_image",
    "roms_available",
]
