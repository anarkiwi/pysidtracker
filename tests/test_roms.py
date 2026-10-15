"""The tune-cache wrappers over :mod:`vagrant_moses.roms`.

The default (hermetic) run never touches the network: vagrant-moses' download is
monkeypatched. The ``roms``-marked tests fetch the real ROMs in their own CI job.
"""

import hashlib

import pytest
from vagrant_moses import roms as vm_roms

from pysidtracker import SidError, roms
from pysidtracker.roms import (
    ROMS,
    RomFetchError,
    default_rom_cache,
    fetch_rom,
    resolve_rom,
    rom_image,
    roms_available,
)

FAKE = {
    name: bytes((i + 1,)) * spec.size for i, (name, spec) in enumerate(ROMS.items())
}


@pytest.fixture(autouse=True)
def _no_local(monkeypatch):
    monkeypatch.delenv("C64_ROMS", raising=False)
    monkeypatch.delenv("C64_ROM_MIRROR", raising=False)
    monkeypatch.delenv("PYSID_ROMCACHE", raising=False)


@pytest.fixture
def served(monkeypatch):
    """Serve the fake ROM bytes under their real checksums; record the URLs."""
    for name, data in FAKE.items():
        monkeypatch.setitem(
            ROMS, name, ROMS[name]._replace(sha256=hashlib.sha256(data).hexdigest())
        )
    calls = []

    def fake(url, **_kw):
        calls.append(url)
        return next(FAKE[n] for n, spec in ROMS.items() if url.endswith(spec.filename))

    monkeypatch.setattr(vm_roms, "_download", fake)
    return calls


def _offline(monkeypatch):
    def boom(url, **_kw):
        raise vm_roms.RomFetchError(f"{url}: offline")

    monkeypatch.setattr(vm_roms, "_download", boom)


def test_default_rom_cache_env_and_default(monkeypatch, tmp_path):
    assert default_rom_cache(tmp_path) == tmp_path / ".tunecache" / "roms"
    monkeypatch.setenv("PYSID_ROMCACHE", str(tmp_path / "r"))
    assert default_rom_cache(tmp_path) == tmp_path / "r"


def test_fetches_default_to_the_tune_cache(served, monkeypatch, tmp_path):
    monkeypatch.setenv("PYSID_ROMCACHE", str(tmp_path))
    assert fetch_rom("kernal") == tmp_path / ROMS["kernal"].filename
    assert fetch_rom("kernal", cache_dir=tmp_path / "x").parent == tmp_path / "x"
    assert len(served) == 2


def test_rom_errors_are_sid_errors_and_vagrant_moses_errors(monkeypatch, tmp_path):
    with pytest.raises(RomFetchError, match="unknown ROM") as exc:
        fetch_rom("charen")
    assert isinstance(exc.value, SidError)
    assert isinstance(exc.value, vm_roms.RomFetchError)
    _offline(monkeypatch)
    with pytest.raises(RomFetchError, match="offline"):
        fetch_rom("kernal", cache_dir=tmp_path)


def test_resolve_rom_none_when_offline(monkeypatch, tmp_path):
    _offline(monkeypatch)
    assert resolve_rom("kernal", cache_dir=tmp_path) is None
    assert roms_available(cache_dir=tmp_path) is False
    with pytest.raises(RomFetchError, match="unavailable"):
        rom_image(cache_dir=tmp_path)
    assert roms.poweron_writes(cache_dir=tmp_path) is None


@pytest.mark.usefixtures("served")
def test_rom_image_maps_kernal_and_basic(tmp_path):
    mem = rom_image(cache_dir=tmp_path)
    assert bytes(mem[0xE000:]) == FAKE["kernal"]
    assert bytes(mem[0xA000:0xC000]) == FAKE["basic"]
    assert roms_available(cache_dir=tmp_path) is True


def test_poweron_is_pinned_to_the_oracles_libsidplayfp():
    spec = roms.ASSETS["poweron"]
    assert spec.url.endswith(f"/{spec.filename}") and "9c464110" in spec.url


@pytest.mark.roms
def test_real_roms_fetch_verify_and_map(c64_roms):
    """Fetch the real ROMs into the shared cache and check they are the C64 ones."""
    mem = c64_roms.image(tuple(ROMS))
    assert mem[0xFFFC] | mem[0xFFFD] << 8 == 0xFCE2  # KERNAL reset vector
    assert mem[0xFFFE] | mem[0xFFFF] << 8 == 0xFF48  # IRQ/BRK vector
    assert mem[0xA000] | mem[0xA001] << 8 == 0xE394  # BASIC cold start
    assert bytes(mem[0xD000:0xD008]) == bytes.fromhex("3c666e6e60623c00")  # '@' glyph
    assert [c64_roms(n).name for n in ROMS] == [r.filename for r in ROMS.values()]


@pytest.mark.roms
def test_real_poweron_image_holds_the_kernal_vectors():
    ram = bytearray(0x400)
    for addr, data in roms.poweron_writes():
        ram[addr : addr + len(data)] = data
    assert ram[0x314:0x31A] == bytes.fromhex("31ea66fe47fe")
