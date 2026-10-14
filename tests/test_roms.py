"""Tests for the C64 ROM fetch/cache and 64 KiB mapping helper.

The default (hermetic) run never touches the network: the download is
monkeypatched. The ``roms``-marked test at the end fetches the real ROMs and is
excluded from the default matrix, running in its own CI job.
"""

import hashlib

import pytest

from pysidtracker import roms
from pysidtracker.image import MEM_SIZE
from pysidtracker.roms import (
    BANK_DEFAULT,
    DEFAULT_ROMS,
    DEFAULT_ROM_MIRROR,
    ROMS,
    RomFetchError,
    apply_roms,
    default_rom_cache,
    fetch_rom,
    resolve_rom,
    rom_image,
    roms_available,
)
from pysidtracker.testing import TuneFetchError

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
        for name, spec in ROMS.items():
            if url.endswith(spec.filename):
                return FAKE[name]
        raise TuneFetchError(f"{url}: not found")

    monkeypatch.setattr(roms, "_download", fake)
    return calls


def test_rom_map_is_the_c64_layout():
    assert [(r.base, r.size) for r in ROMS.values()] == [
        (0xE000, 0x2000),
        (0xA000, 0x2000),
        (0xD000, 0x1000),
    ]
    assert DEFAULT_ROMS == ("kernal", "basic")  # CHARGEN would alias the I/O bank
    assert DEFAULT_ROM_MIRROR.startswith("https://")


def test_default_rom_cache_env_and_default(monkeypatch, tmp_path):
    assert default_rom_cache(tmp_path) == tmp_path / ".tunecache" / "roms"
    monkeypatch.setenv("PYSID_ROMCACHE", str(tmp_path / "r"))
    assert default_rom_cache(tmp_path) == tmp_path / "r"


def test_unknown_rom_raises():
    with pytest.raises(RomFetchError, match="unknown ROM"):
        fetch_rom("charen")


def test_fetch_rom_downloads_then_caches(served, tmp_path):
    dest = fetch_rom("kernal", cache_dir=tmp_path)
    assert dest.read_bytes() == FAKE["kernal"]
    assert fetch_rom("kernal", cache_dir=tmp_path) == dest
    assert len(served) == 1  # warm cache: no second download
    fetch_rom("kernal", cache_dir=tmp_path, force=True)
    assert len(served) == 2


def test_fetch_rom_refetches_a_corrupt_cache_entry(served, tmp_path):
    dest = fetch_rom("basic", cache_dir=tmp_path)
    dest.write_bytes(b"\x00" * ROMS["basic"].size)
    assert fetch_rom("basic", cache_dir=tmp_path).read_bytes() == FAKE["basic"]
    assert len(served) == 2


def test_fetch_rom_checksum_mismatch_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(roms, "_download", lambda url, **kw: b"\xea" * 0x2000)
    with pytest.raises(RomFetchError, match="checksum mismatch"):
        fetch_rom("kernal", cache_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_fetch_rom_unreachable_raises(monkeypatch, tmp_path):
    def boom(url, **_kw):
        raise TuneFetchError("offline")

    monkeypatch.setattr(roms, "_download", boom)
    with pytest.raises(RomFetchError, match="offline"):
        fetch_rom("kernal", cache_dir=tmp_path)


def test_fetch_rom_honours_mirror_env(served, monkeypatch, tmp_path):
    monkeypatch.setenv("C64_ROM_MIRROR", "https://mirror.example/roms/")
    fetch_rom("kernal", cache_dir=tmp_path)
    assert served == ["https://mirror.example/roms/kernal-901227-03.bin"]


def test_resolve_rom_prefers_a_local_dir(served, monkeypatch, tmp_path):
    local = tmp_path / "vice"
    local.mkdir()
    (local / ROMS["kernal"].filename).write_bytes(FAKE["kernal"])
    (local / "basic").write_bytes(FAKE["basic"])
    monkeypatch.setenv("C64_ROMS", str(local))
    assert resolve_rom("kernal", cache_dir=tmp_path).parent == local
    assert resolve_rom("basic", cache_dir=tmp_path).name == "basic"
    assert served == []


def test_resolve_rom_ignores_a_bad_local_copy(served, monkeypatch, tmp_path):
    local = tmp_path / "vice"
    local.mkdir()
    (local / ROMS["kernal"].filename).write_bytes(b"\x00" * 0x2000)
    monkeypatch.setenv("C64_ROMS", str(local))
    assert resolve_rom("kernal", cache_dir=tmp_path).parent == tmp_path
    assert len(served) == 1


def test_resolve_rom_none_when_offline(monkeypatch, tmp_path):
    def boom(url, **_kw):
        raise TuneFetchError("offline")

    monkeypatch.setattr(roms, "_download", boom)
    assert resolve_rom("kernal", cache_dir=tmp_path) is None
    assert roms_available(cache_dir=tmp_path) is False


@pytest.mark.usefixtures("served")
def test_rom_image_maps_kernal_and_basic(tmp_path):
    mem = rom_image(cache_dir=tmp_path)
    assert len(mem) == MEM_SIZE
    assert bytes(mem[0xE000:]) == FAKE["kernal"]
    assert bytes(mem[0xA000:0xC000]) == FAKE["basic"]
    assert mem[0x01] == BANK_DEFAULT and mem[0x00] == 0x2F
    assert not any(mem[0xD000:0xE000])  # I/O bank, not CHARGEN
    assert roms_available(cache_dir=tmp_path) is True


@pytest.mark.usefixtures("served")
def test_apply_roms_overlays_an_existing_image(tmp_path):
    mem = bytearray(b"\xaa" * MEM_SIZE)
    assert apply_roms(mem, ("chargen",), cache_dir=tmp_path, bank=None) is mem
    assert bytes(mem[0xD000:0xE000]) == FAKE["chargen"]
    assert mem[0x01] == 0xAA  # bank=None leaves the CPU port alone
    assert mem[0x9FFF] == 0xAA


def test_apply_roms_rejects_a_wrong_sized_image(tmp_path):
    with pytest.raises(RomFetchError, match="expected"):
        apply_roms(bytearray(16), cache_dir=tmp_path)


def test_apply_roms_raises_when_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(roms, "resolve_rom", lambda *a, **k: None)
    with pytest.raises(RomFetchError, match="unavailable"):
        rom_image(cache_dir=tmp_path)


@pytest.mark.roms
def test_real_roms_fetch_verify_and_map(c64_roms):
    """Fetch the real ROMs into the shared cache and check they are the C64 ones."""
    mem = c64_roms.image(tuple(ROMS))
    assert mem[0xFFFC] | mem[0xFFFD] << 8 == 0xFCE2  # KERNAL reset vector
    assert mem[0xFFFE] | mem[0xFFFF] << 8 == 0xFF48  # IRQ/BRK vector
    assert mem[0xA000] | mem[0xA001] << 8 == 0xE394  # BASIC cold start
    assert bytes(mem[0xD000:0xD008]) == bytes.fromhex("3c666e6e60623c00")  # '@' glyph
    assert [c64_roms(n).name for n in ROMS] == [r.filename for r in ROMS.values()]


def test_poweron_writes_decode_libsidplayfps_rle(tmp_path, monkeypatch):
    path = tmp_path / "poweron.bin"
    path.write_text(
        "/* addr, off, rle, values */\n"
        "/*$0003*/ 0x83, 0x01, 0xaa, 0xbb,\n"
        "/*$0008*/ 0x03,       0x4c,\n"
        "/*$000c*/ 0x83, 0x82, 0x00,\n"
    )
    monkeypatch.setattr(roms, "resolve_rom", lambda *a, **k: path)
    assert roms.poweron_writes() == [(3, b"\xaa\xbb"), (8, b"\x4c"), (12, bytes(3))]


def test_poweron_writes_offline_is_none(monkeypatch):
    monkeypatch.setattr(roms, "resolve_rom", lambda *a, **k: None)
    assert roms.poweron_writes() is None


def test_poweron_is_pinned_to_the_oracles_libsidplayfp():
    spec = roms.ASSETS["poweron"]
    assert spec.url.endswith(f"/{spec.filename}") and "9c464110" in spec.url
    with pytest.raises(RomFetchError, match="unknown ROM"):
        fetch_rom("nonesuch")


@pytest.mark.roms
def test_real_poweron_image_holds_the_kernal_vectors():
    ram = bytearray(0x400)
    for addr, data in roms.poweron_writes():
        ram[addr : addr + len(data)] = data
    assert ram[0x314:0x31A] == bytes.fromhex("31ea66fe47fe")
