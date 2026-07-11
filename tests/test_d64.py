"""Tests for the .d64 disk-image reader and the disk fetch helpers."""

import io
import zipfile

import pytest

from pysidtracker import D64File, read_d64
from pysidtracker import testing
from pysidtracker.d64 import (
    D64_SIZE,
    DIR_SECTOR,
    DIR_TRACK,
    FILETYPE_PRG,
    _ts_offset,
)
from pysidtracker.testing import fetch_disk, fetch_prgs


def _build_d64(files):
    """A minimal valid .d64 holding ``files`` (list of ``(name, prg_bytes)``).

    Each file gets one sector on track 1; the single directory block on track 18
    lists them. Payloads must fit one sector (<= 254 bytes) -- enough for tests.
    """
    img = bytearray(D64_SIZE)
    dir_off = _ts_offset(DIR_TRACK, DIR_SECTOR)
    img[dir_off] = 0  # last (only) directory block
    img[dir_off + 1] = 0xFF
    for slot, (name, prg) in enumerate(files):
        file_sector = slot
        foff = _ts_offset(1, file_sector)
        img[foff] = 0  # terminal file block
        img[foff + 1] = len(prg) + 1  # index of last used byte
        img[foff + 2 : foff + 2 + len(prg)] = prg
        entry = dir_off + 2 + slot * 32
        img[entry] = FILETYPE_PRG
        img[entry + 1] = 1  # start track
        img[entry + 2] = file_sector
        padded = name.encode("latin-1").ljust(16, b"\xa0")[:16]
        img[entry + 3 : entry + 19] = padded
    return bytes(img)


def test_read_d64_extracts_prgs_in_order():
    prgs = [(b"\x00\x20" + b"hello"), (b"\x00\x08" + bytes(range(40)))]
    d64 = _build_d64([("FIRST", prgs[0]), ("SECOND TUNE", prgs[1])])
    files = read_d64(d64)
    assert files == [D64File("FIRST", prgs[0]), D64File("SECOND TUNE", prgs[1])]
    # The name is PETSCII-trimmed of the 0xA0 shift-space padding.
    assert files[0].name == "FIRST"


def test_read_d64_skips_non_prg_slots():
    d64 = bytearray(_build_d64([("A", b"\x00\x20xy")]))
    # A second slot that is not a PRG (filetype 0) is ignored.
    d64[_ts_offset(DIR_TRACK, DIR_SECTOR) + 2 + 32] = 0x00
    assert [f.name for f in read_d64(bytes(d64))] == ["A"]


def test_read_d64_too_short_raises():
    with pytest.raises(ValueError, match="not a d64"):
        read_d64(b"\x00" * (D64_SIZE - 1))


def test_read_d64_detects_sector_loop():
    d64 = bytearray(_build_d64([("A", b"\x00\x20xy")]))
    foff = _ts_offset(1, 0)
    d64[foff] = 1  # next track -> self, a loop
    d64[foff + 1] = 0
    with pytest.raises(ValueError, match="loops"):
        read_d64(bytes(d64))


def _zip_of(name, d64_bytes):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, d64_bytes)
    return buf.getvalue()


def test_fetch_disk_raw_and_prgs(monkeypatch, tmp_path):
    d64 = _build_d64([("TUNE", b"\x00\x20abc")])
    monkeypatch.setattr(testing, "_download", lambda url, **k: d64)
    path = fetch_disk("http://x/game.d64", cache_dir=tmp_path)
    assert path.name == "game.d64" and path.read_bytes() == d64
    # Second call is a cache hit (download would raise if re-called).
    monkeypatch.setattr(
        testing, "_download", lambda *a, **k: pytest.fail("should be cached")
    )
    files = fetch_prgs("http://x/game.d64", cache_dir=tmp_path)
    assert [f.name for f in files] == ["TUNE"]


def test_fetch_disk_from_zip_member(monkeypatch, tmp_path):
    d64 = _build_d64([("T", b"\x00\x20z")])
    blob = _zip_of("release/withtunes.d64", d64)
    monkeypatch.setattr(testing, "_download", lambda url, **k: blob)
    path = fetch_disk("http://x/rel.zip", cache_dir=tmp_path, member="withtunes.d64")
    assert path.name == "withtunes.d64" and read_d64(path.read_bytes())[0].name == "T"


def test_fetch_disk_zip_missing_member_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(
        testing, "_download", lambda url, **k: _zip_of("other.txt", b"")
    )
    with pytest.raises(testing.TuneFetchError, match="no withtunes.d64"):
        fetch_disk("http://x/rel.zip", cache_dir=tmp_path, member="withtunes.d64")
