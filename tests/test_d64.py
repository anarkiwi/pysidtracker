"""Release-media reading (vagrant-moses readers), ``media extract`` and disk fetches."""

import gzip
import io
import zipfile

import pytest

from pysidtracker import iter_prgs, maincli, read_d64, testing
from pysidtracker.d64 import MediaError
from pysidtracker.testing import fetch_disk, fetch_prgs

from .helpers import build_disk, build_t64

PRG, SEQ, LOCKED_PRG = 0x82, 0x81, 0xC2
TUNE = b"\x00\x20" + bytes(range(256)) * 3


def _zip_of(name, data):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, data)
    return buf.getvalue()


def test_read_d64_rejects_a_short_image():
    with pytest.raises(MediaError, match="not a disk image"):
        read_d64(bytes(1000))


@pytest.mark.strain_exempt(reason="exercises code outside the measured package")
@pytest.mark.parametrize(
    "geometry, error_block, first_track",
    [
        ("d64", False, 1),
        ("d64", True, 1),
        ("d64-40", True, 36),
        ("d64-42", False, 41),
        ("d71", True, 36),
        ("d81", False, 41),
    ],
)
def test_every_disk_geometry_is_read_by_size(geometry, error_block, first_track):
    disk = build_disk([("TUNE", TUNE, PRG)], geometry, error_block, first_track)
    assert list(iter_prgs("x.img", disk)) == [("x.img/TUNE", TUNE)]


@pytest.mark.strain_exempt(reason="exercises code outside the measured package")
def test_iter_prgs_walks_nested_zip_gzip_disk_and_tape():
    disk = gzip.compress(build_disk([("ON DISK", TUNE, PRG)]))
    tape = build_t64([("ONE", b"\x01\x08ab"), ("TWO", b"\x00\xc0cde")])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("rel/disk.d64.gz", disk)
        zf.writestr("rel/tape.t64", tape)
        zf.writestr("rel/bare.prg", b"\x00\x10z")
    assert list(iter_prgs("r.zip", buf.getvalue())) == [
        ("r.zip/rel/disk.d64/ON DISK", TUNE),
        ("r.zip/rel/tape.t64/ONE", b"\x01\x08ab"),
        ("r.zip/rel/tape.t64/TWO", b"\x00\xc0cde"),
        ("r.zip/rel/bare.prg", b"\x00\x10z"),
    ]


def test_media_extract_writes_each_prg_and_its_index_line(tmp_path, capsys):
    src = tmp_path / "dl"
    src.mkdir()
    (src / "rel.zip").write_bytes(_zip_of("d.d64", build_disk([("T", TUNE, PRG)])))
    out = tmp_path / "prg"
    assert maincli.main(["media", "extract", str(src), "-o", str(out)]) == 0
    assert (out / "rel.zip" / "d.d64" / "T").read_bytes() == TUNE
    line = capsys.readouterr().out.split()
    assert line[1:] == [str(len(TUNE)), "$2000", "rel.zip/d.d64/T"]


def test_media_extract_fails_on_unreadable_media(tmp_path, capsys):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"PK\x03\x04broken")
    assert maincli.main(["media", "extract", str(bad), "-o", str(tmp_path)]) == 1
    assert "1 unreadable media" in capsys.readouterr().err


def test_fetch_disk_raw_and_prgs(monkeypatch, tmp_path):
    d64 = build_disk([("TUNE", b"\x00\x20abc", PRG)])
    monkeypatch.setattr(testing, "_download", lambda url, **k: d64)
    path = fetch_disk("http://x/game.d64", cache_dir=tmp_path)
    assert path.name == "game.d64" and path.read_bytes() == d64
    monkeypatch.setattr(
        testing, "_download", lambda *a, **k: pytest.fail("should be cached")
    )
    files = fetch_prgs("http://x/game.d64", cache_dir=tmp_path)
    assert [f.name for f in files] == ["TUNE"]


def test_fetch_disk_from_zip_member(monkeypatch, tmp_path):
    blob = _zip_of("release/withtunes.d64", build_disk([("T", b"\x00\x20z", PRG)]))
    monkeypatch.setattr(testing, "_download", lambda url, **k: blob)
    path = fetch_disk("http://x/rel.zip", cache_dir=tmp_path, member="withtunes.d64")
    assert path.name == "withtunes.d64" and read_d64(path.read_bytes())[0].name == "T"


def test_fetch_disk_zip_missing_member_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(
        testing, "_download", lambda url, **k: _zip_of("other.txt", b"")
    )
    with pytest.raises(testing.TuneFetchError, match="no withtunes.d64"):
        fetch_disk("http://x/rel.zip", cache_dir=tmp_path, member="withtunes.d64")
