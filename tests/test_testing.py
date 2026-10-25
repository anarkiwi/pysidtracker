"""Tests for the HVSC fetch/resolve test helper."""

import hashlib
import http.client
import io
import json
import tarfile
import threading
import time
import urllib.error
import zipfile

import pytest

from pysidtracker import testing
from pysidtracker.testing import (
    CACHE_KINDS,
    default_cache,
    DEFAULT_MIRROR,
    ChecksumMismatch,
    TuneFetchError,
    default_tune_cache,
    fetch_archive_member,
    fetch_tune,
    fetch_verified,
    gather_tune_relpaths,
    make_tune_fixtures,
    resolve_tune,
)
from pysidtracker.testing import main as fetch_main

_SID = b"PSID" + b"\x00" * 60


class _FakeResp:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data

    def geturl(self):
        return "final"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(testing.time, "sleep", lambda *_: None)


def _patch_urlopen(monkeypatch, fn):
    monkeypatch.setattr(testing.urllib.request, "urlopen", fn)


def test_default_mirror_constant():
    assert DEFAULT_MIRROR == "https://hvsc.brona.dk/HVSC/C64Music"


def test_fetch_tune_success_and_cache(monkeypatch, tmp_path):
    calls = []

    def fake(req, timeout=60):  # pylint: disable=unused-argument
        calls.append(req.full_url)
        return _FakeResp(_SID)

    _patch_urlopen(monkeypatch, fake)
    dest = fetch_tune("A/x.sid", cache_dir=tmp_path)
    assert dest.exists() and dest.read_bytes() == _SID
    assert len(calls) == 1
    # Second call is a cache hit: no new download.
    fetch_tune("A/x.sid", cache_dir=tmp_path)
    assert len(calls) == 1


def test_fetch_tune_honours_hvsc_mirror(monkeypatch, tmp_path):
    seen = {}

    def fake(req, timeout=60):  # pylint: disable=unused-argument
        seen["url"] = req.full_url
        return _FakeResp(_SID)

    monkeypatch.setenv("HVSC_MIRROR", "https://mirror.example/C64/")
    _patch_urlopen(monkeypatch, fake)
    fetch_tune("A/x.sid", cache_dir=tmp_path)
    assert seen["url"] == "https://mirror.example/C64/A/x.sid"


def test_fetch_tune_not_sid_raises(monkeypatch, tmp_path):
    _patch_urlopen(monkeypatch, lambda *a, **k: _FakeResp(b"NOPE" + b"\x00" * 8))
    with pytest.raises(TuneFetchError):
        fetch_tune("A/x.sid", cache_dir=tmp_path)


def test_fetch_tune_404_raises(monkeypatch, tmp_path):
    def fake(*_a, **_k):
        raise urllib.error.HTTPError("u", 404, "nf", {}, None)

    _patch_urlopen(monkeypatch, fake)
    with pytest.raises(TuneFetchError, match="not found"):
        fetch_tune("A/x.sid", cache_dir=tmp_path)


def test_fetch_tune_retries_then_fails(monkeypatch, tmp_path):
    attempts = []

    def fake(*_a, **_k):
        attempts.append(1)
        raise urllib.error.URLError("down")

    _patch_urlopen(monkeypatch, fake)
    with pytest.raises(TuneFetchError, match="unreachable"):
        fetch_tune("A/x.sid", cache_dir=tmp_path, retries=3)
    assert len(attempts) == 3


def test_resolve_tune_local_tree_first(monkeypatch, tmp_path):
    local = tmp_path / "hvsc"
    (local / "A").mkdir(parents=True)
    (local / "A" / "x.sid").write_bytes(_SID)
    monkeypatch.setenv("HVSC", str(local))
    got = resolve_tune("A/x.sid", cache_dir=tmp_path / "cache")
    assert got == local / "A" / "x.sid"


def test_resolve_tune_cache_second(monkeypatch, tmp_path):
    monkeypatch.delenv("HVSC", raising=False)
    cache = tmp_path / "cache"
    (cache / "A").mkdir(parents=True)
    (cache / "A" / "x.sid").write_bytes(_SID)
    got = resolve_tune("A/x.sid", cache_dir=cache)
    assert got == cache / "A" / "x.sid"


def test_resolve_tune_fetch_then_none(monkeypatch, tmp_path):
    monkeypatch.delenv("HVSC", raising=False)

    def fake(*_a, **_k):
        raise urllib.error.URLError("offline")

    _patch_urlopen(monkeypatch, fake)
    assert resolve_tune("A/x.sid", cache_dir=tmp_path / "cache") is None


def test_resolve_tune_fetches_when_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("HVSC", raising=False)
    _patch_urlopen(monkeypatch, lambda *a, **k: _FakeResp(_SID))
    got = resolve_tune("A/x.sid", cache_dir=tmp_path / "cache")
    assert got is not None and got.read_bytes() == _SID


def test_make_tune_fixtures_returns_two_fixtures():
    tune_id, tune_path = make_tune_fixtures({"x": "A/x.sid"}, "/tmp/cache")
    assert tune_id is not None and tune_path is not None


def test_download_success(monkeypatch):
    _patch_urlopen(monkeypatch, lambda *a, **k: _FakeResp(b"blob"))
    assert testing._download("http://x/f") == b"blob"


def test_download_404_raises(monkeypatch):
    def fake(*_a, **_k):
        raise urllib.error.HTTPError("u", 404, "nf", {}, None)

    _patch_urlopen(monkeypatch, fake)
    with pytest.raises(TuneFetchError, match="not found"):
        testing._download("http://x/f")


def test_download_retries_then_fails(monkeypatch):
    attempts = []

    def fake(*_a, **_k):
        attempts.append(1)
        raise urllib.error.URLError("down")

    _patch_urlopen(monkeypatch, fake)
    with pytest.raises(TuneFetchError, match="unreachable"):
        testing._download("http://x/f", retries=3)
    assert len(attempts) == 3


def test_default_tune_cache_default_and_env(monkeypatch, tmp_path):
    monkeypatch.delenv("PYSID_TUNECACHE", raising=False)
    assert default_tune_cache(tmp_path) == tmp_path / ".tunecache"
    monkeypatch.setenv("PYSID_TUNECACHE", str(tmp_path / "c"))
    assert default_tune_cache(tmp_path) == tmp_path / "c"


def test_default_cache_kinds(monkeypatch, tmp_path):
    monkeypatch.setenv("PYSID_TUNECACHE", str(tmp_path / "c"))
    assert default_cache() == tmp_path / "c"
    assert [default_cache(k) for k in CACHE_KINDS] == [
        tmp_path / "c" / k for k in CACHE_KINDS
    ]
    with pytest.raises(ValueError, match="cache kind"):
        default_cache("hvsc")


def test_fetches_default_into_the_cache_root(monkeypatch, tmp_path):
    monkeypatch.setenv("PYSID_TUNECACHE", str(tmp_path))
    url = "https://x.example/rel.zip"
    _serve(monkeypatch, {url: b"one"})
    assert fetch_verified(url, _sha(b"one")) == tmp_path / "archives" / "rel.zip"


def test_host_slot_spaces_requests_to_a_throttled_host(monkeypatch, tmp_path):
    monkeypatch.setattr(testing.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(testing, "HOST_INTERVAL", {"slow.example": 60.0})
    sleeps = []
    monkeypatch.setattr(testing.time, "sleep", sleeps.append)
    with testing._host_slot("https://www.slow.example/a"):
        pass
    with testing._host_slot("https://slow.example/b"):
        pass
    with testing._host_slot("https://fast.example/c"):
        pass
    assert len(sleeps) == 2 and sleeps[1] > 59


def test_gather_tune_relpaths_from_ast(tmp_path):
    (tmp_path / "t_a.py").write_text(
        'TUNES = {"x": "DEMOS/x.sid"}\nOTHER = ["GAMES/y.sid", "notme.txt"]\n'
        'Z = "MUSICIANS/z.sid"\n',
        encoding="utf-8",
    )
    (tmp_path / "t_b.py").write_text(
        'S = "DEMOS/x.sid"  # dup\nF = f"{S}.sid"\nB = "bare.sid"\nT = "x/y.sid"\n',
        encoding="utf-8",
    )
    assert gather_tune_relpaths([tmp_path]) == [
        "DEMOS/x.sid",
        "GAMES/y.sid",
        "MUSICIANS/z.sid",
    ]


def test_gather_tune_relpaths_from_a_manifest(tmp_path):
    (tmp_path / "t.py").write_text('T = "DEMOS/x.sid"\n', encoding="utf-8")
    manifest = tmp_path / "tunes.txt"
    manifest.write_text(
        "GAMES/y.sid\n\n  /MUSICIANS/z.sid a/b.sid notes.txt\n", encoding="utf-8"
    )
    assert gather_tune_relpaths([manifest]) == ["/MUSICIANS/z.sid", "GAMES/y.sid"]
    assert gather_tune_relpaths([tmp_path]) == ["DEMOS/x.sid"]


def test_gather_tune_relpaths_from_a_pathdiff_manifest(tmp_path):
    cases = [
        {"id": "a", "relpath": "MUSICIANS/X/x.sid", "song": 1},
        {"id": "b", "relpath": "GAMES/y.sid", "reason": "C/not_me.sid differs"},
        {"id": "c", "relpath": "bare.sid"},
        "D/neither.sid",
    ]
    (tmp_path / "pathdiff.json").write_text(json.dumps(cases), encoding="utf-8")
    (tmp_path / "one.json").write_text('{"relpath": "DEMOS/z.sid"}', encoding="utf-8")
    want = ["DEMOS/z.sid", "GAMES/y.sid", "MUSICIANS/X/x.sid"]
    assert gather_tune_relpaths([tmp_path]) == want
    assert gather_tune_relpaths([tmp_path / "pathdiff.json"]) == want[1:]


def test_fetch_main_caches_reachable_reports_missing(monkeypatch, tmp_path, capsys):
    def fake(req, timeout=60):  # pylint: disable=unused-argument
        if "good.sid" in req.full_url:
            return _FakeResp(_SID)
        raise urllib.error.HTTPError("u", 404, "nf", {}, None)

    _patch_urlopen(monkeypatch, fake)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "t.py").write_text(
        'T = ["GAMES/good.sid", "GAMES/bad.sid"]\n', encoding="utf-8"
    )
    cache = tmp_path / "cache"
    assert fetch_main(["--cache", str(cache), "--tests", str(tests_dir)]) == 0
    assert (cache / "GAMES" / "good.sid").read_bytes() == _SID
    assert "GAMES/bad.sid" in capsys.readouterr().err


def test_fetch_tune_retries_an_incomplete_read(monkeypatch, tmp_path):
    replies = [http.client.IncompleteRead(b"PS"), _FakeResp(_SID)]

    def fake(*_a, **_k):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    _patch_urlopen(monkeypatch, fake)
    assert fetch_tune("A/x.sid", cache_dir=tmp_path).read_bytes() == _SID


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _serve(monkeypatch, blobs):
    """Serve ``{url: bytes}``; return the list of fetched URLs."""
    fetched = []

    def fake(req, timeout=60):  # pylint: disable=unused-argument
        fetched.append(req.full_url)
        return _FakeResp(blobs[req.full_url])

    _patch_urlopen(monkeypatch, fake)
    return fetched


def test_fetch_verified_caches_while_it_verifies(monkeypatch, tmp_path):
    url = "https://x.example/get.php/1/rel.tar.gz"
    fetched = _serve(monkeypatch, {url: b"one"})
    dest = fetch_verified(url, _sha(b"one"), cache_dir=tmp_path)
    assert dest == tmp_path / "rel.tar.gz" and dest.read_bytes() == b"one"
    assert fetch_verified(url, _sha(b"one"), cache_dir=tmp_path) == dest
    assert len(fetched) == 1
    dest.write_bytes(b"torn")
    assert fetch_verified(url, _sha(b"one"), cache_dir=tmp_path).read_bytes() == b"one"
    assert len(fetched) == 2
    with pytest.raises(ChecksumMismatch, match="SHA-256"):
        fetch_verified(url, _sha(b"two"), cache_dir=tmp_path, name="other")
    assert not (tmp_path / "other").exists()


def test_download_retries_an_empty_body(monkeypatch):
    replies = [b"", b"data"]
    _patch_urlopen(monkeypatch, lambda *a, **k: _FakeResp(replies.pop(0)))
    assert testing._download("http://x/f") == b"data"
    _patch_urlopen(monkeypatch, lambda *a, **k: _FakeResp(b""))
    with pytest.raises(TuneFetchError, match="an empty body"):
        testing._download("http://x/f", retries=2)


def test_fetch_verified_downloads_once_for_concurrent_callers(monkeypatch, tmp_path):
    url = "https://x.example/rel.zip"
    fetched = []

    def fake(req, timeout=60):  # pylint: disable=unused-argument
        fetched.append(req.full_url)
        time.sleep(0.05)
        return _FakeResp(b"one")

    _patch_urlopen(monkeypatch, fake)
    got = []
    workers = [
        threading.Thread(
            target=lambda: got.append(
                fetch_verified(url, _sha(b"one"), cache_dir=tmp_path)
            )
        )
        for _ in range(4)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert fetched == [url] and got == [tmp_path / "rel.zip"] * 4


def _zip(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _tgz(members, dirs=()):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name in dirs:
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            tf.addfile(info)
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@pytest.mark.parametrize("pack", [_zip, _tgz])
def test_fetch_archive_member_extracts_and_verifies(monkeypatch, tmp_path, pack):
    archive = pack({"rel/a/tune.swm": b"tune", "rel/b.swm": b"other"})
    url = "https://x.example/rel.pack"
    fetched = _serve(monkeypatch, {url: archive})
    args = (url, _sha(archive), "rel/a/tune.swm", _sha(b"tune"))
    dest = fetch_archive_member(*args, cache_dir=tmp_path)
    assert dest == tmp_path / "tune.swm" and dest.read_bytes() == b"tune"
    (tmp_path / "rel.pack").unlink()
    named = fetch_archive_member(*args, cache_dir=tmp_path, name="t-1.swm")
    assert named.read_bytes() == b"tune" and len(fetched) == 2
    assert fetch_archive_member(*args, cache_dir=tmp_path) == dest
    assert len(fetched) == 2
    fetch_archive_member(*args, cache_dir=tmp_path, name="t-2.swm", archive_name="r.z")
    assert (tmp_path / "r.z").read_bytes() == archive and len(fetched) == 3
    with pytest.raises(TuneFetchError, match="rel/b.swm: SHA-256"):
        fetch_archive_member(
            url, _sha(archive), "rel/b.swm", _sha(b"x"), cache_dir=tmp_path
        )
    with pytest.raises(KeyError):
        fetch_archive_member(
            url, _sha(archive), "rel/none", _sha(b""), cache_dir=tmp_path
        )


def test_fetch_archive_member_rejects_a_directory(monkeypatch, tmp_path):
    archive = _tgz({}, dirs=("rel/sub",))
    _serve(monkeypatch, {"https://x.example/r.tgz": archive})
    with pytest.raises(TuneFetchError, match="not a regular file"):
        fetch_archive_member(
            "https://x.example/r.tgz", _sha(archive), "rel/sub", "", cache_dir=tmp_path
        )


def _prg(value):
    # LDA #value; STA $D400; RTS (play at the load address)
    return bytes([0x00, 0x10, 0xA9, value, 0x8D, 0x00, 0xD4, 0x60])


def _as_grid(data, nframes, song):
    return [[data[3]] * 25 for _ in range(nframes)]


def test_decode_fidelity_exact_and_differs():
    same = testing.decode_fidelity(_prg(5), _prg(5), render=_as_grid, frames=4)
    assert same.ok
    diff = testing.decode_fidelity(_prg(5), _prg(6), render=_as_grid, frames=4)
    assert not diff.ok and diff.first == 0


def test_assert_decode_fidelity():
    testing.assert_decode_fidelity(_prg(5), lambda d: d, render=_as_grid, frames=4)
    with pytest.raises(AssertionError, match="replays differently"):
        testing.assert_decode_fidelity(
            _prg(5), lambda d: _prg(6), render=_as_grid, frames=4, name="t"
        )
