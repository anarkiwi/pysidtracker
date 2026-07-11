"""Tests for the HVSC fetch/resolve test helper."""

import urllib.error

import pytest

from pysidtracker import testing
from pysidtracker.testing import (
    DEFAULT_MIRROR,
    TuneFetchError,
    default_tune_cache,
    fetch_tune,
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


def test_gather_tune_relpaths_from_ast(tmp_path):
    (tmp_path / "t_a.py").write_text(
        'TUNES = {"x": "A/x.sid"}\nOTHER = ["B/y.sid", "notme.txt"]\nZ = "C/z.sid"\n',
        encoding="utf-8",
    )
    (tmp_path / "t_b.py").write_text('S = "A/x.sid"  # dup\n', encoding="utf-8")
    assert gather_tune_relpaths([tmp_path]) == ["A/x.sid", "B/y.sid", "C/z.sid"]


def test_fetch_main_caches_reachable_reports_missing(monkeypatch, tmp_path, capsys):
    def fake(req, timeout=60):  # pylint: disable=unused-argument
        if "good.sid" in req.full_url:
            return _FakeResp(_SID)
        raise urllib.error.HTTPError("u", 404, "nf", {}, None)

    _patch_urlopen(monkeypatch, fake)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "t.py").write_text(
        'T = ["A/good.sid", "A/bad.sid"]\n', encoding="utf-8"
    )
    cache = tmp_path / "cache"
    assert fetch_main(["--cache", str(cache), "--tests", str(tests_dir)]) == 0
    assert (cache / "A" / "good.sid").read_bytes() == _SID
    assert "A/bad.sid" in capsys.readouterr().err
