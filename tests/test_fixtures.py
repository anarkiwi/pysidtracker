"""Tests for the CSDB native-format fixture fetch / resolve / manifest helpers.

The default (hermetic) run never touches the network: ``fixtures._http_get`` is
monkeypatched. The scraper, manifest round-trip, skip-on-missing, and cache-skip
behaviours are all exercised against the fake HTTP layer.
"""

import hashlib
import io
import urllib.error
import zipfile

import pytest

from pysidtracker import fixtures, testing
from pysidtracker.fixtures import (
    DEFAULT_CSDB,
    SUMS_NAME,
    USER_AGENT,
    CsdbFetchError,
    SumEntry,
    NativeFixture,
    _http_get,
    default_native_cache,
    fetch_csdb,
    fixture,
    index_media,
    load_manifest,
    main,
    make_native_fixtures,
    manifest_from_cache,
    read_sums,
    write_manifest,
)

from .helpers import build_disk

_GOAT = b"\x00GOAT-blitz\x01"
_DMC = b"\x00DMC-drive\x02"
_FILES = {101: _GOAT, 102: _DMC}
_FINAL_URL = {
    101: "https://csdb.dk/files/101/Blitz%20Demo.sng",
    102: "https://csdb.dk/files/102/Drive.sng",
}
_RELEASES = {"777": (101, 102), "999": ()}  # 999: reachable page with no links


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _page_html(dids):
    links = "".join(f'<a href="/release/download.php?id={d}">x</a>' for d in dids)
    return f"<html>{links}</html>"


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("PYSID_TUNECACHE", raising=False)
    monkeypatch.setattr(testing, "HOST_INTERVAL", {})


@pytest.fixture
def http(monkeypatch):
    """Route release pages and file downloads from the in-memory tables."""
    calls = []

    def fake(url, **_kw):
        calls.append(url)
        for rid, dids in _RELEASES.items():
            if url == f"{DEFAULT_CSDB}/release/?id={rid}":
                return url, _page_html(dids).encode("latin-1")
        for did, data in _FILES.items():
            if url == f"{DEFAULT_CSDB}/release/download.php?id={did}":
                return _FINAL_URL[did], data
        raise CsdbFetchError(f"{url}: not found")

    monkeypatch.setattr(fixtures, "_http_get", fake)
    return calls


class _FakeResp:
    def __init__(self, data, final):
        self._data, self._final = data, final

    def read(self):
        return self._data

    def geturl(self):
        return self._final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.strain_exempt(reason="exercises code outside the measured package")
def test_constants():
    assert DEFAULT_CSDB == "https://csdb.dk"
    assert USER_AGENT.startswith("pysidtracker/")
    assert SUMS_NAME == "SHA256SUMS.txt"


def test_default_native_cache_nests_under_the_tune_cache(monkeypatch, tmp_path):
    assert default_native_cache(tmp_path) == tmp_path / ".tunecache" / "native"
    monkeypatch.setenv("PYSID_TUNECACHE", str(tmp_path / "tc"))
    assert default_native_cache(tmp_path) == tmp_path / "tc" / "native"


@pytest.fixture
def _no_sleep(monkeypatch):
    monkeypatch.setattr(testing.time, "sleep", lambda *_: None)


def test_http_get_is_throttled_per_host(monkeypatch, tmp_path):
    monkeypatch.setattr(testing.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(testing, "HOST_INTERVAL", {"csdb.dk": 60.0})
    sleeps = []
    monkeypatch.setattr(testing.time, "sleep", sleeps.append)
    monkeypatch.setattr(
        testing.urllib.request, "urlopen", lambda *a, **k: _FakeResp(b"b", "u")
    )
    _http_get("https://csdb.dk/x", retries=1)
    _http_get("https://csdb.dk/y", retries=1)
    assert sleeps[-1] > 59


def test_http_get_retries_an_empty_body(_no_sleep, monkeypatch):
    replies = [b"", b"b"]
    monkeypatch.setattr(
        testing.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResp(replies.pop(0), "u"),
    )
    assert _http_get("https://csdb.dk/x") == ("u", b"b")


def test_http_get_retries_then_fails(monkeypatch):
    attempts = []

    def fake(*_a, **_k):
        attempts.append(1)
        raise urllib.error.URLError("down")

    monkeypatch.setattr(testing.urllib.request, "urlopen", fake)
    monkeypatch.setattr(testing.time, "sleep", lambda *_: None)
    with pytest.raises(CsdbFetchError, match="unreachable"):
        _http_get("https://csdb.dk/x", retries=3)
    assert len(attempts) == 3


@pytest.mark.usefixtures("http")
def test_fetch_csdb_tolerates_a_dead_release_page(tmp_path, capsys):
    # 888 is unreachable and 777 is not: the batch must not abort.
    entries = fetch_csdb(["888", "777"], tmp_path)
    assert {e.relpath for e in entries} == {
        "dl/777/101_Blitz Demo.sng",
        "dl/777/102_Drive.sng",
    }
    assert "888" in capsys.readouterr().err  # the dead release is reported


def test_fetch_csdb_resumes_from_cache(http, tmp_path):
    fetch_csdb(["777"], tmp_path)
    downloads = [u for u in http if "download.php" in u]
    assert len(downloads) == 2
    # Second run: pages and files already cached -> no new downloads at all.
    entries = fetch_csdb(["777"], tmp_path)
    assert len(entries) == 2
    assert len([u for u in http if "download.php" in u]) == 2


@pytest.mark.usefixtures("http")
def test_fetch_csdb_reported_missing_file_not_fatal(tmp_path, capsys):
    # Did 102 is a genuine 404: the fetcher must skip it, not abort the release.
    del _FILES[102]
    try:
        entries = fetch_csdb(["777"], tmp_path)
    finally:
        _FILES[102] = _DMC
    rels = {e.relpath for e in entries}
    assert "dl/777/101_Blitz Demo.sng" in rels
    assert not any(r.startswith("dl/777/102_") for r in rels)
    assert "102" in capsys.readouterr().err  # the failure is reported


def test_read_sums_parses_and_skips_bad_lines(tmp_path):
    (tmp_path / SUMS_NAME).write_text(
        f"{_sha(_GOAT)}  dl/777/101_Blitz Demo.sng  https://csdb.dk/x\n"
        "\n"
        "# a comment\n"
        "not-a-valid-line\n",
        encoding="utf-8",
    )
    rows = read_sums(tmp_path)
    assert rows == [
        SumEntry(_sha(_GOAT), "dl/777/101_Blitz Demo.sng", "https://csdb.dk/x")
    ]
    assert read_sums(tmp_path / "nowhere") == []


@pytest.mark.usefixtures("http")
def test_fixture_none_when_absent_or_corrupt(tmp_path):
    fetch_csdb(["777"], tmp_path)
    assert fixture("0" * 64, tmp_path) is None
    # A cache entry whose content no longer matches reads as absent.
    (tmp_path / "dl" / "777" / "101_Blitz Demo.sng").write_bytes(b"corrupt")
    assert fixture(_sha(_GOAT), tmp_path) is None


def test_load_manifest_rejects_short_lines(tmp_path):
    p = tmp_path / "m.txt"
    p.write_text("only\ttwo_fields\n", encoding="utf-8")
    with pytest.raises(CsdbFetchError, match="need release_id, name, sha256"):
        load_manifest(p)


def test_make_native_fixtures_returns_two():
    native_id, native_file = make_native_fixtures({"x": "0" * 64}, "/tmp/cache")
    assert native_id is not None and native_file is not None


@pytest.mark.strain_exempt(reason="exercises code outside the measured package")
def test_make_native_fixtures_resolves_when_warm(pytester, tmp_path):
    cache = tmp_path / "cache"
    dl = cache / "dl" / "777"
    dl.mkdir(parents=True)
    (dl / "101_Blitz.sng").write_bytes(_GOAT)
    (cache / "SHA256SUMS.txt").write_text(
        f"{_sha(_GOAT)}  dl/777/101_Blitz.sng  https://csdb.dk/x\n", encoding="utf-8"
    )
    pytester.makepyfile(f"""
        from pysidtracker import make_native_fixtures
        native_id, native_file = make_native_fixtures({{"x": "{_sha(_GOAT)}"}}, {str(cache)!r})

        def test_it(native_id, native_file):
            assert native_id == "x"
            assert native_file.read_bytes() == {_GOAT!r}
        """)
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)


@pytest.mark.strain_exempt(reason="exercises code outside the measured package")
def test_make_native_fixtures_skips_when_cold(pytester, tmp_path):
    pytester.makepyfile(f"""
        from pysidtracker import make_native_fixtures
        native_id, native_file = make_native_fixtures({{"x": "{_sha(_GOAT)}"}}, {str(tmp_path / "nope")!r})

        def test_it(native_file):
            assert False, "should have skipped"
        """)
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")
    result.assert_outcomes(skipped=1)


@pytest.mark.usefixtures("http")
def test_cli_fetch_fails_when_expected_missing(tmp_path, capsys):
    m = tmp_path / "m.txt"
    write_manifest([NativeFixture("777", "Nope.sng", "f" * 64, "")], m)
    assert main(["fetch", str(m), "--cache", str(tmp_path)]) == 1
    assert "missing" in capsys.readouterr().err


@pytest.mark.usefixtures("http")
def test_cli_verify_and_manifest(tmp_path, capsys):
    fetch_csdb(["777"], tmp_path)
    m = tmp_path / "out.txt"
    main(["manifest", "--cache", str(tmp_path), "--out", str(m)])
    assert main(["verify", str(m), "--cache", str(tmp_path)]) == 0
    assert "2/2 expected fixture(s)" in capsys.readouterr().out


_ON_DISK = b"\x00\x43" + bytes(range(256)) * 3


def _disk_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("rel/Song Disk.d64", build_disk([("S.SONG", _ON_DISK, 0x82)]))
        zf.writestr("rel/readme.txt", b"not media")
        zf.writestr("rel/bad.d64", b"short")
    return buf.getvalue()


@pytest.fixture
def disk_release(http):  # pylint: disable=redefined-outer-name,unused-argument
    """Release 777's file 102 is a zip holding a disk with the wanted PRG."""
    saved = _FILES[102], _FINAL_URL[102]
    _FILES[102], _FINAL_URL[102] = _disk_zip(), "https://csdb.dk/files/102/rel.zip"
    yield
    _FILES[102], _FINAL_URL[102] = saved


@pytest.mark.usefixtures("disk_release")
def test_index_media_pins_a_prg_inside_a_release_disk(tmp_path):
    downloads = fetch_csdb(["777"], tmp_path)
    found = index_media(downloads, {_sha(_ON_DISK), "f" * 64}, tmp_path)
    rel = "media/777/102_rel.zip/rel/Song_Disk.d64/S.SONG"
    url = f"{DEFAULT_CSDB}/release/download.php?id=102"
    assert found == [SumEntry(_sha(_ON_DISK), rel, url)]
    assert fixture(_sha(_ON_DISK), tmp_path).read_bytes() == _ON_DISK
    assert index_media(downloads, set(), tmp_path) == []
    index_media(downloads, {_sha(_ON_DISK)}, tmp_path)
    assert (tmp_path / SUMS_NAME).read_text().count(rel) == 1
    by_sha = {r.sha256: r for r in manifest_from_cache(tmp_path)}
    assert by_sha[_sha(_ON_DISK)][:2] == ("777", "S.SONG")
