"""Tests for the CSDB native-format fixture fetch / resolve / manifest helpers.

The default (hermetic) run never touches the network: ``fixtures._http_get`` is
monkeypatched. The scraper, manifest round-trip, skip-on-missing, and cache-skip
behaviours are all exercised against the fake HTTP layer.
"""

import hashlib
import urllib.error

import pytest

from pysidtracker import fixtures
from pysidtracker.fixtures import (
    DEFAULT_CSDB,
    RATE_LIMIT,
    SUMS_NAME,
    USER_AGENT,
    CsdbFetchError,
    SumEntry,
    NativeFixture,
    _file_name,
    _http_get,
    _scrape_release,
    default_native_cache,
    fetch_csdb,
    fixture,
    load_manifest,
    main,
    make_native_fixtures,
    manifest_from_cache,
    read_sums,
    write_manifest,
)

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
    monkeypatch.delenv("PYSID_NATIVECACHE", raising=False)
    monkeypatch.delenv("PYSID_TUNECACHE", raising=False)


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


def test_constants():
    assert DEFAULT_CSDB == "https://csdb.dk"
    assert USER_AGENT.startswith("pysidtracker/")
    assert RATE_LIMIT > 0 and 0 < RATE_LIMIT < 5
    assert SUMS_NAME == "SHA256SUMS.txt"


def test_default_native_cache_default_and_env(monkeypatch, tmp_path):
    monkeypatch.delenv("PYSID_TUNECACHE", raising=False)
    assert default_native_cache(tmp_path) == tmp_path / ".tunecache" / "native"
    monkeypatch.setenv("PYSID_NATIVECACHE", str(tmp_path / "nc"))
    assert default_native_cache(tmp_path) == tmp_path / "nc"
    # It nests under the shared tune cache, like the C64 ROMs.
    monkeypatch.delenv("PYSID_NATIVECACHE", raising=False)
    monkeypatch.setenv("PYSID_TUNECACHE", str(tmp_path / "tc"))
    assert default_native_cache(tmp_path) == tmp_path / "tc" / "native"


@pytest.fixture
def _no_sleep(monkeypatch):
    monkeypatch.setattr(fixtures.time, "sleep", lambda *_: None)


def test_http_get_success_returns_final_url_and_bytes(monkeypatch):
    monkeypatch.setattr(
        fixtures.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResp(b"blob", "https://csdb.dk/final.bin"),
    )
    url, data = _http_get("https://csdb.dk/release/download.php?id=1", rate_limit=0)
    assert data == b"blob" and url == "https://csdb.dk/final.bin"


def test_http_get_sends_user_agent(monkeypatch):
    seen = {}

    def fake(req, timeout=60):  # pylint: disable=unused-argument
        seen["ua"] = req.get_header("User-agent")
        return _FakeResp(b"b", req.full_url)

    monkeypatch.setattr(fixtures.urllib.request, "urlopen", fake)
    _http_get("https://csdb.dk/x", rate_limit=0)
    assert seen["ua"] == USER_AGENT


def test_http_get_is_polite(_no_sleep, monkeypatch):
    sleeps = []
    monkeypatch.setattr(fixtures.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(
        fixtures.urllib.request, "urlopen", lambda *a, **k: _FakeResp(b"b", "u")
    )
    _http_get("https://csdb.dk/x", retries=1)
    assert RATE_LIMIT in sleeps


def test_http_get_404_raises(monkeypatch):
    def fake(*_a, **_k):
        raise urllib.error.HTTPError("u", 404, "nf", {}, None)

    monkeypatch.setattr(fixtures.urllib.request, "urlopen", fake)
    with pytest.raises(CsdbFetchError, match="not found"):
        _http_get("https://csdb.dk/x", retries=1, rate_limit=0)


def test_http_get_retries_then_fails(monkeypatch):
    attempts = []

    def fake(*_a, **_k):
        attempts.append(1)
        raise urllib.error.URLError("down")

    monkeypatch.setattr(fixtures.urllib.request, "urlopen", fake)
    monkeypatch.setattr(fixtures.time, "sleep", lambda *_: None)
    with pytest.raises(CsdbFetchError, match="unreachable"):
        _http_get("https://csdb.dk/x", retries=3, rate_limit=0)
    assert len(attempts) == 3


def test_scrape_release_extracts_and_caches_pages(http, tmp_path):
    dids = _scrape_release("777", tmp_path, mirror=DEFAULT_CSDB, retries=4, force=False)
    assert dids == [101, 102]
    assert (tmp_path / "html" / "rel_777.html").exists()
    before = len(http)
    # Second call is served from the cached page: no new page request.
    assert _scrape_release(
        "777", tmp_path, mirror=DEFAULT_CSDB, retries=4, force=False
    ) == [101, 102]
    assert len(http) == before


def test_scrape_release_force_refetches(http, tmp_path):
    _scrape_release("777", tmp_path, mirror=DEFAULT_CSDB, retries=4, force=False)
    before = len(http)
    _scrape_release("777", tmp_path, mirror=DEFAULT_CSDB, retries=4, force=True)
    assert len(http) == before + 1  # one more page request


@pytest.mark.usefixtures("http")
def test_scrape_release_empty_and_unreachable(tmp_path):
    # A reachable page with no download links -> []. An unreachable page raises.
    assert (
        _scrape_release("999", tmp_path, mirror=DEFAULT_CSDB, retries=4, force=True)
        == []
    )
    with pytest.raises(CsdbFetchError, match="not found"):
        _scrape_release("888", tmp_path, mirror=DEFAULT_CSDB, retries=4, force=True)


@pytest.mark.usefixtures("http")
def test_fetch_csdb_tolerates_a_dead_release_page(tmp_path, capsys):
    # 888 is unreachable and 777 is not: the batch must not abort.
    entries = fetch_csdb(["888", "777"], tmp_path)
    assert {e.relpath for e in entries} == {
        "dl/777/101_Blitz Demo.sng",
        "dl/777/102_Drive.sng",
    }
    assert "888" in capsys.readouterr().err  # the dead release is reported


def test_file_name_from_final_url():
    assert (
        _file_name("https://csdb.dk/files/101/Blitz%20Demo.sng", "101")
        == "Blitz Demo.sng"
    )
    assert _file_name("https://csdb.dk/", "42") == "42"


@pytest.mark.usefixtures("http")
def test_fetch_csdb_downloads_and_writes_sums(tmp_path):
    entries = fetch_csdb(["777"], tmp_path)
    # Two files, named <did>_<basename>, under dl/777.
    assert {(e.relpath) for e in entries} == {
        "dl/777/101_Blitz Demo.sng",
        "dl/777/102_Drive.sng",
    }
    assert (tmp_path / "dl" / "777" / "101_Blitz Demo.sng").read_bytes() == _GOAT
    assert (tmp_path / "dl" / "777" / "102_Drive.sng").read_bytes() == _DMC
    sums = (tmp_path / SUMS_NAME).read_text()
    assert (
        f"{_sha(_GOAT)}  dl/777/101_Blitz Demo.sng  {DEFAULT_CSDB}/release/download.php?id=101"
        in sums
    )
    assert (
        f"{_sha(_DMC)}  dl/777/102_Drive.sng  {DEFAULT_CSDB}/release/download.php?id=102"
        in sums
    )
    assert len(entries) == 2


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
def test_fixture_resolves_verified_file(tmp_path):
    fetch_csdb(["777"], tmp_path)
    p = fixture(_sha(_GOAT), tmp_path)
    assert p == tmp_path / "dl" / "777" / "101_Blitz Demo.sng"
    # Case-insensitive + tolerant of surrounding whitespace.
    assert (
        fixture(_sha(_DMC).upper(), tmp_path)
        == tmp_path / "dl" / "777" / "102_Drive.sng"
    )
    assert fixture("  " + _sha(_GOAT) + "  ", tmp_path) is not None


@pytest.mark.usefixtures("http")
def test_fixture_none_when_absent_or_corrupt(tmp_path):
    fetch_csdb(["777"], tmp_path)
    assert fixture("0" * 64, tmp_path) is None
    # A cache entry whose content no longer matches reads as absent.
    (tmp_path / "dl" / "777" / "101_Blitz Demo.sng").write_bytes(b"corrupt")
    assert fixture(_sha(_GOAT), tmp_path) is None


def test_manifest_round_trip(tmp_path):
    entries = [
        NativeFixture("777", "Blitz.sng", _sha(_GOAT), "GoatTracker tune 'Blitz'"),
        NativeFixture("777", "Drive.sng", _sha(_DMC), "DMCSID module 'Drive'"),
    ]
    path = write_manifest(entries, tmp_path / "native_fixtures.txt")
    assert load_manifest(path) == entries


def test_load_manifest_ignores_comments_and_blanks(tmp_path):
    p = tmp_path / "m.txt"
    p.write_text(
        f"# comment\n\n777\tBlitz.sng\t{_sha(_GOAT)}\tGoatTracker tune\n",
        encoding="utf-8",
    )
    got = load_manifest(p)
    assert got == [NativeFixture("777", "Blitz.sng", _sha(_GOAT), "GoatTracker tune")]


def test_load_manifest_rejects_short_lines(tmp_path):
    p = tmp_path / "m.txt"
    p.write_text("only\ttwo_fields\n", encoding="utf-8")
    with pytest.raises(CsdbFetchError, match="need release_id, name, sha256"):
        load_manifest(p)


@pytest.mark.usefixtures("http")
def test_manifest_from_cache_recovers_id_name_and_desc(tmp_path):
    fetch_csdb(["777"], tmp_path)
    descs = {_sha(_GOAT): "GoatTracker tune 'Blitz'"}
    rows = manifest_from_cache(tmp_path, descs)
    by_name = {r.name: r for r in rows}
    assert by_name["Blitz Demo.sng"].release_id == "777"
    assert by_name["Blitz Demo.sng"].description == "GoatTracker tune 'Blitz'"
    assert [r.name for r in rows] == sorted([r.name for r in rows])


def test_make_native_fixtures_returns_two():
    native_id, native_file = make_native_fixtures({"x": "0" * 64}, "/tmp/cache")
    assert native_id is not None and native_file is not None


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
def test_cli_fetch_recreates_and_verifies(tmp_path):
    m = tmp_path / "m.txt"
    write_manifest([NativeFixture("777", "Blitz Demo.sng", _sha(_GOAT), "")], m)
    assert main(["fetch", str(m), "--cache", str(tmp_path)]) == 0
    assert (tmp_path / "dl" / "777" / "101_Blitz Demo.sng").exists()


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
