import pytest

"""The ``hvsc`` / ``sidtrace_oracle`` / ``disk_prgs`` / ``c64_roms`` pytest11 fixtures.

Exercised end to end with ``pytester``, so a dependent suite gets the retriever
fixtures with no conftest of its own -- just by installing pysidtracker.
"""

from .helpers import build_disk

_SID = b"PSID" + b"\x00" * 60


@pytest.mark.strain_exempt(reason="exercises code outside the measured package")
def test_hvsc_fixture_resolves_local(pytester, tmp_path):
    hvsc = tmp_path / "hvsc"
    (hvsc / "A").mkdir(parents=True)
    (hvsc / "A" / "x.sid").write_bytes(_SID)
    pytester.makepyfile(f"""
        import os
        os.environ["HVSC"] = {str(hvsc)!r}
        os.environ["PYSID_TUNECACHE"] = {str(tmp_path / "cache")!r}

        def test_it(hvsc):
            path = hvsc("A/x.sid")
            assert path.exists()
            assert hvsc.read("A/x.sid")[:4] == b"PSID"
        """)
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)


@pytest.mark.strain_exempt(reason="exercises code outside the measured package")
def test_hvsc_fixture_raises_when_unreachable(pytester, tmp_path):
    pytester.makepyfile(f"""
        import os, pytest, urllib.request, urllib.error
        os.environ.pop("HVSC", None)
        os.environ["PYSID_TUNECACHE"] = {str(tmp_path / "cache")!r}
        import pysidtracker.testing as t
        t.time.sleep = lambda *_: None
        urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(
            urllib.error.URLError("offline")
        )
        from pysidtracker.testing import TuneFetchError

        def test_it(hvsc):
            with pytest.raises(TuneFetchError):
                hvsc("A/missing.sid")
        """)
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)


@pytest.mark.strain_exempt(reason="exercises code outside the measured package")
def test_disk_prgs_fixture_reads_d64(pytester, tmp_path):
    d64 = tmp_path / "game.d64"
    d64.write_bytes(build_disk([("TUNE", b"\x00\x20abc", 0x82)]))
    pytester.makepyfile(f"""
        import os
        os.environ["PYSID_TUNECACHE"] = {str(tmp_path / "cache")!r}
        import pysidtracker.testing as t
        t._download = lambda url, **k: open({str(d64)!r}, "rb").read()

        def test_it(disk_prgs):
            files = disk_prgs("http://example/game.d64")
            assert [f.name for f in files] == ["TUNE"]
            assert files[0].prg == b"\\x00\\x20abc"
        """)
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)


@pytest.mark.strain_exempt(reason="exercises code outside the measured package")
def test_hvsc_local_envs_ini_option(pytester, tmp_path):
    tree = tmp_path / "tree"
    (tree / "A").mkdir(parents=True)
    (tree / "A" / "x.sid").write_bytes(_SID)
    pytester.makeini("""
        [pytest]
        hvsc_local_envs = MY_HVSC
        """)
    pytester.makepyfile(f"""
        import os
        os.environ["MY_HVSC"] = {str(tree)!r}
        os.environ["PYSID_TUNECACHE"] = {str(tmp_path / "cache")!r}

        def test_it(hvsc):
            assert hvsc("A/x.sid").exists()
        """)
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)


@pytest.mark.strain_exempt(reason="exercises code outside the measured package")
def test_c64_roms_fixture_resolves_local(pytester, tmp_path):
    local = tmp_path / "vice"
    local.mkdir()
    (local / "kernal-901227-03.bin").write_bytes(b"\xea" * 0x2000)
    pytester.makepyfile(f"""
        import hashlib, os
        os.environ["C64_ROMS"] = {str(local)!r}
        os.environ["PYSID_TUNECACHE"] = {str(tmp_path / "cache")!r}
        from pysidtracker.roms import ROMS
        ROMS["kernal"] = ROMS["kernal"]._replace(
            sha256=hashlib.sha256(b"\\xea" * 0x2000).hexdigest())

        def test_it(c64_roms):
            assert c64_roms.available(("kernal",))
            assert c64_roms("kernal").read_bytes() == b"\\xea" * 0x2000
            assert bytes(c64_roms.image(("kernal",))[0xE000:]) == b"\\xea" * 0x2000
        """)
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)


@pytest.mark.strain_exempt(reason="exercises code outside the measured package")
def test_c64_roms_fixture_skips_when_unavailable(pytester, tmp_path):
    pytester.makepyfile(f"""
        import os, pytest
        os.environ.pop("C64_ROMS", None)
        os.environ["PYSID_TUNECACHE"] = {str(tmp_path / "cache")!r}
        import vagrant_moses.roms as r
        r._download = lambda url, **k: (_ for _ in ()).throw(
            r.RomFetchError("offline"))

        def test_it(c64_roms):
            if not c64_roms.available():
                pytest.skip("C64 ROMs unavailable (offline, not cached)")
            assert False, "should have skipped"
        """)
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")
    result.assert_outcomes(skipped=1)


@pytest.mark.strain_exempt(reason="exercises code outside the measured package")
def test_native_fixture_resolves_when_warm(pytester, tmp_path):
    import hashlib

    data = b"\x00GOAT\x01"
    cache = tmp_path / "cache"
    native_dir = cache / "native"
    (native_dir / "dl" / "777").mkdir(parents=True)
    (native_dir / "dl" / "777" / "101_Blitz.sng").write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    (native_dir / "SHA256SUMS.txt").write_text(
        f"{sha}  dl/777/101_Blitz.sng  https://csdb.dk/x\n", encoding="utf-8"
    )
    pytester.makepyfile(f"""
        import os
        os.environ["PYSID_TUNECACHE"] = {str(cache)!r}

        def test_it(native):
            path = native("{sha}")
            assert path is not None and path.read_bytes() == b"\\x00GOAT\\x01"
            assert native.available("{sha}") is True
        """)
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)


@pytest.mark.strain_exempt(reason="exercises code outside the measured package")
def test_native_fixture_none_when_cold(pytester, tmp_path):
    pytester.makepyfile(f"""
        import os
        os.environ["PYSID_TUNECACHE"] = {str(tmp_path / "cache")!r}

        def test_it(native):
            assert native("f" * 64) is None
            assert native.available("f" * 64) is False
        """)
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)
