"""The ``hvsc`` / ``sidtrace_oracle`` / ``disk_prgs`` fixtures ship via pytest11.

Exercised end to end with ``pytester``, so a dependent suite gets the retriever
fixtures with no conftest of its own -- just by installing pysidtracker.
"""

from .test_d64 import _build_d64

_SID = b"PSID" + b"\x00" * 60


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


def test_disk_prgs_fixture_reads_d64(pytester, tmp_path):
    d64 = tmp_path / "game.d64"
    d64.write_bytes(_build_d64([("TUNE", b"\x00\x20abc")]))
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
