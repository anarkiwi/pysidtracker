"""Exercise the make_tune_fixtures pytest fixtures end to end via pytester."""


def test_make_tune_fixtures_resolves_local(pytester, tmp_path):
    hvsc = tmp_path / "hvsc"
    tune = hvsc / "A" / "x.sid"
    tune.parent.mkdir(parents=True)
    tune.write_bytes(b"PSID" + b"\x00" * 32)
    cache = hvsc / "cache"

    pytester.makepyfile(f"""
        import os
        from pysidtracker import make_tune_fixtures

        os.environ["HVSC"] = {str(hvsc)!r}
        tune_id, tune_path = make_tune_fixtures({{"x": "A/x.sid"}}, {str(cache)!r})

        def test_resolved(tune_id, tune_path):
            assert tune_id == "x"
            assert tune_path.exists()
        """)
    result = pytester.runpytest_inprocess("-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)


def test_make_tune_fixtures_skips_when_unavailable(pytester, tmp_path):
    cache = tmp_path / "cache"
    pytester.makepyfile(f"""
        import os
        from pysidtracker import make_tune_fixtures

        os.environ.pop("HVSC", None)
        os.environ["HVSC_MIRROR"] = "http://127.0.0.1:9/nothing"
        tune_id, tune_path = make_tune_fixtures(
            {{"x": "A/x.sid"}}, {str(cache)!r}
        )

        def test_skipped(tune_path):
            assert False, "should have skipped"
        """)
    result = pytester.runpytest_inprocess("-p", "no:cacheprovider")
    result.assert_outcomes(skipped=1)
