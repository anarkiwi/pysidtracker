"""Dead-symbol and unresolved-doc-reference lint (:mod:`pysidtracker.lint`)."""

import textwrap

import pytest

from pysidtracker.errors import SidError
from pysidtracker.lint import check_project, run_lint, add_lint_arguments


def _project(tmp_path, **files):
    """Write ``files`` (``/`` spelled ``_S_``, ``.`` spelled ``_D_``) under tmp_path."""
    for rel, text in files.items():
        path = tmp_path / rel.replace("_S_", "/").replace("_D_", ".")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))
    return str(tmp_path / "pkg")


def _check(tmp_path, **kw):
    return check_project(
        str(tmp_path / "pkg"),
        tests=[str(tmp_path / "tests")],
        docs=[str(tmp_path / "docs")],
        **kw,
    )


def test_unresolved_references(tmp_path):
    _project(
        tmp_path,
        pkg_S___init___D_py="",
        pkg_S_core_D_py='''
            """Mentions :func:`_gone_helper` and ``pkg.core.Missing``."""
            import numpy

            def f():
                """Uses :class:`numpy.ndarray`, :class:`int`, ``numpy.zeros``."""
                return f
            ''',
        tests_S_test_x_D_py="from pkg.core import f\n\ndef test_a():\n    f()\n",
        docs_S_b_D_md="See :class:`pkg.Player` and ``pkg.core.f``.\n",
    )
    out = _check(tmp_path)
    assert any("_gone_helper" in p for p in out)
    assert any("pkg.core.Missing" in p for p in out)
    assert any(
        "pkg.Player" in p and "b.md" in p.replace("b_md", "b.md") or "b_md" in p
        for p in out
    )
    assert not any("ndarray" in p or "'int'" in p or "zeros" in p for p in out)
    assert not any("pkg.core.f" in p for p in out)


def test_run_lint_raises_on_problems(tmp_path, capsys):
    pkg = _project(tmp_path, pkg_S___init___D_py="UNUSED = 1\n")
    import argparse

    parser = argparse.ArgumentParser()
    add_lint_arguments(parser)
    args = parser.parse_args([pkg, "--tests", "none", "--docs", "none"])
    with pytest.raises(SidError):
        run_lint(args)
    assert "UNUSED" in capsys.readouterr().out


# Public API kept for downstream format packages (nothing in-repo reads them).
_PUBLIC = [
    "notefreq",
    "FIDELITY_FIELDS",
    "SID_FILTER_LO",
    "SID_FILTER_HI",
    "SID_RES_FILT",
    # Exercised only inside pytester sources (tests/test_testing_fixtures.py).
    "make_tune_fixtures",
]


def test_own_package_is_lint_clean():
    assert check_project("src/pysidtracker", whitelist=_PUBLIC) == []


def test_truncated_comparisons(tmp_path):
    _project(
        tmp_path,
        pkg_S___init___D_py="",
        tests_S_consts_D_py="FRAMES = 250\nSPAN = FRAMES * 2\n",
        tests_S_test_x_D_py="""
            from pysidtracker.testing import song_frames, whole_song
            from consts import FRAMES

            N = 100
            M = song_frames(b"")

            def test_a(player, data, n):
                player.render_grid(250)
                player.render_grid(N + 1)
                register_grid(data, FRAMES)
                oracle_grid(data, frames=SPAN)
                oracle_grid(data, seconds=-2.5)
                survey.oracle_rows(data, "o.csv", 10, 1)
                list(player.iter_frames(max_frames=N))
                assert_matches(data, player, None, frames=8)
                player.render_grid(whole_song(8))
                player.render_grid(M)
                player.render_grid(n)
                register_grid(data, len(data))
                survey.oracle_rows(data, "o.csv", M, 1, calls=M)
                other(250)
                for k in range(3):
                    player.render_grid(k)
            """,
    )
    lines = [p.split(":")[1] for p in _check(tmp_path)]
    assert lines == ["9", "10", "11", "12", "13", "14", "15", "16"]
    assert all("not the whole song" in p for p in _check(tmp_path))
