"""Tests for the content-addressed render cache (pysidtracker.rendercache)."""

import sys


from pysidtracker import rendercache
from pysidtracker.rendercache import CachedRender

from .test_snapshot import TOY, _ToyPlayer


class _Counting(_ToyPlayer):
    renders = 0

    def render_grid(self, nframes):
        type(self).renders += 1
        return super().render_grid(nframes)


def _model(body=b"TOY1"):
    return ("toy", 0, body)


def test_player_version_follows_package_data_files(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "tmp_data_pkg", raising=False)
    rendercache._modules_digest.cache_clear()  # pylint: disable=protected-access
    pkg = tmp_path / "tmp_data_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    spec = pkg / "tracker.play"
    spec.write_text("one")
    monkeypatch.syspath_prepend(str(tmp_path))
    fmt = TOY.__class__(
        **{**TOY.__dict__, "player_modules": ("tmp_data_pkg:tracker.play",)}
    )
    before = rendercache.player_version(fmt)
    assert before != rendercache.player_version(TOY)
    spec.write_text("two")
    rendercache._modules_digest.cache_clear()  # pylint: disable=protected-access
    assert rendercache.player_version(fmt) != before


def test_cached_render_is_picklable(tmp_path):
    import pickle  # pylint: disable=import-outside-toplevel

    assert pickle.loads(pickle.dumps(CachedRender("toy", tmp_path))).name == "toy"
