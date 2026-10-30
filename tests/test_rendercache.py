"""Tests for the content-addressed render cache (pysidtracker.rendercache)."""

import numpy as np

from pysidtracker import rendercache
from pysidtracker.rendercache import HIT, MISS, CachedRender, RenderCache

from .test_snapshot import TOY, _ToyPlayer


class _Counting(_ToyPlayer):
    renders = 0

    def render_grid(self, nframes):
        type(self).renders += 1
        return super().render_grid(nframes)


def _model(body=b"TOY1"):
    return ("toy", 0, body)


def test_grid_is_rendered_once_and_serves_shorter_requests(tmp_path):
    fmt = TOY.__class__(**{**TOY.__dict__, "player": _Counting})
    cache = RenderCache(tmp_path)
    _Counting.renders = 0
    long = cache.grid(fmt, _model(), 1, 8)
    assert cache.last == MISS and long.dtype == np.int16 and len(long) == 8
    assert (cache.grid(fmt, _model(), 1, 5) == long[:5]).all()
    assert cache.last == HIT and _Counting.renders == 1
    assert len(cache.grid(fmt, _model(), 1, 12)) == 12
    assert cache.last == MISS and _Counting.renders == 2
    cache.grid(fmt, _model(b"TOY2"), 1, 8)
    cache.grid(fmt, _model(), 2, 8)
    assert _Counting.renders == 4


def test_player_code_change_invalidates(tmp_path, monkeypatch):
    cache = RenderCache(tmp_path)
    cache.grid(TOY, _model(), 1, 4)
    monkeypatch.setattr(rendercache, "player_version", lambda fmt: "changed")
    cache.grid(TOY, _model(), 1, 4)
    assert cache.last == MISS


def test_player_version_follows_player_modules(tmp_path, monkeypatch):
    mod = tmp_path / "tmp_player_mod.py"
    mod.write_text("X = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    fmt = TOY.__class__(**{**TOY.__dict__, "player_modules": ("tmp_player_mod",)})
    before = rendercache.player_version(fmt)
    assert before != rendercache.player_version(TOY)
    mod.write_text("X = 2\n")
    rendercache._modules_digest.cache_clear()  # pylint: disable=protected-access
    assert rendercache.player_version(fmt) != before


def test_corrupt_entry_is_a_miss(tmp_path):
    cache = RenderCache(tmp_path)
    cache.grid(TOY, _model(), 1, 4)
    for path in tmp_path.glob("*.npz"):
        path.write_bytes(b"junk")
    cache.grid(TOY, _model(), 1, 4)
    assert cache.last == MISS


def test_cached_render_is_picklable(tmp_path):
    import pickle  # pylint: disable=import-outside-toplevel

    assert pickle.loads(pickle.dumps(CachedRender("toy", tmp_path))).name == "toy"
