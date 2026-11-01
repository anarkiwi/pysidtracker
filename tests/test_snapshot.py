"""Tests for the corpus snapshot / diff tool (pysidtracker.snapshot)."""

import pytest

from pysidtracker import BaseSidParser, SidFormat, SidImage, SidParseError
from pysidtracker.maincli import main
from pysidtracker.player import MemPlayer
from pysidtracker.rendercache import HIT, MISS, RenderCache
from pysidtracker.testing import check_outcome, corpus_params
from pysidtracker.snapshot import (
    expected_errors,
    fingerprint,
    load_spec,
    parse_probes,
    resolve_format,
)

SPEC = "tests.test_snapshot:TOY"


class _ToyParser(BaseSidParser):
    def parse(self, data, **kwargs):
        img = SidImage.from_bytes(data)
        if img.image[:3] == b"BAD":
            raise SidParseError("unsupported toy")
        return ("toy", kwargs.get("subtune"), img.image)

    def recognize(self, image):
        return image.image[:3] == b"TOY"


def rebuild(model):
    """The toy's rebuilt image."""
    return model[2]


def explode(model):
    """A probe that always fails."""
    raise ValueError(f"no export for {model[0]}")


class _ToyPlayer(MemPlayer):
    """Writes the model's last byte, then counts frames, to ``$D400``."""

    def __init__(self, model):
        self.last = model[2][-1]
        super().__init__(b"", 0x1000)

    def _init(self, subtune):
        self.cnt = 0

    def _frame(self):
        self._wr(0xD400, self.last + self.cnt)
        self.cnt += 1


TOY = SidFormat("toy", _ToyParser(), _ToyPlayer, probes={"rebuild": rebuild})


def toy_factory():
    """A zero-argument factory returning the toy format."""
    return TOY


def _tunes(tmp_path, **bodies):
    paths = []
    for name, body in bodies.items():
        path = tmp_path / f"{name}.prg"
        path.write_bytes(b"\x00\x10" + body)
        paths.append((path.name, path))
    return paths


def test_load_spec_and_parse_probes():
    assert load_spec(SPEC) is TOY
    assert load_spec("tests.test_snapshot:TOY.name") == "toy"
    assert parse_probes(["x=tests.test_snapshot:explode"]) == {"x": explode}
    with pytest.raises(ValueError):
        load_spec("tests.test_snapshot")
    with pytest.raises(ValueError):
        parse_probes(["tests.test_snapshot:explode"])


def test_resolve_format_by_spec_factory_or_name():
    assert resolve_format(b"\x00\x10TOY", SPEC) is TOY
    assert resolve_format(b"\x00\x10TOY", "tests.test_snapshot:toy_factory") is TOY
    with pytest.raises(SidParseError):
        resolve_format(b"\x00\x10TOY", "no-such-format")


def test_cli_build_and_diff(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _tunes(corpus, a=b"TOY1", c=b"BAD")
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    argv = ["snapshot", "build", str(corpus), "--format", SPEC, "--jobs", "1"]
    assert main([*argv, "-o", str(before)]) == 0
    assert "2 tunes, 1 with errors" in capsys.readouterr().out
    assert main(["snapshot", "diff", str(before), str(before)]) == 0
    assert "0 of 2 tunes differ" in capsys.readouterr().out
    (corpus / "a.prg").write_bytes(b"\x00\x10TOY9")
    assert main([*argv, "-o", str(after)]) == 0
    with pytest.raises(SystemExit) as exc:
        main(["snapshot", "diff", str(before), str(after)])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "a.prg  model:" in out and "a.prg  rebuild:" in out
    assert "1 of 2 tunes differ" in out


def test_corpus_cases_from_a_snapshot(tmp_path):
    tunes = {"A/good.sid": b"\x00\x10TOY1", "B/bad.sid": b"\x00\x10BAD"}
    snap = {rel: fingerprint(data, TOY) for rel, data in tunes.items()}
    expected = expected_errors(snap)
    assert expected == {"A/good.sid": None, "B/bad.sid": "unsupported toy"}
    params = corpus_params(tunes, expected)
    assert [(p.id, p.values) for p in params] == [
        ("A/good.sid", ("A/good.sid", None)),
        ("B/bad.sid", ("B/bad.sid", "unsupported toy")),
    ]
    for rel, data in tunes.items():
        (tmp_path / rel.replace("/", "_")).write_bytes(data)

    def load(rel):
        return TOY.load((tmp_path / rel.replace("/", "_")).read_bytes())

    assert check_outcome(load, "A/good.sid", None) == ("toy", 0, b"TOY1")
    assert check_outcome(load, "B/bad.sid", "unsupported toy", SidParseError) is None
    with pytest.raises(AssertionError, match="not 'other'"):
        check_outcome(load, "B/bad.sid", "other")
    with pytest.raises(AssertionError, match="decoded, expected"):
        check_outcome(load, "A/good.sid", "unsupported toy")
    with pytest.raises(SidParseError):
        check_outcome(load, "B/bad.sid", None)


def test_snapshot_reuses_cached_renders(songlengths, tmp_path):
    data = b"\x00\x10TOY1"
    songlengths.add(data, 0.06)
    cache = RenderCache(tmp_path / "render")
    first = fingerprint(data, TOY, cache=cache)
    assert cache.last == MISS and first == fingerprint(data, TOY)
    assert fingerprint(data, TOY, cache=cache) == first and cache.last == HIT
    changed = fingerprint(b"\x00\x10TOY2", TOY, cache=cache)
    assert cache.last == MISS and changed["render"] != first["render"]
