"""Tests for the corpus snapshot / diff tool (pysidtracker.snapshot)."""

import json

import pytest

from pysidtracker import BaseSidParser, SidFormat, SidImage, SidParseError
from pysidtracker.maincli import main
from pysidtracker.testing import check_outcome, corpus_params
from pysidtracker.snapshot import (
    MISSING,
    diff,
    digest,
    expected_errors,
    fingerprint,
    load_spec,
    parse_probes,
    read_snapshot,
    resolve_format,
    snapshot,
    write_snapshot,
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


TOY = SidFormat("toy", _ToyParser(), None, probes={"rebuild": rebuild})


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


def test_digest_is_content_addressed():
    assert digest(b"ab") == digest(bytearray(b"ab")) != digest(b"ac")
    assert digest(("x", 1)) == digest(("x", 1)) != digest(("x", 2))
    assert len(digest(b"")) == 32


def test_fingerprint_records_model_probes_and_errors():
    data = b"\x00\x10TOY1"
    out = fingerprint(data, TOY, {"export": explode})
    assert out["model"] == digest(("toy", 0, b"TOY1"))
    assert out["rebuild"] == digest(b"TOY1")
    assert out["export"] == "!ValueError: no export for toy"
    assert fingerprint(b"\x00\x10BAD", TOY) == {
        "model": "!SidParseError: unsupported toy"
    }


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


def test_snapshot_in_workers(tmp_path):
    tunes = _tunes(tmp_path, b=b"TOY2", a=b"TOY1", c=b"BAD")
    snap = snapshot(
        tunes, SPEC, ["export=tests.test_snapshot:explode"], workers=2, progress=False
    )
    assert list(snap) == ["a.prg", "b.prg", "c.prg"]
    assert snap["a.prg"] == fingerprint(b"\x00\x10TOY1", TOY, {"export": explode})
    assert snap["c.prg"]["model"].startswith("!SidParseError")
    unknown = snapshot(tunes[:1], "no-such-format", workers=1, progress=False)
    assert unknown["b.prg"]["model"].startswith("!SidParseError")


def test_diff_reports_changed_added_and_removed():
    old = {"a": {"model": "1", "rebuild": "2"}, "b": {"model": "3"}}
    new = {"a": {"model": "1", "rebuild": "9", "export": "4"}, "c": {"model": "5"}}
    assert diff(old, new) == [
        ("a", "export", MISSING, "4"),
        ("a", "rebuild", "2", "9"),
        ("b", "model", "3", MISSING),
        ("c", "model", MISSING, "5"),
    ]
    assert diff(old, old) == []


def test_write_and_read_snapshot(tmp_path):
    snap = {"a": {"model": "1"}}
    path = write_snapshot(snap, tmp_path / "sub" / "s.json")
    assert read_snapshot(path) == snap
    assert json.loads(path.read_text()) == snap


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
