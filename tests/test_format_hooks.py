"""Tests for the SidFormat grammar / canonical hooks and the grammar / equiv commands."""

import dataclasses

import pytest

from pysidtracker import BaseSidParser, FormatCommand, SidFormat, SidImage
from pysidtracker import maincli
from pysidtracker.formats import CANONICAL
from pysidtracker.snapshot import digest, fingerprint

from .test_bytegrammar import TOY, Seq


class _SeqParser(BaseSidParser):
    """``SEQ`` then one toy ``seq`` stream; the subtune is kept on the model."""

    def recognize(self, image):
        return image.image[:3] == b"SEQ"

    def parse(self, data, **kwargs):
        seq = TOY.parse_prefix(SidImage.from_bytes(data).image[3:], "seq").model
        return seq, kwargs.get("subtune")


def canonical(model) -> str:
    """Items with id 0 are no-ops, so only the others play."""
    seq, _ = model
    return "\n".join(f"{i:02x} {v:02x}" for i, v in seq.items if i)


SEQ = SidFormat(
    "seq",
    _SeqParser(),
    None,
    grammar=TOY,
    streams=lambda model: [("voice 0", model[0], "seq")],
    canonical=canonical,
)
BARE = SidFormat("bare", _SeqParser(), None)


def _tune(tmp_path, name, *items):
    path = tmp_path / f"{name}.prg"
    path.write_bytes(b"\x00\x10SEQ" + Seq(items).to_bytes())
    return str(path)


def _run(formats, argv):
    args = maincli.build_parser(formats).parse_args(argv)
    args.func(args)


def test_grammar_and_equiv_need_their_hooks(tmp_path, capsys):
    song = _tune(tmp_path, "a", (1, 2))
    assert (
        maincli.run_cli(
            lambda: maincli.build_parser([BARE]), Exception, ["grammar", song]
        )
        == 1
    )
    assert "format 'bare' has no byte grammar" in capsys.readouterr().err
    argv = ["equiv", song, song]
    assert maincli.run_cli(lambda: maincli.build_parser([BARE]), Exception, argv) == 1
    assert "format 'bare' has no canonical text" in capsys.readouterr().err


def test_format_by_name_and_spec(tmp_path, capsys):
    song = _tune(tmp_path, "a", (1, 2))
    _run([BARE, SEQ], ["grammar", song, "--format", "seq"])
    assert capsys.readouterr().out.startswith("voice 0:")
    _run([], ["grammar", song, "--format", "tests.test_format_hooks:SEQ"])
    assert capsys.readouterr().out.startswith("voice 0:")


def test_equiv_is_silent_when_tunes_play_the_same(tmp_path, capsys):
    a = _tune(tmp_path, "a", (1, 2), (0, 5), (3, 4))
    b = _tune(tmp_path, "b", (1, 2), (3, 4))
    _run([SEQ], ["equiv", a, b])
    assert capsys.readouterr().out == ""


def test_equiv_diffs_and_exits_1(tmp_path, capsys):
    a = _tune(tmp_path, "a", (1, 2), (3, 4))
    b = _tune(tmp_path, "b", (1, 2), (3, 5))
    with pytest.raises(SystemExit) as exc:
        _run([SEQ], ["equiv", a, b])
    assert exc.value.code == 1
    out = capsys.readouterr().out.splitlines()
    assert out[:2] == [f"--- {a}", f"+++ {b}"]
    assert out[3:] == [" 01 02", "-03 04", "+03 05"]


def test_shared_commands_win_over_contributed_ones(tmp_path, capsys):
    own = FormatCommand("grammar", "own", lambda p: p.add_argument("song"), print)
    fmt = dataclasses.replace(SEQ, commands=(own,))
    _run([fmt], ["grammar", _tune(tmp_path, "a", (1, 2))])
    assert capsys.readouterr().out.startswith("voice 0:")


def test_snapshot_probes_canonical_text():
    data = b"\x00\x10SEQ" + Seq(((1, 2), (0, 9))).to_bytes()
    prints = fingerprint(data, SEQ)
    assert prints[CANONICAL] == digest(b"01 02")
    own = dataclasses.replace(SEQ, probes={CANONICAL: lambda model: b"own"})
    assert fingerprint(data, own)[CANONICAL] == digest(b"own")
    assert CANONICAL not in fingerprint(data, BARE)
