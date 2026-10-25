"""Tests for the generic ``pysidtracker`` CLI and its format registry."""

import dataclasses
import hashlib

import numpy as np
import pytest

from pysidtracker import (
    SID_BASE,
    BaseSidParser,
    FormatCommand,
    MemPlayer,
    SidFormat,
    SidError,
    SidParseError,
)
from pysidtracker import formats as formats_mod
from pysidtracker import maincli

from .helpers import build_psid

# init at $1000: LDA #$0A / STA $D400 / RTS.
_SIMPLE = bytes([0xA9, 0x0A, 0x8D, 0x00, 0xD4, 0x60])


class _FakeModel:
    name = "Tune"
    author = "Author"
    released = "1988"
    load_addr = 0x1000
    init_addr = 0x1000
    play_addr = 0x1000


class _FakeParser(BaseSidParser):
    """Recognises the ``_SIMPLE`` image by its leading opcode."""

    def recognize(self, image):
        return image.mem[0x1000] == 0xA9

    def parse(self, data, **kwargs):
        return _FakeModel()


class _FakePlayer(MemPlayer):
    def _init(self, subtune):
        pass

    def _frame(self):
        self._wr(SID_BASE, 0x0A)


def _echo_args(subparser):
    subparser.add_argument("song")
    subparser.add_argument("--tag", default="x")


def _echo_handler(args, fmt):
    print(f"echo {fmt.name} {args.tag}")


def _fake_format():
    return SidFormat(
        name="fake",
        parser=_FakeParser(),
        player=lambda model: _FakePlayer(b"\x00", 0x1000),
        describe=lambda model: [f"detail:  {model.name}!"],
        commands=(FormatCommand("echo", "echo a tag", _echo_args, _echo_handler),),
    )


@pytest.fixture
def tune(tmp_path):
    path = tmp_path / "tune.sid"
    path.write_bytes(build_psid(_SIMPLE, load=0x1000, init=0x1000, play=0x1000))
    return path


def _run(formats, argv):
    parser = maincli.build_parser(formats)
    args = parser.parse_args(argv)
    args.func(args)


def test_info_uses_metadata_and_describe(capsys, tune):
    _run([_fake_format()], ["info", str(tune)])
    out = capsys.readouterr().out
    assert "name:     Tune" in out
    assert "format:   fake" in out
    assert "detail:  Tune!" in out


def test_reglog_writes_log(tmp_path, tune):
    out = tmp_path / "out.reglog"
    _run([_fake_format()], ["reglog", str(tune), str(out), "--seconds", "0.1"])
    assert out.exists() and out.read_text().strip()


def test_wav_renders(monkeypatch, tmp_path, tune):
    written = {}
    monkeypatch.setattr(
        maincli,
        "render_player_wav",
        lambda p, dst, **kw: written.update(dst=dst, kw=kw),
    )
    _run(
        [_fake_format()], ["wav", str(tune), str(tmp_path / "o.wav"), "--seconds", "1"]
    )
    assert written["dst"].endswith("o.wav")
    assert written["kw"]["cycles_per_frame"] > 0  # framed at the tune's clock
    assert written["kw"]["clock_frequency"] > 0


def test_format_specific_subcommand(capsys, tune):
    _run([_fake_format()], ["echo", str(tune), "--tag", "hi"])
    assert "echo fake hi" in capsys.readouterr().out


class _OtherParser(BaseSidParser):
    """Recognises a tune whose init starts with ``LDX``."""

    def recognize(self, image):
        return image.mem[0x1000] == 0xA2

    def parse(self, data, **kwargs):
        return _FakeModel()


def _other_args(subparser):
    subparser.add_argument("song")
    subparser.add_argument("output")
    subparser.add_argument("--level", type=int, choices=(1, 2), default=1)


def _other_format():
    return SidFormat(
        name="other",
        parser=_OtherParser(),
        player=lambda model: _FakePlayer(b"\x00", 0x1000),
        commands=(
            FormatCommand(
                "echo",
                "echo a level",
                _other_args,
                lambda args, fmt: print(f"echo {fmt.name} {args.output} {args.level}"),
            ),
        ),
    )


def _shared_cli(argv):
    formats = [_fake_format(), _other_format()]
    return maincli.run_cli(lambda: maincli.build_parser(formats), SidError, argv)


def test_shared_subcommand_runs_the_recognised_formats_command(capsys, tmp_path, tune):
    other = tmp_path / "other.sid"
    other.write_bytes(
        build_psid(b"\xa2\x00\x60", load=0x1000, init=0x1000, play=0x1000)
    )
    assert _shared_cli(["echo", str(tune), "--tag", "hi"]) == 0
    assert _shared_cli(["echo", str(other), "out.bin", "--level", "2"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "echo fake hi",
        "echo other out.bin 2",
    ]


def test_shared_subcommand_parses_with_the_owners_arguments(capsys, tune):
    with pytest.raises(SystemExit):
        _shared_cli(["echo", str(tune), "--level", "2"])
    assert "unrecognized arguments: --level" in capsys.readouterr().err


def test_shared_subcommand_help_lists_each_owner(capsys):
    with pytest.raises(SystemExit):
        _shared_cli(["echo", "--help"])
    out = capsys.readouterr().out
    assert "fake: usage: echo [-h] [--tag TAG] song" in out
    assert "other: usage: echo [-h] [--level {1,2}] song output" in out


def test_shared_subcommand_rejects_an_unowned_tune(capsys, tmp_path):
    rts = tmp_path / "rts.sid"
    rts.write_bytes(build_psid(b"\x60", load=0x1000, init=0x1000, play=0x1000))
    assert _shared_cli(["echo", str(rts)]) == 1
    assert "format with 'echo' recognises" in capsys.readouterr().err


def test_recognize_format_selects_and_rejects(tune):
    formats = [_fake_format()]
    data = tune.read_bytes()
    assert formats_mod.recognize_format(formats, data).name == "fake"
    # A parser whose recognizer raises is skipped, not fatal.
    boom = SidFormat(
        name="boom",
        parser=type(
            "P",
            (BaseSidParser,),
            {"recognize": lambda s, i: 1 / 0, "parse": lambda s, d: None},
        )(),
        player=lambda m: None,
    )
    assert formats_mod.recognize_format([boom] + formats, data).name == "fake"


def test_unrecognized_tune_exits_one(capsys, tune):
    rc = maincli.run_cli(
        lambda: maincli.build_parser([]), maincli.SidError, ["info", str(tune)]
    )
    assert rc == 1
    assert "no installed pysidtracker format recognises" in capsys.readouterr().err


def test_discover_formats(monkeypatch):
    fmt = _fake_format()

    class _EP:
        def load(self):
            return fmt

    monkeypatch.setattr(formats_mod, "entry_points", lambda group: [_EP()])
    assert formats_mod.discover_formats() == [fmt]


def test_discover_formats_accepts_callable(monkeypatch):
    fmt = _fake_format()

    class _EP:
        def load(self):
            return lambda: fmt

    monkeypatch.setattr(formats_mod, "entry_points", lambda group: [_EP()])
    assert formats_mod.discover_formats() == [fmt]


def test_cadence_uses_cia_timer_latch():
    # A CIA-speed PSID whose init latches CIA #1 Timer-A to 0x1234 plays every
    # latch + 1 cycles, not every PAL video frame.
    init = bytes(
        [0xA9, 0x34, 0x8D, 0x04, 0xDC, 0xA9, 0x12, 0x8D, 0x05, 0xDC]
        + [0xEE, 0x00, 0xD4, 0x60]
    )  # LDA #$34 STA $DC04 / LDA #$12 STA $DC05 / INC $D400 / RTS
    data = build_psid(init, load=0x1000, init=0x1000, play=0x1000, speed=1)
    cycles_per_frame, _clock = maincli._cadence(data)
    assert cycles_per_frame == 0x1234 + 1


def test_cadence_defaults_to_pal_video_frame():
    from pysidtracker import PAL_CLOCK_HZ, PAL_CYCLES_PER_FRAME

    data = build_psid(_SIMPLE, load=0x1000, init=0x1000, play=0x1000)
    cycles_per_frame, clock_hz = maincli._cadence(data)
    assert cycles_per_frame == PAL_CYCLES_PER_FRAME
    assert clock_hz == float(PAL_CLOCK_HZ)


def test_main_discovers_and_runs(monkeypatch, capsys, tune):
    monkeypatch.setattr(maincli, "discover_formats", lambda: [_fake_format()])
    rc = maincli.main(["info", str(tune)])
    assert rc == 0
    assert "format:   fake" in capsys.readouterr().out


def test_editor_io_cli_lists_the_transfer_call_sites(tmp_path, capsys):
    code = (
        b"\xa9\x01\xa2\x00\x20\xd5\xff"  # LDA #$01 / LDX #$00 / JSR $FFD5  -> LOAD @ $1004
        b"\xa9\x02\x20\xd8\xff"  # LDA #$02 / JSR $FFD8  -> SAVE @ $1009
        b"\xa9\x03\x20\xc0\xff"  # LDA #$03 / JSR $FFC0  -> OPEN @ $100E
    )
    path = tmp_path / "editor.prg"
    path.write_bytes(bytes([0x00, 0x10]) + code)
    _run([], ["editor-io", str(path)])
    out = capsys.readouterr().out
    assert "LOAD  $1004" in out and "SAVE  $1009" in out and "OPEN  $100E" in out
    assert "JSR $ffd5" in out  # the call site's disassembled row


def test_fixtures_subcommand_is_wired_into_the_cli(tmp_path):
    data = b"\x00GOAT\x01"
    sha = hashlib.sha256(data).hexdigest()
    native = tmp_path / "native"
    (native / "dl" / "777").mkdir(parents=True)
    (native / "dl" / "777" / "101_Blitz.sng").write_bytes(data)
    (native / "SHA256SUMS.txt").write_text(
        f"{sha}  dl/777/101_Blitz.sng  https://csdb.dk/x\n", encoding="utf-8"
    )
    out = tmp_path / "m.txt"
    parser = maincli.build_parser([])
    args = parser.parse_args(
        ["fixtures", "manifest", "--cache", str(native), "--out", str(out)]
    )
    assert args.func(args) == 0
    assert out.exists() and sha in out.read_text()


class _SubtuneParser(_FakeParser):
    def parse(self, data, subtune=0, **kwargs):
        return subtune


class _GridPlayer:
    def __init__(self, subtune):
        self.subtune = subtune

    def render_grid(self, nframes):
        return [[self.subtune] * 25] * nframes


def _subtune_format():
    return SidFormat(name="sub", parser=_SubtuneParser(), player=_GridPlayer)


def test_start_subtune_reads_the_header():
    psid = build_psid(_SIMPLE, load=0x1000, init=0x1000, play=0x1000, songs=3)
    assert formats_mod.start_subtune(psid) == 0
    psid = build_psid(_SIMPLE, load=0x1000, init=0x1000, songs=3, start_song=3)
    assert formats_mod.start_subtune(psid) == 2
    assert formats_mod.start_subtune(bytes([0x00, 0x10]) + _SIMPLE) == 0


def test_load_and_play_select_the_subtune():
    fmt = _subtune_format()
    psid = build_psid(_SIMPLE, load=0x1000, init=0x1000, songs=3, start_song=2)
    assert fmt.load(psid) == 1 and fmt.load(psid, 2) == 2 and fmt.load(psid, 0) == 0
    assert fmt.play(psid).subtune == 1 and fmt.play(psid, 2).subtune == 2


def test_find_format_by_name_or_recognition(tune):
    data = tune.read_bytes()
    formats = [_fake_format(), _subtune_format()]
    assert formats_mod.find_format(data, "sub", formats).name == "sub"
    assert formats_mod.find_format(data, "auto", formats).name == "fake"
    for name in ("none", "auto"):
        with pytest.raises(SidParseError, match="no installed pysidtracker format"):
            formats_mod.find_format(data, name, [] if name == "auto" else formats)


def test_render_plays_the_one_based_song(monkeypatch):
    monkeypatch.setattr(formats_mod, "discover_formats", lambda: [_subtune_format()])
    psid = build_psid(_SIMPLE, load=0x1000, init=0x1000, songs=3)
    grid, info = formats_mod.render("auto", psid, 3, 4)
    assert info["format"] == "sub" and {"rsid", "trigger", "speed", "dynamic"} <= set(
        info
    )
    assert grid.dtype == np.int16
    assert grid.shape == (4, 25) and (grid == 2).all()


def test_render_gates_a_named_format_on_recognition(monkeypatch):
    fmt = dataclasses.replace(_subtune_format(), survey=lambda *_: {"extra": 1})
    monkeypatch.setattr(formats_mod, "discover_formats", lambda: [fmt])
    psid = build_psid(_SIMPLE, load=0x1000, init=0x1000, songs=3)
    assert formats_mod.render("sub", psid, 1, 2)[1]["extra"] == 1
    monkeypatch.setattr(_SubtuneParser, "recognize", lambda self, image: False)
    with pytest.raises(SidParseError, match="does not recognise"):
        formats_mod.render("sub", psid, 1, 2)
