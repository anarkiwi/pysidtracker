"""Tests for the generic ``pysidtracker`` CLI and its format registry."""

import pytest

from pysidtracker import (
    SID_BASE,
    BaseSidParser,
    FormatCommand,
    MemPlayer,
    SidFormat,
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


def test_format_specific_subcommand(capsys, tune):
    _run([_fake_format()], ["echo", str(tune), "--tag", "hi"])
    assert "echo fake hi" in capsys.readouterr().out


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
    # init latches CIA #1 Timer-A ($DC04/$DC05) to 0x1234, so the play cadence is
    # the timer period (latch + 1), not the nominal PAL video frame.
    init = bytes(
        [0xA9, 0x34, 0x8D, 0x04, 0xDC, 0xA9, 0x12, 0x8D, 0x05, 0xDC, 0x60]
    )  # LDA #$34 STA $DC04 / LDA #$12 STA $DC05 / RTS
    data = build_psid(init, load=0x1000, init=0x1000, play=0x1000)
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
