"""Tests for the shared CLI scaffold."""

import argparse

from pysidtracker import (
    add_reglog_command,
    add_wav_command,
    print_info,
    run_cli,
)


class _Err(Exception):
    pass


def _build_parser():
    parser = argparse.ArgumentParser(prog="demo")
    subs = parser.add_subparsers(dest="command", required=True)
    add_reglog_command(subs, _reglog_ok)
    add_wav_command(subs, _wav_fail)
    return parser


def _reglog_ok(args):
    assert args.output == "out.log"
    assert args.seconds == 30.0


def _wav_fail(args):  # pylint: disable=unused-argument
    raise _Err("kaboom")


def test_run_cli_success():
    assert (
        run_cli(_build_parser, _Err, ["reglog", "t.sid", "out.log", "--seconds", "30"])
        == 0
    )


def test_run_cli_error_class():
    assert run_cli(_build_parser, _Err, ["wav", "t.sid", "o.wav"]) == 1


def test_run_cli_oserror(capsys):
    def _boom(_args):
        raise OSError("disk full")

    def _build():
        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers(dest="c", required=True)
        p = subs.add_parser("x")
        p.set_defaults(func=_boom)
        return parser

    assert run_cli(_build, _Err, ["x"]) == 1
    assert "error: disk full" in capsys.readouterr().err


def test_wav_command_defaults():
    parser = _build_parser()
    args = parser.parse_args(["wav", "t.sid", "o.wav"])
    assert args.model == "8580"
    assert args.seconds == 60.0


def test_print_info(capsys):
    print_info("N", "A", "R", 0x1000, 0x1003, 0x1006)
    out = capsys.readouterr().out
    assert "name:     N" in out
    assert "load:     $1000" in out
    assert "play:     $1006" in out
