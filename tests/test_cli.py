"""Tests for the shared CLI scaffold."""

import argparse

from pysidtracker import (
    add_reglog_command,
    add_wav_command,
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


def test_run_cli_error_class():
    assert run_cli(_build_parser, _Err, ["wav", "t.sid", "o.wav"]) == 1


def test_wav_command_defaults():
    parser = _build_parser()
    args = parser.parse_args(["wav", "t.sid", "o.wav"])
    assert args.model == "8580"
    assert args.seconds == 60.0
