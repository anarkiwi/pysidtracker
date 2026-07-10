"""Shared argparse scaffold for the ``py*`` command-line tools.

Each leaf CLI dispatches ``args.func(args)`` and wraps its own parse error plus
``OSError`` into an ``error: ...``/exit-1 message; :func:`run_cli` owns that
loop, and :func:`add_reglog_command`/:func:`add_wav_command`/:func:`print_info`
establish the shared ``song``/``--seconds``/``--model`` arguments and metadata
printout every tool repeats.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Optional, Sequence

from .audio import CHIP_MODELS

Handler = Callable[[argparse.Namespace], object]
ParserFactory = Callable[[], argparse.ArgumentParser]


def run_cli(
    build_parser: ParserFactory,
    error_class: type,
    argv: Optional[Sequence[str]] = None,
) -> int:
    """Parse ``argv``, dispatch ``args.func(args)``, return a process exit code.

    ``build_parser`` returns the configured :class:`argparse.ArgumentParser`.
    An ``error_class`` (the tool's parse error) or ``OSError`` raised by the
    handler is printed as ``error: ...`` and mapped to exit code 1.
    """
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (error_class, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def add_reglog_command(subparsers, handler: Handler, *, song_help: str = "input tune"):
    """Add the shared ``reglog`` subcommand (``song output --seconds``)."""
    parser = subparsers.add_parser("reglog", help="write a SID register log")
    parser.add_argument("song", help=song_help)
    parser.add_argument("output", help="register log file to write")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.set_defaults(func=handler)
    return parser


def add_wav_command(subparsers, handler: Handler, *, song_help: str = "input tune"):
    """Add the shared ``wav`` subcommand (``song output --seconds --model``)."""
    parser = subparsers.add_parser("wav", help="render through an emulated SID")
    parser.add_argument("song", help=song_help)
    parser.add_argument("output", help="WAV file to write")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--model", choices=CHIP_MODELS, default="8580")
    parser.set_defaults(func=handler)
    return parser


def print_info(
    name: str,
    author: str,
    released: str,
    load: int,
    init: int,
    play: int,
    *,
    file=None,
) -> None:
    """Print the shared tune-metadata block."""
    out = file if file is not None else sys.stdout
    print(f"name:     {name}", file=out)
    print(f"author:   {author}", file=out)
    print(f"released: {released}", file=out)
    print(f"load:     ${load:04X}", file=out)
    print(f"init:     ${init:04X}", file=out)
    print(f"play:     ${play:04X}", file=out)
