"""The generic ``pysidtracker`` command-line tool.

Discovers every installed SID format (:mod:`pysidtracker.formats`) and offers
the shared ``info`` / ``reglog`` / ``wav`` commands -- plus any format-specific
subcommands each format contributes -- so a dependent package registers a
:class:`~pysidtracker.formats.SidFormat` instead of shipping its own CLI.

Format selection is by content: each command loads the tune, picks the first
registered format whose parser recognises it, and runs against that format's
model and player.
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence, Tuple

from .audio import CHIP_MODELS, render_player_wav, seconds_to_frames
from .cadence import playroutine_cadence
from .cli import print_info, run_cli
from .errors import SidError, SidParseError
from .formats import SidFormat, discover_formats, recognize_format
from .image import SidImage
from .registers import PAL_CLOCK_HZ, PAL_CYCLES_PER_FRAME
from .reglog import register_writes_from_player, write_reglog
from .source import read_bytes


def _load(song, formats: Sequence[SidFormat]) -> Tuple[SidFormat, bytes, object]:
    """Recognise ``song`` among ``formats`` and return ``(format, data, model)``."""
    data = read_bytes(song)
    fmt = recognize_format(formats, data)
    if fmt is None:
        raise SidParseError(f"no installed pysidtracker format recognises {song!r}")
    return fmt, data, fmt.parser.parse(data)


def _cadence(data: bytes) -> Tuple[int, float]:
    """The tune's play cadence (cycles per frame) and CPU clock (Hz).

    Derived from what the tune's init actually programs
    (:func:`~pysidtracker.cadence.playroutine_cadence`): a CIA-timer latch period
    for a timer-driven or multi-speed tune, else its PAL/NTSC video frame -- not
    merely the header's advertised rate, so a timed tune is framed at its real
    period rather than a nominal 50 Hz. A bare ``.prg`` (no init to trace)
    defaults to the PAL frame."""
    if data[:4] in (b"PSID", b"RSID"):
        image = SidImage.from_bytes(data)
        if image.header is not None:
            cadence = playroutine_cadence(image)
            return cadence.cycles_per_call, float(cadence.clock_hz)
    return PAL_CYCLES_PER_FRAME, float(PAL_CLOCK_HZ)


def _info(args, formats: Sequence[SidFormat]) -> None:
    fmt, _data, model = _load(args.song, formats)
    print_info(*fmt.model_metadata(model))
    print(f"format:   {fmt.name}")
    if fmt.describe is not None:
        for line in fmt.describe(model):
            print(line)


def _reglog(args, formats: Sequence[SidFormat]) -> None:
    fmt, data, model = _load(args.song, formats)
    cycles_per_frame, clock_hz = _cadence(data)
    writes = register_writes_from_player(
        fmt.player(model),
        max_frames=seconds_to_frames(args.seconds, cycles_per_frame, clock_hz),
        cycles_per_frame=cycles_per_frame,
    )
    write_reglog(writes, args.output)
    print(f"wrote {args.output}")


def _wav(args, formats: Sequence[SidFormat]) -> None:
    fmt, data, model = _load(args.song, formats)
    cycles_per_frame, _clock_hz = _cadence(data)
    render_player_wav(
        fmt.player(model),
        args.output,
        seconds=args.seconds,
        model=args.model,
        cycles_per_frame=cycles_per_frame,
    )
    print(f"wrote {args.output}")


def build_parser(formats: Sequence[SidFormat]) -> argparse.ArgumentParser:
    """The ``pysidtracker`` argument parser for the given registered ``formats``."""
    parser = argparse.ArgumentParser(
        prog="pysidtracker", description="Inspect and render C64 SID tunes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="print tune metadata")
    info.add_argument("song", help="input tune")
    info.set_defaults(func=lambda args: _info(args, formats))

    reglog = subparsers.add_parser("reglog", help="write a SID register log")
    reglog.add_argument("song", help="input tune")
    reglog.add_argument("output", help="register log file to write")
    reglog.add_argument("--seconds", type=float, default=60.0)
    reglog.set_defaults(func=lambda args: _reglog(args, formats))

    wav = subparsers.add_parser("wav", help="render through an emulated SID")
    wav.add_argument("song", help="input tune")
    wav.add_argument("output", help="WAV file to write")
    wav.add_argument("--seconds", type=float, default=60.0)
    wav.add_argument("--model", choices=CHIP_MODELS, default="8580")
    wav.set_defaults(func=lambda args: _wav(args, formats))

    for fmt in formats:
        for command in fmt.commands:
            sub = subparsers.add_parser(command.name, help=command.help)
            command.add_arguments(sub)
            sub.set_defaults(
                func=lambda args, cmd=command, owner=fmt: cmd.handler(args, owner)
            )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the ``pysidtracker`` console script."""
    formats = discover_formats()
    return run_cli(lambda: build_parser(formats), SidError, argv)
