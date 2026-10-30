"""The generic ``pysidtracker`` command-line tool.

Discovers every installed SID format (:mod:`pysidtracker.formats`) and offers
the shared ``info`` / ``reglog`` / ``wav`` / ``disasm`` / ``grammar`` / ``equiv``
commands -- plus any
format-specific subcommands each format contributes -- so a dependent package
registers a :class:`~pysidtracker.formats.SidFormat` instead of shipping its
own CLI.

Format selection is by content: each command loads the tune, picks the first
registered format whose parser recognises it, and runs against that format's
model and player.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from vagrant_moses.memmap import PAL

from .audio import CHIP_MODELS, render_player_wav, seconds_to_frames
from .cadence import playroutine_cadence
from .census import add_census_arguments
from .cli import add_grammar_command, print_info, run_cli
from .d64 import add_media_commands
from .disasm import disassemble, entry_labels, image_after_init
from .editorio import editor_io
from .errors import SidError, SidParseError
from .fixtures import add_fixtures_commands
from .formats import FormatCommand, SidFormat, discover_formats, recognize_format
from .image import SidImage
from .grammardiff import add_grammardiff_arguments
from .lint import add_lint_arguments, run_lint
from .playspec.cli import add_playspec_commands
from .reglog import register_writes_from_player, write_reglog
from .snapshot import add_snapshot_commands, resolve_format
from .source import read_bytes


def _load(song, formats: Sequence[SidFormat]) -> Tuple[SidFormat, bytes, object]:
    """Recognise ``song`` among ``formats`` and return ``(format, data, model)``."""
    data = read_bytes(song)
    fmt = recognize_format(formats, data)
    if fmt is None:
        raise SidParseError(f"no installed pysidtracker format recognises {song!r}")
    return fmt, data, fmt.parser.parse(data)


def _cadence(data: bytes) -> Tuple[int, float]:
    """The tune's play cadence (cycles per call) and CPU clock (Hz).

    From how the tune programs its interrupts (:func:`playroutine_cadence`); a
    bare ``.prg`` defaults to the PAL frame."""
    if data[:4] in (b"PSID", b"RSID"):
        image = SidImage.from_bytes(data)
        if image.header is not None:
            cadence = playroutine_cadence(image)
            return cadence.cycles_per_call, float(cadence.clock_hz)
    return PAL.cycles_per_frame, float(PAL.clock_hz)


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
    cycles_per_frame, clock_hz = _cadence(data)
    render_player_wav(
        fmt.player(model),
        args.output,
        seconds=args.seconds,
        model=args.model,
        cycles_per_frame=cycles_per_frame,
        clock_frequency=clock_hz,
    )
    print(f"wrote {args.output}")


def _model(song, spec: str, subtune, formats: Sequence[SidFormat]):
    """``(format, model)`` of ``song``'s ``subtune`` on the format ``spec`` names."""
    data = read_bytes(song)
    fmt = resolve_format(data, spec, formats)
    return fmt, fmt.load(data, subtune)


def _grammar(args, formats: Sequence[SidFormat]) -> None:
    fmt, model = _model(args.song, args.format, args.subtune, formats)
    for line in fmt.listing(model):
        print(line)


def _equiv(args, formats: Sequence[SidFormat]) -> None:
    texts = [
        fmt.canonical_text(model).splitlines()
        for fmt, model in (
            _model(song, args.format, args.subtune, formats) for song in args.songs
        )
    ]
    lines = list(difflib.unified_diff(*texts, *map(str, args.songs), lineterm=""))
    for line in lines:
        print(line)
    if lines:
        sys.exit(1)


def _disasm(args) -> None:
    image = SidImage.from_bytes(read_bytes(args.song))
    if args.after_init:
        image = image_after_init(image, subtune=args.subtune)
    for addr, raw, text in disassemble(
        image, args.start, args.end, labels=entry_labels(image)
    ):
        print(f"{addr:04X}  {raw.hex(' '):<9} {text}")


def _editor_io(args) -> None:
    image = SidImage.from_bytes(read_bytes(args.song))
    for name, address, rows in editor_io(image):
        print(f"{name}  ${address:04X}")
        for addr, raw, text in rows:
            print(f"{addr:04X}  {raw.hex(' '):<9} {text}")


def _hex_addr(value: str) -> int:
    """A hex C64 address: ``1000``, ``0x1000``, or ``$1000``."""
    text = value[1:] if value[:1] == "$" else value
    try:
        return int(text, 16)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a hex C64 address, got {value!r}"
        ) from exc


_FORMAT_HELP = "installed format name, module:attr, or auto (the recognising format)"


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

    grammar = add_grammar_command(
        subparsers, lambda args: _grammar(args, formats), song_help="input tune"
    )
    grammar.add_argument("--format", default="auto", help=_FORMAT_HELP)

    equiv = subparsers.add_parser(
        "equiv",
        help="exit 0 if two tunes play the same, else diff their canonical text",
    )
    equiv.add_argument("songs", nargs=2, metavar="SONG", help="input tunes")
    equiv.add_argument("--subtune", type=int, default=None, help="0-based subtune")
    equiv.add_argument("--format", default="auto", help=_FORMAT_HELP)
    equiv.set_defaults(func=lambda args: _equiv(args, formats))

    disasm = subparsers.add_parser(
        "disasm", help="disassemble a range of the loaded .sid/.prg image"
    )
    disasm.add_argument("song", help="input tune")
    disasm.add_argument(
        "start",
        nargs="?",
        type=_hex_addr,
        default=None,
        help="first address (hex; default the load address)",
    )
    disasm.add_argument(
        "end",
        nargs="?",
        type=_hex_addr,
        default=None,
        help="one past the last address (hex; default the image end)",
    )
    disasm.add_argument(
        "--after-init",
        action="store_true",
        help="disassemble the image state after running the tune's init routine",
    )
    disasm.add_argument(
        "--subtune",
        type=int,
        default=0,
        help="the subtune number passed to init (with --after-init)",
    )
    disasm.set_defaults(func=lambda args: _disasm(args))

    eio = subparsers.add_parser(
        "editor-io", help="list KERNAL LOAD/SAVE/OPEN call sites and their setup"
    )
    eio.add_argument("song", help="input tune")
    eio.set_defaults(func=_editor_io)

    fixtures = subparsers.add_parser(
        "fixtures",
        help="manage the CSDB native-fixture cache (fetch / manifest / verify)",
    )
    add_fixtures_commands(fixtures)

    media = subparsers.add_parser(
        "media", help="extract PRGs from release media (zip/gz, d64/d71/d81, t64)"
    )
    add_media_commands(media)

    snap = subparsers.add_parser(
        "snapshot", help="digest a corpus's models and probes, or diff two digests"
    )
    add_snapshot_commands(snap)

    gdiff = subparsers.add_parser(
        "grammardiff", help="exit 0 if two Lark grammars are structurally equal"
    )
    add_grammardiff_arguments(gdiff)

    lint = subparsers.add_parser(
        "lint", help="find unused symbols and unresolved doc references"
    )
    add_lint_arguments(lint)
    lint.set_defaults(func=run_lint)

    playspec = subparsers.add_parser(
        "playspec", help="check, run and prove playroutine state-machine specs"
    )
    add_playspec_commands(playspec)

    variant = subparsers.add_parser("variant", help="variant-level analysis")
    variant_sub = variant.add_subparsers(dest="variant_command", required=True)
    census = variant_sub.add_parser(
        "census", help="group tunes by variant vector and verdict"
    )
    add_census_arguments(census)
    census.set_defaults(formats=formats)

    _add_format_commands(subparsers, formats)
    return parser


def _command_parser(command: FormatCommand, prog: Optional[str] = None):
    """A standalone parser for one format's contributed ``command``."""
    parser = argparse.ArgumentParser(prog=prog, description=command.help)
    command.add_arguments(parser)
    return parser


def _dispatch(name: str, owners: Sequence[Tuple[SidFormat, FormatCommand]], args):
    """Run ``name`` for the owner whose parser recognises ``args.song``."""
    fmt = recognize_format([owner for owner, _ in owners], read_bytes(args.song))
    if fmt is None:
        raise SidParseError(
            f"no installed pysidtracker format with {name!r} recognises {args.song!r}"
        )
    command = next(cmd for owner, cmd in owners if owner is fmt)
    parser = _command_parser(command, f"pysidtracker {name} ({fmt.name})")
    command.handler(parser.parse_args([args.song, *args.rest]), fmt)


def _add_format_commands(subparsers, formats: Sequence[SidFormat]) -> None:
    """Add every contributed subcommand; a name several formats share dispatches
    on the tune's format, so each format keeps its own arguments, and a shared
    command's name stays the shared command."""
    owners: Dict[str, List[Tuple[SidFormat, FormatCommand]]] = {}
    for fmt in formats:
        for command in fmt.commands:
            owners.setdefault(command.name, []).append((fmt, command))
    for name, pairs in owners.items():
        if name in subparsers.choices:
            continue
        if len(pairs) == 1:
            fmt, command = pairs[0]
            sub = subparsers.add_parser(name, help=command.help)
            command.add_arguments(sub)
            sub.set_defaults(
                func=lambda args, cmd=command, owner=fmt: cmd.handler(args, owner)
            )
            continue
        usages = "\n".join(
            f"{fmt.name}: {_command_parser(cmd, name).format_usage().strip()}"
            for fmt, cmd in pairs
        )
        sub = subparsers.add_parser(
            name,
            help="; ".join(f"{fmt.name}: {cmd.help}" for fmt, cmd in pairs),
            description=f"The tune's format selects the arguments:\n{usages}",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        sub.add_argument("song", help="input tune")
        sub.add_argument(
            "rest", nargs=argparse.REMAINDER, help="the tune format's own arguments"
        )
        sub.set_defaults(func=lambda args, n=name, p=pairs: _dispatch(n, p, args))


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the ``pysidtracker`` console script."""
    formats = discover_formats()
    return run_cli(lambda: build_parser(formats), SidError, argv)
