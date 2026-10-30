"""``pysidtracker playspec``: check, compile, run, model-check and validate specs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tqdm import tqdm

from ..header import parse_sid_header
from ..image import SidImage
from ..snapshot import load_spec
from ..source import read_bytes
from .lower import elaborate
from .player import SpecPlayer
from .pyexec import source as python_source
from .smt import bmc
from .syntax import load
from .tv import validate
from .unnative import equivalent, unnative


def _binding(args):
    return json.loads(Path(args.binding).read_text()) if args.binding else None


def _check(args) -> None:
    program = elaborate(load(args.spec), _binding(args))
    print(f"ok: {program.name}; order {', '.join(program.order)}")
    for note in program.notes:
        print(f"note: {note}")


def _python(args) -> None:
    print(python_source(elaborate(load(args.spec), _binding(args))), end="")


def _player(args, spec=None) -> SpecPlayer:
    data = read_bytes(args.tune)
    image = SidImage.from_bytes(data)
    header = parse_sid_header(data)
    entries = load_spec(args.entries) if args.entries else None
    return SpecPlayer(
        load(args.spec) if spec is None else spec,
        image.image,
        image.load,
        header.init_address,
        header.play_address,
        args.subtune,
        bind=load_spec(args.binder),
        entries=entries,
        psid=header.is_psid,
        sid=image if header.play_address == 0 else None,
        validate=getattr(args, "validate", False),
    )


def _run(args) -> None:
    player = _player(args)
    digest = hashlib.blake2b(digest_size=16)
    for _ in tqdm(range(args.frames), desc="frames", unit="frame"):
        player.play_frame()
        digest.update(bytes(player.regs))
    print(f"{args.frames} frames, grid digest {digest.hexdigest()}")
    if player.validator is not None:
        for (ref, base), message in player.validator.failures().items():
            print(f"malformed {ref} at ${base:04x}: {message}")


def _unnative(args) -> None:
    old = load(args.spec)
    player = _player(args, old)
    binding = load_spec(args.binder)(player.view(), player.entry)
    new = unnative(old, binding, player.view())
    equivalent(lambda spec: _player(args, spec), old, new, args.frames)
    Path(args.output).write_text(new.source)
    print(f"{args.output}: native-free, same output over {args.frames} frames")


def _bmc(args) -> None:
    program = elaborate(load(args.spec), _binding(args))
    result = bmc(program, args.frames, progress=True)
    if result.holds:
        print(f"holds for {args.frames} frames")
        return
    print(f"frame {result.frame}: {result.what} fails")
    for cell, value in sorted(result.model.items()):
        print(f"  {cell} = {value}")


def _tv(args) -> None:
    player = _player(args)
    view = player.view()
    binding = load_spec(args.binder)(view, player.entry)
    regs = dict((k, int(v, 0)) for k, v in (r.split("=") for r in args.reg))
    result = validate(
        load(args.spec),
        binding,
        args.routine,
        view,
        int(args.pc, 16),
        registers=regs,
        preserve=args.preserve,
        stop=None if args.stop is None else int(args.stop, 16),
    )
    print(
        f"{args.routine} vs ${result.pc:04x}: {'proven' if result.proven else 'counterexample'}"
        f" over {result.paths} 6502 paths"
    )
    if result.counterexample:
        print(json.dumps(result.counterexample, indent=1))


def add_playspec_commands(parser) -> None:
    """Add the ``check``/``python``/``run``/``bmc``/``tv``/``unnative`` subcommands."""
    sub = parser.add_subparsers(dest="playspec_command", required=True)

    def spec_cmd(name, func, text, binding=True):
        cmd = sub.add_parser(name, help=text)
        cmd.add_argument("spec", help=".play file")
        if binding:
            cmd.add_argument("--binding", help="JSON layout binding")
        cmd.set_defaults(func=func)
        return cmd

    spec_cmd("check", _check, "parse and type-check (with --binding: bind too)")
    spec_cmd("python", _python, "print the compiled Python frame (needs --binding)")
    bmc_cmd = spec_cmd("bmc", _bmc, "bounded check of invariants from symbolic memory")
    bmc_cmd.add_argument("--frames", type=int, default=2)
    for name, func, text in (
        ("run", _run, "play a tune with the spec"),
        ("tv", _tv, "validate a routine against the tune's 6502 code"),
        ("unnative", _unnative, "replace native hooks with spec statements"),
    ):
        cmd = spec_cmd(name, func, text, binding=False)
        cmd.add_argument("tune", help="input .sid")
        cmd.add_argument(
            "--binder", required=True, help="module:attr bind(view, entry)"
        )
        cmd.add_argument("--entries", help="module:attr entries(view)")
        cmd.add_argument("--subtune", type=int, default=0)
        if name == "unnative":
            cmd.add_argument("-o", "--output", required=True, help="new .play file")
            cmd.add_argument(
                "--frames", type=int, default=500, help="frames compared old vs new"
            )
        elif name == "run":
            cmd.add_argument("--frames", type=int, default=500)
            cmd.add_argument(
                "--validate", action="store_true", help="check stream grammars"
            )
        else:
            cmd.add_argument("--routine", required=True, help='e.g. "voice[0].parse"')
            cmd.add_argument(
                "--pc", required=True, help="hex address of the 6502 routine"
            )
            cmd.add_argument(
                "--reg", action="append", default=[], help="REG=value on entry"
            )
            cmd.add_argument(
                "--preserve", action="append", default=[], help="register kept"
            )
            cmd.add_argument(
                "--stop", help="hex address ending a stretch of a longer routine"
            )


__all__ = ["add_playspec_commands"]
