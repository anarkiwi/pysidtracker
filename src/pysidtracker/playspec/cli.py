"""``pysidtracker playspec``: check, compile, run, model-check and validate specs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from ..header import parse_sid_header
from ..image import SidImage
from ..jobs import imap
from ..snapshot import load_spec
from ..source import read_bytes
from .lower import elaborate
from .outorder import ORDER, survey
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
    new = unnative(old, binding, player.view(), name=Path(args.output).stem)
    equivalent(lambda spec: _player(args, spec), old, new, args.frames)
    Path(args.output).write_text(new.source)
    print(
        f"{args.output}: native-free {new.name} extends {old.name},"
        f" same output over {args.frames} frames"
    )


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


def _tunes(paths) -> list:
    """The tune paths named, a ``.csv`` naming them in its ``path`` column."""
    out = []
    for path in paths:
        if path.endswith(".csv"):
            with open(path, newline="", encoding="utf-8") as rows:
                out += [row["path"] for row in csv.DictReader(rows)]
        else:
            out.append(path)
    return out


def _reorder(args) -> None:
    tunes = _tunes(args.tunes)
    tasks = [
        (t, args.order, args.frames, args.subtune, args.prove, args.budget)
        for t in tunes
    ]
    report, failed = {}, 0
    for index, value, error in tqdm(
        imap(survey, tasks, args.jobs), total=len(tasks), desc="tunes", unit="tune"
    ):
        report[tunes[index]] = value if error is None else {"error": repr(error)}
        failed += error is not None or value["verdict"] is not None
    staged = Counter(
        v for r in report.values() for p in r.get("programs", ()) for v in p["staged"]
    )
    print(
        f"{len(tunes) - failed}/{len(tunes)} tunes the same over {args.frames} frames"
    )
    print("staged:", dict(staged.most_common()))
    proofs = [
        p["proof"]["holds"]
        for r in report.values()
        for p in r.get("programs", ())
        if "proof" in p
    ]
    if proofs:
        print(f"{sum(proofs)}/{len(proofs)} programs proved")
    if args.emit:
        _emit(report, Path(args.emit))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


def _emit(report: dict, out: Path) -> None:
    """Write each emitted spec: to ``out`` itself for one ``.play``, else under it."""
    texts = [
        (Path(tune), r["format"], i, p["text"])
        for tune, r in report.items()
        for i, p in enumerate(r.get("programs", ()))
    ]
    if out.suffix == ".play" and len(texts) == 1:
        out.write_text(texts[0][3])
        return
    for tune, fmt, index, text in texts:
        path = out / fmt / f"{tune.stem}{f'_{index}' if index else ''}.play"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def add_playspec_commands(parser) -> None:
    """Add the ``check``/``python``/``run``/``bmc``/``tv``/``unnative``/``reorder``
    subcommands."""
    sub = parser.add_subparsers(dest="playspec_command", required=True)
    cmd = sub.add_parser(
        "reorder", help="play tunes with their specs rescheduled to a write order"
    )
    cmd.add_argument("tunes", nargs="+", help=".sid files, or .csv with a path column")
    cmd.add_argument("--order", default=ORDER, help="canonical write order")
    cmd.add_argument("--frames", type=int, default=1500)
    cmd.add_argument("--subtune", type=int, default=None)
    cmd.add_argument("--jobs", type=int, default=None)
    cmd.add_argument("--out", help="JSON report per tune")
    cmd.add_argument("--emit", help="a .play file (one program) or a directory of them")
    cmd.add_argument(
        "--prove", type=int, default=0, help="prove each call, per-query timeout in ms"
    )
    cmd.add_argument(
        "--budget", type=float, default=300.0, help="seconds per call's proof"
    )
    cmd.set_defaults(func=_reorder)

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
