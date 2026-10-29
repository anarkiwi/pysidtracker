"""Corpus snapshots: digests of what a format makes of every tune, and their diff.

A snapshot maps each tune to the digest of its decoded model and of every probe
(``SidFormat.probes`` plus ``name=module:attr`` specs), or the error raised, so
two snapshots taken around a refactor show exactly which tunes it changed
(``pysidtracker snapshot build|diff``).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from tqdm import tqdm

from . import jobs
from .census import tune_paths
from .formats import SidFormat, find_format
from .source import read_bytes

MODEL = "model"
MISSING = "*"

Probe = Callable[[Any], Any]
Snapshot = Dict[str, Dict[str, str]]


def digest(value: Any) -> str:
    """BLAKE2b digest of ``value``'s bytes (``bytes``-like) or ``repr``."""
    raw = value if isinstance(value, (bytes, bytearray)) else repr(value).encode()
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def load_spec(spec: str) -> Any:
    """The object ``module:attr`` names."""
    module, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"expected module:attr, got {spec!r}")
    obj: Any = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def resolve_format(
    data: bytes, spec: str = "auto", formats: Optional[Iterable[SidFormat]] = None
) -> SidFormat:
    """The format ``spec`` names: ``module:attr`` (a :class:`SidFormat` or a
    factory of one), else a name among ``formats`` (default: installed) or ``auto``."""
    if ":" not in spec:
        return find_format(data, spec, formats)
    obj = load_spec(spec)
    return obj if isinstance(obj, SidFormat) else obj()


def _failed(exc: BaseException) -> str:
    return f"!{type(exc).__name__}: {exc}"


def _digest_of(probe: Probe, model: Any) -> str:
    try:
        return digest(probe(model))
    except Exception as exc:  # pylint: disable=broad-except
        return _failed(exc)


def fingerprint(
    data: bytes, fmt: SidFormat, probes: Optional[Mapping[str, Probe]] = None
) -> Dict[str, str]:
    """``{key: digest}`` of ``data``'s model and every probe of it
    (:meth:`SidFormat.snapshot_probes` then ``probes``); a key whose step raised
    holds ``!Error: message``."""
    try:
        model = fmt.load(data)
    except Exception as exc:  # pylint: disable=broad-except
        return {MODEL: _failed(exc)}
    out = {MODEL: digest(model)}
    for name, probe in {**fmt.snapshot_probes(), **(probes or {})}.items():
        out[name] = _digest_of(probe, model)
    return out


def parse_probes(specs: Iterable[str]) -> Dict[str, Probe]:
    """``{name: probe}`` of ``name=module:attr`` specs."""
    out: Dict[str, Probe] = {}
    for spec in specs:
        name, sep, target = spec.partition("=")
        if not sep:
            raise ValueError(f"expected name=module:attr, got {spec!r}")
        out[name] = load_spec(target)
    return out


def _task(path: str, fmt: str, specs: Tuple[str, ...]) -> Dict[str, str]:
    data = read_bytes(path)
    try:
        found = resolve_format(data, fmt)
    except Exception as exc:  # pylint: disable=broad-except
        return {MODEL: _failed(exc)}
    return fingerprint(data, found, parse_probes(specs))


def snapshot(
    tunes: Iterable[Tuple[str, Path]],
    fmt: str = "auto",
    probes: Iterable[str] = (),
    workers: Optional[int] = None,
    progress: bool = True,
) -> Snapshot:
    """``{name: fingerprint}`` of every ``(name, path)`` tune, over ``workers``
    processes; ``fmt`` and ``probes`` as :func:`resolve_format` /
    :func:`parse_probes` take them."""
    tunes = list(tunes)
    specs = tuple(probes)
    parse_probes(specs)
    tasks = [(str(path), fmt, specs) for _, path in tunes]
    out: Snapshot = {}
    bar = tqdm(total=len(tasks), unit="tune", disable=not progress)
    for index, value, error in jobs.imap(_task, tasks, workers):
        out[tunes[index][0]] = value if error is None else {MODEL: _failed(error)}
        bar.update()
    bar.close()
    return dict(sorted(out.items()))


def diff(old: Snapshot, new: Snapshot) -> List[Tuple[str, str, str, str]]:
    """``(tune, key, old, new)`` of every digest that differs; a tune or key one
    side lacks reads :data:`MISSING` there."""
    rows = []
    for name in sorted(old.keys() | new.keys()):
        before, after = old.get(name, {}), new.get(name, {})
        for key in sorted(before.keys() | after.keys()):
            was, now = before.get(key, MISSING), after.get(key, MISSING)
            if was != now:
                rows.append((name, key, was, now))
    return rows


def expected_errors(snap: Snapshot) -> Dict[str, Optional[str]]:
    """``{tune: message}`` of the error each tune's model raised, None where it decoded."""
    out: Dict[str, Optional[str]] = {}
    for name, fp in snap.items():
        model = fp.get(MODEL, "")
        out[name] = model.partition(": ")[2] if model.startswith("!") else None
    return out


def write_snapshot(snap: Snapshot, out) -> Path:
    """Write ``snap`` to ``out`` as JSON."""
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n", "utf-8")
    return path


def read_snapshot(path) -> Snapshot:
    """The snapshot JSON at ``path``."""
    return json.loads(Path(path).read_text("utf-8"))


def _build(args) -> None:
    snap = snapshot(
        tune_paths(args.inputs, args.hvsc), args.format, args.probe, args.jobs
    )
    failed = sum(any(v.startswith("!") for v in fp.values()) for fp in snap.values())
    write_snapshot(snap, args.out)
    print(f"wrote {args.out}: {len(snap)} tunes, {failed} with errors")


def _diff(args) -> None:
    old, new = read_snapshot(args.old), read_snapshot(args.new)
    rows = diff(old, new)
    for name, key, was, now in rows:
        print(f"{name}  {key}: {was} -> {now}")
    changed = len({row[0] for row in rows})
    print(f"{changed} of {len(old.keys() | new.keys())} tunes differ")
    if rows:
        sys.exit(1)


def add_snapshot_commands(parser) -> None:
    """Add ``build INPUTS... -o OUT`` and ``diff OLD NEW`` (exit 1 on a change)."""
    sub = parser.add_subparsers(dest="snapshot_command", required=True)
    build = sub.add_parser("build", help="digest every tune's model and probes")
    build.add_argument("inputs", nargs="+", help="tune files, directories, find CSVs")
    build.add_argument("-o", "--out", required=True, help="snapshot JSON to write")
    build.add_argument(
        "--format", default="auto", help="installed format name or module:attr"
    )
    build.add_argument(
        "--probe",
        action="append",
        default=[],
        help="name=module:attr, a callable of the model to digest (repeatable)",
    )
    build.add_argument("--hvsc", help="HVSC root that find CSV paths are relative to")
    build.add_argument("--jobs", type=int, default=None, help="worker processes")
    build.set_defaults(func=_build)
    cmp = sub.add_parser("diff", help="list the digests two snapshots differ in")
    cmp.add_argument("old", help="snapshot JSON before")
    cmp.add_argument("new", help="snapshot JSON after")
    cmp.set_defaults(func=_diff)


__all__ = [
    "add_snapshot_commands",
    "diff",
    "digest",
    "fingerprint",
    "load_spec",
    "parse_probes",
    "read_snapshot",
    "resolve_format",
    "snapshot",
    "write_snapshot",
]
