"""``pysidtracker variant census``: group tunes by variant vector and verdict.

A census collapses a survey into its distinct :meth:`Variant.as_info` vectors, so
a format package sees which build variants exist, how many tunes each covers and
how the renderer fares on each.  Records come from a survey ``results.csv`` (the
``info`` column and ``verdict``) or straight from tune files / directories / a
``find`` CSV through the installed formats (no verdict).  ``--sample N`` draws a
deterministic stratified sample of up to ``N`` tunes per class, so one tune of
every variant can be pulled for study without a hand-written sampler.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from tqdm import tqdm

from .errors import SidError
from .formats import SidFormat, find_format
from .source import read_bytes

# Survey info fields describing the play routine, not the format's variant.
CADENCE_KEYS = ("rsid", "trigger", "speed", "dynamic")
CENSUS_FIELDS = ["class", "count", "verdicts", "sample"]
TUNE_SUFFIXES = (".sid", ".prg")

Record = Dict[str, Any]


def vector(info: Dict[str, Any], keys: Optional[Sequence[str]] = None) -> Tuple:
    """The class vector of ``info``: ``(key, value)`` pairs, sorted by key.

    ``keys`` selects the fields; by default every field except the ``.src``
    provenance strings and the cadence fields.
    """
    if keys is None:
        keys = [k for k in info if not k.endswith(".src") and k not in CADENCE_KEYS]
    return tuple((k, info.get(k)) for k in sorted(keys))


def results_records(path) -> Iterator[Record]:
    """Records of a survey ``results.csv``: ``name``, ``verdict`` and ``info``."""
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield {
                "name": row.get("key") or row["path"],
                "verdict": row.get("verdict", ""),
                "info": json.loads(row.get("info") or "{}"),
            }


def tune_paths(paths: Iterable, hvsc=None) -> List[Tuple[str, Path]]:
    """``(name, file)`` of every tune under ``paths``: files, directories
    (recursive) and ``find`` CSVs (rows relative to ``hvsc``)."""
    found: List[Tuple[str, Path]] = []
    for path in map(Path, paths):
        if path.suffix.lower() == ".csv":
            root = Path(hvsc) if hvsc else Path(".")
            with open(path, newline="", encoding="utf-8") as f:
                found.extend((r["path"], root / r["path"]) for r in csv.DictReader(f))
        elif path.is_dir():
            files = sorted(
                p for p in path.rglob("*") if p.suffix.lower() in TUNE_SUFFIXES
            )
            found.extend((str(p.relative_to(path)), p) for p in files)
        else:
            found.append((path.name, path))
    return found


def tune_records(
    tunes: Sequence[Tuple[str, Path]],
    formats: Optional[Sequence[SidFormat]] = None,
    progress: bool = True,
) -> Iterator[Record]:
    """Records of tune files, each classed by the installed format recognising it
    (``verdict`` ``unrecognised`` when none does)."""
    for name, path in tqdm(tunes, unit="tune", disable=not progress):
        data = read_bytes(path)
        try:
            fmt = find_format(data, formats=formats)
            info = {"format": fmt.name, **fmt.variant_info(fmt.load(data))}
            verdict = ""
        except SidError:
            info, verdict = {}, "unrecognised"
            yield {"name": name, "verdict": verdict, "info": info, "data": data}
            continue
        yield {"name": name, "verdict": verdict, "info": info}


def cluster_unrecognised(
    records: Iterable[Record], threshold: float = 0.8
) -> List[Record]:
    """``records`` with each unrecognised tune classed by its ``variants`` cluster.

    A tune no installed format recognises gets ``info["variant"] = "c<N>"`` (the
    :func:`~pysidtracker.variants.cluster` id among the unrecognised tunes), so
    the residue groups by player build; one that cannot be fingerprinted keeps
    an empty class.
    """
    from .variants import (
        build_signature,
        cluster,
    )  # pylint: disable=import-outside-toplevel

    recs = list(records)
    sigs, owners = [], []
    for rec in recs:
        data = rec.pop("data", None)
        if rec["verdict"] != "unrecognised" or data is None:
            continue
        try:
            sigs.append(build_signature(data))
        except (SidError, ValueError, ImportError):
            continue
        owners.append(rec)
    assign, _ = cluster(sigs, threshold)
    for rec, ci in zip(owners, assign):
        rec["info"] = {"variant": f"c{ci}"}
    return recs


def census(
    records: Iterable[Record], keys: Optional[Sequence[str]] = None
) -> Dict[Tuple, List[Record]]:
    """The ``records`` grouped by their :func:`vector`, largest class first."""
    classes: Dict[Tuple, List[Record]] = {}
    for rec in records:
        classes.setdefault(vector(rec["info"], keys), []).append(rec)
    return dict(sorted(classes.items(), key=lambda kv: -len(kv[1])))


def stratified_sample(
    classes: Dict[Tuple, List[Record]], per_class: int, seed: int = 0
) -> Dict[Tuple, List[Record]]:
    """Up to ``per_class`` records of every class, chosen with a ``seed``ed RNG
    (the same inputs always give the same sample)."""
    rng = random.Random(seed)
    return {
        cls: sorted(rng.sample(recs, min(per_class, len(recs))), key=_name)
        for cls, recs in classes.items()
    }


def _name(rec: Record) -> str:
    return rec["name"]


def census_rows(
    classes: Dict[Tuple, List[Record]], sample: Dict[Tuple, List[Record]]
) -> List[Dict[str, Any]]:
    """The CSV rows: class vector (JSON), size, verdict counts and sampled names."""
    return [
        {
            "class": json.dumps(dict(cls), sort_keys=True),
            "count": len(recs),
            "verdicts": json.dumps(
                dict(Counter(r["verdict"] or "-" for r in recs).most_common())
            ),
            "sample": " ".join(r["name"] for r in sample.get(cls, ())),
        }
        for cls, recs in classes.items()
    ]


def write_census(rows: Sequence[Dict[str, Any]], out) -> Path:
    """Write ``rows`` to ``out`` as CSV."""
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, CENSUS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def census_cli(args) -> None:
    """``variant census`` handler: collect records, group, sample, print, write."""
    keys = args.keys.split(",") if args.keys else None
    sources = [Path(p) for p in args.inputs]
    if len(sources) == 1 and sources[0].name == "results.csv":
        records = results_records(sources[0])
    else:
        records = cluster_unrecognised(
            tune_records(tune_paths(sources, args.hvsc), args.formats)
        )
    classes = census(records, keys)
    sample = stratified_sample(classes, args.sample, args.seed) if args.sample else {}
    rows = census_rows(classes, sample)
    total = sum(r["count"] for r in rows)
    print(f"{total} tunes, {len(rows)} class(es)")
    for row in rows:
        print(f"{row['count']:7d} {row['class']}  {row['verdicts']}")
        if row["sample"]:
            print(f"        sample: {row['sample']}")
    if args.out:
        write_census(rows, args.out)
        print(f"wrote {args.out}")


def add_census_arguments(parser) -> None:
    """Configure the ``variant census`` subparser."""
    parser.add_argument(
        "inputs",
        nargs="+",
        help="a survey results.csv, or tune files / directories / find CSVs",
    )
    parser.add_argument("--hvsc", help="HVSC root that find CSV paths are relative to")
    parser.add_argument("--keys", help="comma-joined info fields forming the class")
    parser.add_argument(
        "--sample", type=int, default=0, help="sample up to N tunes from every class"
    )
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument("--out", help="census CSV to write")
    parser.set_defaults(func=census_cli, formats=None)


__all__ = [
    "CENSUS_FIELDS",
    "add_census_arguments",
    "census",
    "census_cli",
    "census_rows",
    "cluster_unrecognised",
    "results_records",
    "stratified_sample",
    "tune_paths",
    "tune_records",
    "vector",
    "write_census",
]
