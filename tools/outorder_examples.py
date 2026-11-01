"""Write the emitted specs of a ``pysidtracker playspec reorder --out`` report as examples.

``python tools/outorder_examples.py REPORT.json TUNES.csv OUTDIR DOC.md``: one
``OUTDIR/<format>/<build>.play`` per format and build (the ``build`` column of the CSV),
``OUTDIR/census.json`` and the census table ``DOC.md``.
"""

import argparse
import csv
import json
from pathlib import Path

COLUMNS = (
    ("spec", lambda r: f"[{r['format']}/{r['build']}]({r['file']})"),
    ("tunes", lambda r: ", ".join(Path(t).stem for t in r["tunes"])),
    (
        "proof",
        lambda r: (
            "proved" if r["proved"] else "open" if r["proved"] is not None else "-"
        ),
    ),
    ("statements before/after", lambda r: f"{r['before']} / {r['after']}"),
    (
        "statement forms",
        lambda r: ", ".join(f"{k} {v}" for k, v in r["census"]["statements"].items()),
    ),
    ("operators", lambda r: " ".join(r["census"]["operators"])),
    ("tables", lambda r: len(r["census"]["tables"])),
    ("persistent cells", lambda r: r["census"]["persistent_cells"]),
    ("promoted cells", lambda r: r["census"]["promoted_cells"]),
    ("fixed bytes", lambda r: r["census"]["fixed_bytes"]),
    ("staged", lambda r: " ".join(r["census"]["staged"]) or "none"),
    (
        "voices / song registers",
        lambda r: f"{r['census']['voices']} / {r['census']['song_registers']}",
    ),
    ("loops (bounds)", lambda r: " ".join(map(str, r["census"]["loops"])) or "none"),
    ("self-counting cells", lambda r: r["census"]["counters"]),
    ("programs bound", lambda r: r["programs"]),
)


def main() -> None:
    """Write the examples, census JSON and census table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("tunes")
    parser.add_argument("outdir")
    parser.add_argument("doc")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    with open(args.tunes, newline="", encoding="utf-8") as rows:
        builds = {
            row["path"]: row.get("build") or row["player"]
            for row in csv.DictReader(rows)
        }
    out, rows = Path(args.outdir), {}
    for tune, entry in sorted(report.items()):
        if entry.get("verdict", "error") is not None or not entry.get("programs"):
            continue
        key = (entry["format"], builds.get(tune, entry["format"]))
        row = rows.get(key)
        if row is None:
            first = entry["programs"][0]
            path = out / key[0] / f"{key[1]}.play"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(first["text"])
            proofs = [p.get("proof") for p in entry["programs"]]
            row = rows[key] = {
                "format": key[0],
                "build": key[1],
                "file": str(path),
                "tunes": [],
                "programs": len(entry["programs"]),
                "proved": None if None in proofs else all(p["holds"] for p in proofs),
                "proof": proofs,
                "before": first["inlined"]["statements"],
                "after": first["after"]["statements"],
                "census": first["census"],
            }
        row["tunes"].append(tune)
    table = sorted(rows.values(), key=lambda r: (r["format"], r["build"]))
    (out / "census.json").write_text(json.dumps(table, indent=1))
    lines = ["| " + " | ".join(name for name, _ in COLUMNS) + " |"]
    lines.append("|" + "---|" * len(COLUMNS))
    for row in table:
        lines.append("| " + " | ".join(str(get(row)) for _, get in COLUMNS) + " |")
    Path(args.doc).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
