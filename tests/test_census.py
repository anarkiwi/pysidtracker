"""Tests for ``pysidtracker variant census``."""

import csv
import json

from pysidtracker import census, maincli


def _results(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["key", "path", "verdict", "info"])
        w.writeheader()
        for key, verdict, info in rows:
            w.writerow(
                {"key": key, "path": key, "verdict": verdict, "info": json.dumps(info)}
            )


def test_vector_drops_src_and_cadence():
    info = {"a": 1, "a.src": "x", "speed": 1, "b": 2}
    assert census.vector(info) == (("a", 1), ("b", 2))
    assert census.vector(info, ["b"]) == (("b", 2),)


def test_cli_writes_csv(tmp_path, capsys):
    src = tmp_path / "results.csv"
    _results(src, [("t1", "exact", {"v": 1}), ("t2", "exact", {"v": 2})])
    out = tmp_path / "census.csv"
    code = maincli.main(
        ["variant", "census", str(src), "--sample", "1", "--out", str(out)]
    )
    assert code == 0
    assert "2 tunes, 2 class(es)" in capsys.readouterr().out
    with open(out, newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 2


def test_cluster_unrecognised_classes_by_variants(monkeypatch):
    import importlib

    variants_mod = importlib.import_module("pysidtracker.variants")

    monkeypatch.setattr(
        variants_mod,
        "build_signature",
        lambda data: data,
    )
    monkeypatch.setattr(
        variants_mod,
        "cluster",
        lambda sigs, threshold: ([0 if s == b"a" else 1 for s in sigs], [1, 1]),
    )
    recs = [
        {"name": "x", "verdict": "unrecognised", "info": {}, "data": b"a"},
        {"name": "y", "verdict": "unrecognised", "info": {}, "data": b"b"},
        {"name": "z", "verdict": "", "info": {"format": "f"}},
    ]
    out = census.cluster_unrecognised(recs)
    assert [r["info"] for r in out] == [
        {"variant": "c0"},
        {"variant": "c1"},
        {"format": "f"},
    ]
    assert all("data" not in r for r in out)
