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


def test_census_groups_and_samples(tmp_path):
    src = tmp_path / "results.csv"
    _results(
        src,
        [
            ("t1", "exact", {"v": 1, "speed": 1}),
            ("t2", "differs", {"v": 1, "speed": 2}),
            ("t3", "exact", {"v": 2}),
        ],
    )
    classes = census.census(census.results_records(src))
    assert [len(r) for r in classes.values()] == [2, 1]
    sample = census.stratified_sample(classes, 1, seed=3)
    assert sample == census.stratified_sample(classes, 1, seed=3)
    assert all(len(v) == 1 for v in sample.values())
    rows = census.census_rows(classes, sample)
    assert rows[0]["count"] == 2
    assert json.loads(rows[0]["verdicts"]) == {"exact": 1, "differs": 1}


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
