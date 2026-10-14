"""The host-vs-oracle trace check tool, on recorded traces (no Docker)."""

import csv
import io
import shutil

import pytest

from pysidtracker import host_check, irqhost
from pysidtracker.oracle import SidtraceRow

from . import irqhost_tunes as tunes

SECONDS = 0.08


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(irqhost, "poweron_writes", lambda: None)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """An HVSC tree and work dir holding two synthetic tunes and their traces."""
    monkeypatch.setattr(host_check, "oracle_cache", lambda work: work)
    (tmp_path / "hvsc").mkdir()
    (tmp_path / "work" / "oracle").mkdir(parents=True)
    rows = []
    for name in ("raster_kernal", "badline_loop"):
        (tmp_path / "hvsc" / f"{name}.sid").write_bytes(tunes.build(name))
        shutil.copy(
            tunes.fixture(name), tmp_path / "work" / "oracle" / f"{name}_1.csv.zst"
        )
        rows.append({"path": f"{name}.sid", "md5": name, "player": "x", "songs": "1"})
    with open(tmp_path / "tunes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path


def _job(tree, name):
    tune = {"md5": name, "path": f"{name}.sid"}
    return tree / "hvsc" / f"{name}.sid", tune, 1, tree / "work", SECONDS, 60


def test_first_difference():
    a = [SidtraceRow(1, None, None, None, 0, 0, 1)] * 3
    b = a[:2] + [SidtraceRow(1, None, None, None, 0, 0, 2)]
    assert host_check.first_difference(a, a) is None
    assert host_check.first_difference(a, b) == 2
    assert host_check.first_difference(a, a[:2]) == 2


def test_check_matches_a_recorded_trace(tree):
    row = host_check.check(_job(tree, "raster_kernal"))
    assert row["verdict"] == "match" and row["rows"] > 100


def test_check_reports_the_first_differing_row(tree):
    oracle = tree / "work" / "oracle"
    shutil.copy(oracle / "badline_loop_1.csv.zst", oracle / "raster_kernal_1.csv.zst")
    row = host_check.check(_job(tree, "raster_kernal"))
    assert row["verdict"] == "differs" and row["first"] > 0
    assert row["oracle_row"] != row["host_row"]


def test_check_reports_oracle_and_host_errors(tree, monkeypatch):
    def unavailable(*_a, **_k):
        raise host_check.SidtraceUnavailable("no docker")

    (tree / "hvsc" / "raster_kernal.sid").write_bytes(b"not a sid")
    assert host_check.check(_job(tree, "raster_kernal"))["verdict"] == "host_error"
    monkeypatch.setattr(host_check, "cached_sidtrace", unavailable)
    assert host_check.check(_job(tree, "raster_kernal"))["verdict"] == "oracle_error"


def test_run_and_report(tree, capsys):
    out = tree / "work" / "out.csv"
    tally = host_check.run(
        tree / "hvsc",
        [tree / "tunes.csv"],
        tree / "work",
        seconds=SECONDS,
        jobs=2,
        out=out,
    )
    assert tally == {"match": 2}
    assert host_check.report(out) == {"match": 2}
    assert "'match': 2" in capsys.readouterr().out


def test_report_lists_differences(tmp_path, capsys):
    path = tmp_path / "r.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, host_check.FIELDS)
        writer.writeheader()
        writer.writerow({"verdict": "differs", "path": "a.sid", "song": 1, "first": 3})
    assert host_check.report(path) == {"differs": 1}
    assert "differs a.sid 1 3" in capsys.readouterr().out


def test_trace_prints_each_fetched_instruction():
    out = io.StringIO()
    start = irqhost.boot_cycle(False)
    host_check.trace(tunes.build("init_writes"), start, start + 40, out=out)
    lines = out.getvalue().splitlines()
    assert lines[0].split()[:3] == [str(start), lines[0].split()[1], "SEI"]
    assert any("STA $d418" in line for line in lines)


def test_main(tree, capsys):
    out = tree / "work" / "main.csv"
    argv = [
        "run",
        str(tree / "hvsc"),
        str(tree / "tunes.csv"),
        "--work",
        str(tree / "work"),
    ]
    assert (
        host_check.main(
            argv + ["--seconds", str(SECONDS), "--jobs", "1", "--out", str(out)]
        )
        == 0
    )
    assert host_check.main(["report", str(out)]) == 0
    sid = tree / "hvsc" / "raster_kernal.sid"
    start = irqhost.boot_cycle(False)
    assert host_check.main(["trace", str(sid), str(start), str(start + 10)]) == 0
    assert "SEI" in capsys.readouterr().out
