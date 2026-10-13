"""Tests for the cadence-vs-sidtrace interrupt check (no Docker required)."""

import csv
import hashlib

import pytest
import zstandard

from pysidtracker import cadence_check as cc
from pysidtracker import registers as reg
from pysidtracker import survey
from pysidtracker.oracle import SidtraceRow, SidtraceUnavailable

from .test_cadence import latch, psid

PAL = reg.PAL_CYCLES_PER_FRAME


def _rows(raises, source="video_irq", delay=100):
    """Interrupt rows at ``raises`` each followed by one SID write."""
    rows = []
    for at in raises:
        rows.append(SidtraceRow(at, None, None, None, None, None, source))
        rows.append(SidtraceRow(at + delay, None, None, None, 0, 0, at & 0xFF))
    return rows


def _csv(rows):
    lines = [
        "cycle,cycle_since_nmi,cycle_since_video_irq,cycle_since_cia_irq,chip,reg,value"
    ]
    for r in rows:
        chip = "" if r.chip is None else str(r.chip)
        regs = "" if r.reg is None else str(r.reg)
        lines.append(f"{r.cycle},,,,{chip},{regs},{r.value}")
    return "\n".join(lines) + "\n"


def test_writes_are_attributed_to_the_latest_raise():
    cia = _rows(range(1000, 200000, 4000), "cia_irq")
    video = [
        SidtraceRow(t, None, None, None, None, None, "video_irq")
        for t in range(500, 200000, PAL)
    ]
    source, raises = cc.interrupt_cadence(sorted(cia + video))
    assert source == "cia_irq" and len(raises) == len(range(1000, 200000, 4000))


def test_no_write_after_a_raise_has_no_source():
    rows = [SidtraceRow(10, None, None, None, 0, 0, 1)]
    source, raises = cc.interrupt_cadence(rows)
    assert source is None and not len(raises)


def test_measure_drops_the_first_raise_held_by_init():
    raises = [0] + [100000 + i * PAL // 2 for i in range(40)]
    meas = cc.measure(_rows(raises), PAL)
    assert meas["meas_source"] == "video_irq" and meas["calls"] == 39
    assert float(meas["meas_cpf"]) == pytest.approx(2.0)
    assert meas["gaps"] == 1 and meas["raises"] == 41


def test_measure_with_too_few_raises():
    assert "calls" not in cc.measure(_rows([0, 10]), PAL)


@pytest.mark.parametrize(
    "source,cpf,expect",
    [
        ("pal_video", 2.0, "agree"),
        ("cia_timer", 2.0, "source"),
        ("pal_video", 1.0, "rate"),
        ("pal_video", 2.0 * 40 / 39, "agree"),
        (None, 1.0, "unknown"),
    ],
)
def test_verdict(source, cpf, expect):
    meas = cc.measure(_rows([0] + [100000 + i * PAL // 2 for i in range(40)]), PAL)
    assert cc.verdict(source, cpf, meas, PAL) == expect


def test_verdict_without_raises():
    assert cc.verdict("pal_video", 1.0, {"meas_source": ""}, PAL) == "no_raises"


@pytest.fixture(name="tree")
def _tree(tmp_path, monkeypatch):
    monkeypatch.setattr(survey, "sidtrace_image_id", lambda: None)
    hvsc, work = tmp_path / "hvsc", tmp_path / "work"
    tunes = {
        "a.sid": (psid(b""), "Player_A", _rows(range(0, 40 * PAL, PAL))),
        "b.sid": (
            psid(latch(PAL // 2 - 1), speed=1),
            "Player_B",
            _rows(range(0, 40 * PAL, PAL), "cia_irq"),
        ),
    }
    hvsc.mkdir()
    (work / "oracle").mkdir(parents=True)
    out = tmp_path / "tunes.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, survey.TUNE_FIELDS)
        w.writeheader()
        for name, (data, player, rows) in tunes.items():
            (hvsc / name).write_bytes(data)
            md5 = hashlib.md5(data).hexdigest()
            w.writerow(
                dict(
                    path=name, md5=md5, player=player, songs=1, clock="PAL", lengths=""
                )
            )
            trace = zstandard.ZstdCompressor().compress(_csv(rows).encode())
            (work / "oracle" / f"{md5}_1.csv.zst").write_bytes(trace)
    return hvsc, out, work


def test_run_and_report(tree, capsys):
    hvsc, tunes, work = tree
    assert (
        cc.main(
            [
                "run",
                str(hvsc),
                str(tunes),
                str(tunes),
                "--work",
                str(work),
                "--jobs",
                "1",
            ]
        )
        == 0
    )
    with open(work / "cadence.csv", newline="", encoding="utf-8") as f:
        rows = {r["player"]: r for r in csv.DictReader(f)}
    assert rows["Player_A"]["verdict"] == "agree"
    assert rows["Player_B"]["verdict"] == "rate"
    assert rows["Player_B"]["pred_source"] == "cia_timer"
    assert cc.main(["report", str(work / "cadence.csv")]) == 0
    assert "100.0% agree  Player_A" in capsys.readouterr().out


def test_check_errors(tree, monkeypatch):
    hvsc, tunes, work = tree
    tune = survey.load_tunes(tunes)[0]
    job = (hvsc / tune["path"], tune, work, "builtins:len", 10, 5)
    assert cc.check(job)["verdict"] == "cadence_error"
    (work / "oracle" / f"{tune['md5']}_1.csv.zst").unlink()

    def _fail(*_a, **_k):
        raise SidtraceUnavailable("no docker")

    monkeypatch.setattr(cc, "run_sidtrace", _fail)
    row = cc.check(job[:3] + (cc.DEFAULT_CADENCE, 10, 5))
    assert row["verdict"] == "oracle_error" and "no docker" in row["error"]
