"""Tests for the cadence-vs-sidtrace interrupt check (no Docker required)."""

import bisect
import csv
import hashlib

import pytest
import zstandard

from pysidtracker import cadence_check as cc
from pysidtracker import registers as reg
from pysidtracker import oracle, survey
from pysidtracker.oracle import SidtraceRow, SidtraceUnavailable

from .helpers import event, handled, trace_csv
from .test_cadence import latch, psid

PAL = reg.PAL_CYCLES_PER_FRAME


def _rows(raises, source="video_irq", delay=100):
    """Handled interrupts raised at ``raises``, each making one SID write."""
    return [
        row
        for at in raises
        for row in handled(source, at, [(at + delay, 0, at & 0xFF)])
    ]


def _digi_in_music(period=145):
    """A music IRQ per frame writing voice registers, interrupted by sample NMIs.

    Each NMI is raised every ``period`` cycles and writes ``$D418``; the IRQ is
    entered after the running NMI returns and makes one voice write after each
    of the next seven NMIs, so every voice write's latest raise is an NMI.
    """
    rows = []
    for k, at in enumerate(range(1000, 400000, period)):
        rows += handled("nmi", at, [(at + 20, 24, k & 0x0F)])
    rtis = [r.cycle for r in rows if r.value == "rti"]
    for t in range(800, 400000 - PAL, PAL):
        nxt = bisect.bisect_left(rtis, t)
        busy = rtis[nxt] - 14 <= t
        entry = rtis[nxt] + 1 if busy else t + 1
        after = rtis[nxt + busy : nxt + busy + 7]
        writes = [(c + 2, i, t & 0xFF) for i, c in enumerate(after)]
        rows += [event(t, "cia_irq"), event(entry, "cia_irq_entry")]
        rows += [SidtraceRow(c, None, None, None, 0, r, v) for c, r, v in writes]
        rows.append(event(writes[-1][0] + 1, "rti"))
    return sorted(rows, key=lambda r: (r.cycle, r.value == "rti"))


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
            trace = zstandard.ZstdCompressor().compress(trace_csv(rows).encode())
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

    monkeypatch.setattr(oracle, "run_sidtrace", _fail)
    row = cc.check(job[:3] + (cc.DEFAULT_CADENCE, 10, 5))
    assert row["verdict"] == "oracle_error" and "no docker" in row["error"]
