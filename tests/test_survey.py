"""Unit tests for the whole-song HVSC oracle survey (no Docker required)."""

import csv
import hashlib
import json
import os
import subprocess

import numpy as np
import pytest
import zstandard

from pysidtracker import oracle, registers, survey
from pysidtracker.errors import SidFormatError
from pysidtracker.oracle import SIDTRACE_COLUMNS, SidtraceUnavailable

from .helpers import build_psid

CFG = """Other_Player\r
A9 00 8D END\r
GoatTracker_V2.x\r
BD ?? ?? D0 AND C9 FF END\r
EE 20 D0 end\r
(GT_Editor)\r
4C ?? 10 END\r
Empty_Player\r
"""
CPF = registers.PAL_CYCLES_PER_FRAME
NFRAMES = 30
SECONDS = NFRAMES * CPF / registers.PAL_CLOCK_HZ + 1e-6
GT_CODE = b"\xbd\x12\x34\xd0\x00\x00\xc9\xff"


def _songlengths(root, text):
    docs = root / "DOCUMENTS"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "Songlengths.md5").write_text(text)


def test_songlengths(tmp_path):
    _songlengths(
        tmp_path, "[Database]\n; /X.sid\n" + "a" * 32 + "=2:03.07 0:05 1:00.5(G)\n"
    )
    assert survey.songlengths(tmp_path) == {"a" * 32: [123.07, 5, 60.5]}


def test_find_filters_first_match_by_player(tmp_path):
    hvsc = tmp_path / "hvsc"
    tunes = {
        "MUSICIANS/A/gt.sid": build_psid(GT_CODE, 0x1000, songs=2, flags=0b1000),
        "MUSICIANS/B/other.sid": build_psid(GT_CODE + b"\xa9\x00\x8d", 0x1000),
        "MUSICIANS/C/edit.sid": build_psid(b"\x4c\x00\x10", 0x1000, flags=0b0100),
        "MUSICIANS/D/none.sid": build_psid(b"\x00", 0x1000),
    }
    for rel, data in tunes.items():
        (hvsc / rel).parent.mkdir(parents=True, exist_ok=True)
        (hvsc / rel).write_bytes(data)
    md5 = hashlib.md5(tunes["MUSICIANS/A/gt.sid"]).hexdigest()
    _songlengths(hvsc, f"{md5}=0:10 1:02.5\n")
    cfg = tmp_path / "sidid.cfg"
    cfg.write_bytes(CFG.encode())
    out = tmp_path / "work" / "tunes.csv"
    assert (
        survey.main(
            ["find", str(hvsc), "--sidid-cfg", str(cfg)]
            + ["--out", str(out), "--player", "goat|^\\(gt", "--jobs", "1"]
        )
        == 0
    )
    rows = survey.load_tunes(out)
    assert [
        (r["path"], r["player"], r["songs"], r["clock"], r["lengths"]) for r in rows
    ] == [
        ("MUSICIANS/A/gt.sid", "GoatTracker_V2.x", "2", "NTSC", "10 62.5"),
        ("MUSICIANS/C/edit.sid", "(GT_Editor)", "1", "PAL", ""),
    ]
    assert rows[0]["md5"] == md5
    tally = survey.find(hvsc, out, cfg, jobs=1)
    assert tally == {"GoatTracker_V2.x": 1, "Other_Player": 1, "(GT_Editor)": 1}


def test_grid_csv_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    grid = rng.integers(0, 3, (40, survey.NREGS)).astype(np.int16)
    out = tmp_path / "g.csv.zst"
    survey.write_grid(grid, out)
    assert (survey.read_grid(out, 50)[:40] == grid).all()
    assert (survey.read_grid(out, 50)[40:] == grid[-1]).all()
    assert (survey.read_grid(out, 20) == grid[:20]).all()


def test_compare_picks_lead_and_first_mismatch():
    ref = np.arange(10 * 25, dtype=np.int16).reshape(10, 25)
    lib = np.vstack([np.zeros((3, 25), np.int16), ref, ref[:6]])
    lead, bad, first = survey.compare(ref, lib)
    assert (lead, first, bad.any()) == (3, 10, False)
    lib[3 + 7, 4] ^= 1
    lead, bad, first = survey.compare(ref, lib)
    assert (lead, first, int(bad.sum())) == (3, 7, 1)


def test_compare_oracle_lead_and_gating():
    ref = np.arange(10 * 25, dtype=np.int16).reshape(10, 25) % 7
    lib = np.vstack([ref[2:], np.zeros((10, 25), np.int16)])
    lead, bad, first = survey.compare(ref, lib)
    assert (lead, first, len(bad), bad.any()) == (-2, 8, 8, False)
    o, l = survey.aligned(ref, lib, -2)
    assert (o == ref[2:]).all() and (l == lib[:8]).all()
    assert survey.gated(ref) and not survey.gated(lib[8:])
    only_v3 = np.zeros((2, 25), np.int16)
    only_v3[1, 18] = 0x41
    assert survey.gated(only_v3) and not survey.gated(only_v3 & 0xFE)


def _trace_csv():
    """A video-IRQ sidtrace CSV: reg 0 = frame, voice-1 gate on from frame 2."""
    lines = [",".join(SIDTRACE_COLUMNS), "100,,,,0,24,15"]
    for k in range(NFRAMES - 5):
        at = 1000 + k * CPF + 40
        lines.append(f"{at},,40,,0,0,{k}")
        if k == 2:
            lines.append(f"{at + 20},,60,,0,4,65")
    return "\n".join(lines) + "\n"


def _expected(nframes):
    k = np.minimum(np.arange(nframes), NFRAMES - 6)
    grid = np.zeros((nframes, 25), np.int16)
    grid[:, 0], grid[:, 4], grid[:, 24] = k, np.where(k >= 2, 65, 0), 15
    return grid


def render_exact(data, song, nframes):
    """Fake renderer: the oracle grid after two silent lead-in frames."""
    assert data.startswith(b"PSID") and song == 2
    grid = np.vstack([np.zeros((2, 25), np.int16), _expected(nframes)])
    grid[:2, 24] = 15
    return grid, {"variant": "test", "nframes": nframes}


def render_differs(data, song, nframes):
    """Fake renderer: :func:`render_exact` with voice 2 gated at frame 10."""
    grid, info = render_exact(data, song, nframes)
    grid[12, 11] = 1
    return grid, info


def render_parse_error(data, song, nframes):
    """Fake renderer that cannot decompile."""
    raise SidFormatError(f"cannot decompile {len(data)} {song} {nframes}")


def render_crash(data, song, nframes):
    """Fake renderer that crashes."""
    raise KeyError(len(data) + song + nframes)


def render_short(data, song, nframes):
    """Fake renderer that returns too few frames."""
    return render_exact(data, song, nframes)[0][: nframes - 1], {}


@pytest.fixture(name="survey_tree")
def _survey_tree(tmp_path):
    hvsc, work = tmp_path / "hvsc", tmp_path / "work"
    sid = build_psid(GT_CODE, 0x1000, songs=2)
    md5 = hashlib.md5(sid).hexdigest()
    (hvsc / "MUSICIANS").mkdir(parents=True)
    (hvsc / "MUSICIANS" / "t.sid").write_bytes(sid)
    tune = dict(path="MUSICIANS/t.sid", md5=md5, player="GoatTracker_V2.x")
    tune.update(songs="2", clock="PAL", lengths=f"5 {SECONDS!r}")
    tunes = tmp_path / "tunes.csv"
    with open(tunes, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, survey.TUNE_FIELDS)
        w.writeheader()
        w.writerow(tune)
    for song in (1, 2):
        ocsv = work / "oracle" / f"{md5}_{song}.csv.zst"
        ocsv.parent.mkdir(parents=True, exist_ok=True)
        ocsv.write_bytes(zstandard.ZstdCompressor().compress(_trace_csv().encode()))
    return hvsc, tunes, work, tune


def _job(survey_tree, render):
    hvsc, tunes, work, _ = survey_tree
    jobs = survey.jobs_for(hvsc, survey.load_tunes(tunes), work, render)
    return [j for j in jobs if j[2] == 2][0]


def test_oracle_frames_from_cached_trace(survey_tree):
    _, _, work, tune = survey_tree
    nframes, grid = survey.oracle_frames(
        work / "oracle" / f"{tune['md5']}_2.csv.zst", SECONDS, "PAL"
    )
    assert nframes == NFRAMES and (grid == _expected(NFRAMES)).all()


def test_check_exact(survey_tree):
    row = survey.check(_job(survey_tree, render_exact))
    assert row["verdict"] == "exact" and (row["lead"], row["first_diff"]) == (2, "")
    assert (row["frames"], row["diff_frames"], row["diff_regs"]) == (NFRAMES, 0, "")
    assert json.loads(row["info"]) == {
        "variant": "test",
        "nframes": NFRAMES + survey.LEAD,
    }
    assert row["oracle_gated"] and row["library_gated"]
    lib = survey.read_grid(survey_tree[2] / "library" / f"{row['key']}.csv.zst", 40)
    assert (lib == render_exact(b"PSID", 2, NFRAMES + survey.LEAD)[0][:40]).all()


def test_check_differs_and_diff(survey_tree, capsys):
    row = survey.check(_job(survey_tree, render_differs))
    assert (row["verdict"], row["lead"], row["first_diff"]) == ("differs", 2, 10)
    assert (row["diff_frames"], row["diff_regs"]) == (1, "v2_ctrl")
    hvsc, tunes, work, _ = survey_tree
    survey.diff(hvsc, tunes, row["key"], work, render_differs, context=1)
    out = capsys.readouterr().out.splitlines()
    assert [ln.split()[0] for ln in out[1:]] == ["9", "L", "10", "L", "^^", "11", "L"]
    assert out[5].index("^^") == out[4].index("01")


@pytest.mark.parametrize(
    "render,verdict,error",
    [
        (render_parse_error, "not_decompiled", "cannot decompile"),
        (render_crash, "library_error", "KeyError"),
        (render_short, "library_error", "ValueError: render grid shape"),
    ],
)
def test_check_library_failures(survey_tree, render, verdict, error):
    row = survey.check(_job(survey_tree, render))
    assert row["verdict"] == verdict and error in row["error"]
    assert row["frames"] == NFRAMES and "lead" not in row


def test_check_renders_oracle_once(survey_tree, monkeypatch):
    _, _, work, tune = survey_tree
    cached = work / "oracle" / f"{tune['md5']}_2.csv.zst"
    trace = cached.read_bytes()
    cached.unlink()
    calls = []

    def _render(_sid_path, out, **kw):
        calls.append(kw)
        out.write_bytes(trace)

    monkeypatch.setattr(survey, "run_sidtrace", _render)
    assert survey.check(_job(survey_tree, render_exact))["verdict"] == "exact"
    assert survey.check(_job(survey_tree, render_exact))["verdict"] == "exact"
    assert len(calls) == 1
    kw = calls[0]
    assert (kw["song"], kw["seconds"], kw["extra_args"]) == (2, 2, survey.ORACLE_ARGS)
    assert kw["user"] == f"{os.getuid()}:{os.getgid()}"

    def _fail(*_a, **_k):
        raise SidtraceUnavailable("no docker")

    monkeypatch.setattr(survey, "run_sidtrace", _fail)
    cached.unlink()
    row = survey.check(_job(survey_tree, render_exact))
    assert row["verdict"] == "oracle_error" and "no docker" in row["error"]


def test_run_sidtrace_song_and_user(tmp_path, monkeypatch):
    seen = []

    def _run(cmd, **_kw):
        seen.append(cmd)
        work = cmd[cmd.index("-v") + 1].split(":")[0]
        (tmp_path / work).joinpath("trace.csv.zst").write_bytes(b"x")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(oracle.subprocess, "run", _run)
    tune = tmp_path / "t.sid"
    tune.write_bytes(b"PSID")
    oracle.run_sidtrace(
        tune, tmp_path / "a.csv.zst", song=3, user="1:2", extra_args=["-q"]
    )
    oracle.run_sidtrace(tune, tmp_path / "b.csv.zst")
    assert seen[0][3:5] == ["--user", "1:2"] and seen[0][-2:] == ["-o3", "-q"]
    assert "--user" not in seen[1] and not any(a.startswith("-o") for a in seen[1])


def test_run_report_cli(survey_tree, capsys):
    hvsc, tunes, work, _ = survey_tree
    args = ["run", str(hvsc), str(tunes), "--work", str(work), "--jobs", "1"]
    survey.main(args + ["--render", f"{__name__}:render_exact", "--player", "goat"])
    rows = survey.load_tunes(work / "results.csv")
    assert [(r["song"], r["verdict"]) for r in rows] == [
        ("1", "library_error"),
        ("2", "exact"),
    ]
    capsys.readouterr()
    survey.main(["report", str(work / "results.csv")])
    out = capsys.readouterr().out
    assert "('GoatTracker_V2.x', 'exact'), 1)" in out and "AssertionError" in out
    assert survey.run(hvsc, tunes, work, render_exact, player="nomatch", jobs=1) == {}


@pytest.mark.parametrize(
    "frame,diffs,row,expect",
    [
        (0, [(4, 0x41, 0x51), (11, 0x81, 0x91)], {}, ("frame0", "offset+0x10", "ctrl")),
        (3, [(2, 0x20, 0), (22, 0, 0x20)], {}, ("early", "misplaced", "fc pw")),
        (20, [(5, 0x0F, 0), (6, 0xF0, 0)], {}, ("mid", "missing", "ad sr")),
        (300, [(24, 0, 0x0F), (23, 0, 1)], {}, ("late", "extra", "res vol")),
        (5, [(0, 3, 4), (1, 9, 7)], {}, ("early", "value", "freq")),
        (5, [(4, 0x41, 0x51)], {}, ("early", "value", "ctrl")),
        (
            0,
            [(4, 0x41, 0)],
            {"library_gated": "False", "oracle_gated": "True"},
            ("frame0", "silent", "ctrl"),
        ),
    ],
)
def test_classify(frame, diffs, row, expect):
    assert survey.classify(frame, diffs, row) == expect


def test_info_rates():
    rows = [
        {"verdict": v, "info": json.dumps(i)}
        for v, i in [
            ("exact", {"opt": True, "n": 1}),
            ("differs", {"opt": True, "n": 2}),
            ("differs", {"opt": False, "n": 3}),
            ("library_error", {"opt": False}),
        ]
    ]
    assert survey.info_rates(rows, max_values=2) == {
        ("opt", "true"): (2, 1),
        ("opt", "false"): (1, 0),
    }


def test_triage_cli(survey_tree, capsys):
    hvsc, tunes, work, tune = survey_tree
    survey.run(hvsc, tunes, work, render_differs, jobs=1)
    capsys.readouterr()
    assert survey.main(["triage", str(work), "--jobs", "1"]) == 0
    out = capsys.readouterr().out
    rows = survey.load_tunes(work / "triage.csv")
    assert [
        (r["key"], r["first_diff"], r["when"], r["kind"], r["roles"]) for r in rows
    ] == [(f"{tune['md5']}_2", "10", "mid", "extra", "ctrl")]
    assert json.loads(rows[0]["diffs"]) == [[11, 0, 1]]
    assert "extra" in out and '0.0% exact  variant="test"' in out


def test_diff_cli_reports_failure(survey_tree, capsys):
    hvsc, tunes, work, tune = survey_tree
    key = f"{tune['md5']}_2"
    args = ["diff", str(hvsc), str(tunes), key, "--work", str(work)]
    survey.main(args + ["--render", f"{__name__}:render_crash"])
    assert capsys.readouterr().out.count("\n") == 1
