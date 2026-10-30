"""Unit tests for the whole-song HVSC oracle survey (no Docker required)."""

import csv
import hashlib
import json
import multiprocessing
import os
import subprocess

import numpy as np
import pytest
import zstandard

from pysidtracker import oracle, registers, survey
from pysidtracker.errors import SidFormatError
from pysidtracker.oracle import SIDTRACE_COLUMNS, SidtraceRow, SidtraceUnavailable

from .helpers import build_psid, build_psid_embedded_load, event, trace_csv

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
    for k in range(NFRAMES + 1):
        at = 1000 + k * CPF + 40
        lines.append(f"{at - 40},,0,,,,video_irq")
        lines.append(f"{at - 33},,7,,,,video_irq_entry")
        lines.append(f"{at},,40,,0,0,{k}")
        if k == 2:
            lines.append(f"{at + 20},,60,,0,4,65")
        lines.append(f"{at + 30},,70,,,,rti")
    return "\n".join(lines) + "\n"


def _never_returns_csv():
    """A never-returning main loop: each call raises and the tune writes, but the
    I flag is set so the raised IRQ is never taken -- no entry, no ``rti``."""
    rows = [SidtraceRow(100, None, None, None, 0, 24, 15)]
    for k in range(NFRAMES + 1):
        at = 1000 + k * CPF + 40
        rows.append(event(at - 40, "cia_irq"))
        rows.append(SidtraceRow(at, None, None, None, 0, 0, k))
    return trace_csv(rows)


def _expected(nframes):
    k = np.minimum(np.arange(nframes), NFRAMES)
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


def render_hang(data, song, nframes):
    """Fake renderer that never returns."""
    while data and song and nframes:
        pass


def render_spin(data, song, nframes):
    """Fake renderer that spins in C, where no signal handler runs."""
    return hashlib.pbkdf2_hmac("sha256", data, bytes(song), (1 << 31) - 1 + 0 * nframes)


def render_short(data, song, nframes):
    """Fake renderer that returns too few frames."""
    return render_exact(data, song, nframes)[0][: nframes - 1], {}


@pytest.fixture(autouse=True)
def _image_id(monkeypatch):
    monkeypatch.setattr(survey, "sidtrace_image_id", lambda: "sha256:a")


@pytest.fixture(name="survey_tree")
def _survey_tree(tmp_path):
    hvsc, work = tmp_path / "hvsc", tmp_path / "work"
    sid = build_psid(GT_CODE, 0x1000, songs=2)
    md5 = hashlib.md5(sid).hexdigest()
    (hvsc / "MUSICIANS").mkdir(parents=True)
    (hvsc / "MUSICIANS" / "t.sid").write_bytes(sid)
    tune = dict(path="MUSICIANS/t.sid", md5=md5, player="GoatTracker_V2.x")
    tune.update(songs="2", clock="PAL", lengths=f"{SECONDS!r} {SECONDS!r}")
    tunes = tmp_path / "tunes.csv"
    with open(tunes, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, survey.TUNE_FIELDS)
        w.writeheader()
        w.writerow(tune)
    survey.oracle_cache(work)
    for song in (1, 2):
        ocsv = work / "oracle" / f"{md5}_{song}.csv.zst"
        ocsv.write_bytes(zstandard.ZstdCompressor().compress(_trace_csv().encode()))
    return hvsc, tunes, work, tune


def _job(survey_tree, render, timeout=600):
    hvsc, tunes, work, _ = survey_tree
    tunes = survey.load_tunes(tunes)
    jobs = survey.jobs_for(hvsc, tunes, work, render, timeout=timeout)
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

    monkeypatch.setattr(oracle, "run_sidtrace", _render)
    assert survey.check(_job(survey_tree, render_exact))["verdict"] == "exact"
    assert survey.check(_job(survey_tree, render_exact))["verdict"] == "exact"
    assert len(calls) == 1
    kw = calls[0]
    assert (kw["song"], kw["seconds"], kw["extra_args"]) == (2, 2, survey.ORACLE_ARGS)
    assert kw["user"] == f"{os.getuid()}:{os.getgid()}"

    def _fail(*_a, **_k):
        raise SidtraceUnavailable("no docker")

    monkeypatch.setattr(oracle, "run_sidtrace", _fail)
    cached.unlink()
    row = survey.check(_job(survey_tree, render_exact))
    assert row["verdict"] == "oracle_error" and "no docker" in row["error"]


def test_oracle_frames_never_returning_trace(survey_tree):
    # A trace that raises but never ``rti``s is valid, not unplayable.
    _, _, work, tune = survey_tree
    ocsv = work / "oracle" / f"{tune['md5']}_2.csv.zst"
    ocsv.write_bytes(zstandard.ZstdCompressor().compress(_never_returns_csv().encode()))
    nframes, grid = survey.oracle_frames(ocsv, SECONDS, "PAL")
    assert nframes == NFRAMES and grid.shape == (NFRAMES, survey.NREGS)


def test_check_reuses_never_returning_trace(survey_tree, monkeypatch):
    # The cached never-returning trace is a verdict, not an ``oracle_error``, and
    # is never re-rendered (re-rendering cannot add an ``rti``).
    _, _, work, tune = survey_tree
    ocsv = work / "oracle" / f"{tune['md5']}_2.csv.zst"
    ocsv.write_bytes(zstandard.ZstdCompressor().compress(_never_returns_csv().encode()))

    def _no_render(*_a, **_k):
        raise AssertionError("a never-returning trace must be reused, not re-rendered")

    monkeypatch.setattr(oracle, "run_sidtrace", _no_render)
    row = survey.check(_job(survey_tree, render_exact))
    assert row["verdict"] in ("exact", "differs")


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
        tune,
        tmp_path / "a.csv.zst",
        song=3,
        user="1:2",
        extra_args=["-q"],
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


@pytest.mark.parametrize("image,kept", [("sha256:a", 2), (None, 2), ("sha256:b", 0)])
def test_oracle_cache_stamp(survey_tree, monkeypatch, image, kept):
    work = survey_tree[2]
    monkeypatch.setattr(survey, "sidtrace_image_id", lambda: image)
    cache = survey.oracle_cache(work)
    assert len(list(cache.glob("*.csv.zst"))) == kept
    assert ("sha256:b" in (cache / "STAMP").read_text("utf-8")) == (kept == 0)


def test_oracle_cache_unstamped_is_stale(survey_tree):
    work = survey_tree[2]
    (work / "oracle" / "STAMP").unlink()
    assert not list(survey.oracle_cache(work).glob("*.csv.zst"))


@pytest.mark.parametrize(
    "codes,expect,calls",
    [([0], "sha256:x", 1), ([1, 0, 0], "sha256:x", 3), ([1, 1, 1], None, 3)],
)
def test_sidtrace_image_id(monkeypatch, codes, expect, calls):
    seen = []

    def _run(cmd, **_):
        seen.append(cmd[1])
        return subprocess.CompletedProcess(cmd, codes[len(seen) - 1], "sha256:x\n")

    monkeypatch.setattr(oracle.subprocess, "run", _run)
    assert oracle.sidtrace_image_id("img") == expect and len(seen) == calls
    assert seen[:2] == ["image", "pull"][: min(calls, 2)]


def test_sidtrace_image_id_without_docker(monkeypatch):
    def _run(cmd, **_):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(oracle.subprocess, "run", _run)
    assert oracle.sidtrace_image_id() is None


def test_check_render_timeout(survey_tree):
    row = survey.check(_job(survey_tree, render_hang, timeout=0.2))
    assert row["verdict"] == "library_error"
    assert row["error"] == "TimeoutError: exceeded 0.2s"
    assert survey.check(_job(survey_tree, render_exact, 5))["verdict"] == "exact"


def test_oracle_cache_stamp_without_row_kinds_is_stale(survey_tree):
    work = survey_tree[2]
    stamp = {"image": "sha256:a", "args": survey.ORACLE_ARGS}
    (work / "oracle" / "STAMP").write_text(json.dumps(stamp), "utf-8")
    assert not list(survey.oracle_cache(work).glob("*.csv.zst"))


def render_frame0(data, song, nframes):
    """Fake renderer: :func:`render_exact` with the volume wrong on frame 0 only."""
    grid, info = render_exact(data, song, nframes)
    grid[2, 24] = 14
    return grid, info


def test_scan_leads_matches_aligned_views():
    rng = np.random.default_rng(1)
    ref = rng.integers(0, 2, (12, 25)).astype(np.int16)
    lib = rng.integers(0, 2, (12 + survey.LEAD, 25)).astype(np.int16)
    lib[3:15] = ref
    lib[7, 9] ^= 1
    leads, ne, size, first, nbad = survey.scan_leads(ref, lib)
    assert list(leads[:3]) == [0, -1, 1] and sorted(leads) == list(range(-8, 9))
    for i, lead in enumerate(leads):
        want = np.not_equal(*survey.aligned(ref, lib, lead))
        assert size[i] == len(want) and (ne[i, : size[i]] == want).all()
        assert not ne[i, size[i] :].any()
        rows = want.any(1)
        assert nbad[i] == rows.sum()
        assert first[i] == (np.flatnonzero(rows)[0] if rows.any() else len(rows))
    stats = survey.min_lead((leads, ne, size, first, nbad))
    assert stats == dict(min_lead=3, min_bad=1, min_first=4, min_regs="v2_pw_lo")


def test_check_frame0_fault_min_lead(survey_tree):
    row = survey.check(_job(survey_tree, render_frame0))
    assert (row["verdict"], row["lead"], row["first_diff"]) == ("differs", 0, 1)
    assert row["diff_frames"] > NFRAMES // 2
    assert (row["min_lead"], row["min_bad"], row["min_first"]) == (2, 1, 0)
    assert row["min_regs"] == "mode_vol"
    exact = survey.check(_job(survey_tree, render_exact))
    assert (exact["min_lead"], exact["min_bad"], exact["min_first"]) == (2, 0, "")


def _near_lines(out):
    lines = out.splitlines()
    i = next(i for i, ln in enumerate(lines) if ln.startswith("near-exact"))
    return lines[i : i + 2]


def test_report_and_triage_surface_near_exact(survey_tree, capsys):
    hvsc, tunes, work, tune = survey_tree
    survey.run(hvsc, tunes, work, render_frame0, jobs=1)
    rows = survey.load_tunes(work / "results.csv")
    assert [(r["min_lead"], r["min_bad"], r["min_regs"]) for r in rows] == [
        ("", "", ""),
        ("2", "1", "mode_vol"),
    ]
    capsys.readouterr()
    survey.main(["report", str(work / "results.csv")])
    head, line = _near_lines(capsys.readouterr().out)
    assert head.endswith(": 1")
    assert line.split()[:5] == ["1", "bad=1", "first=0", "lead=2", "mode_vol"]
    survey.main(["report", str(work / "results.csv"), "--near", "0"])
    assert _near_lines(capsys.readouterr().out)[0].endswith(": 0")
    survey.main(["triage", str(work), "--jobs", "1"])
    head, line = _near_lines(capsys.readouterr().out)
    assert head.endswith(": 1") and "mode_vol" in line
    (trow,) = survey.load_tunes(work / "triage.csv")
    assert (trow["key"], trow["first_diff"], trow["min_lead"]) == (
        f"{tune['md5']}_2",
        "1",
        "2",
    )
    assert (trow["min_bad"], trow["min_first"], trow["min_regs"]) == (
        "1",
        "0",
        "mode_vol",
    )


def test_old_results_without_min_columns(survey_tree, capsys):
    hvsc, tunes, work, _ = survey_tree
    survey.run(hvsc, tunes, work, render_frame0, jobs=1)
    rows = survey.load_tunes(work / "results.csv")
    old = [f for f in survey.RESULT_FIELDS if not f.startswith("min_")]
    with open(work / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, old, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    capsys.readouterr()
    verdicts = survey.report(work / "results.csv")
    assert verdicts[("GoatTracker_V2.x", "differs")] == 1
    assert _near_lines(capsys.readouterr().out)[0].endswith(": 0")
    (t,), _ = survey.triage(work, jobs=1)
    assert (t["min_lead"], t["min_bad"], t["min_regs"]) == (2, 1, "mode_vol")
    assert _near_lines(capsys.readouterr().out)[0].endswith(": 1")


def test_leads_one_key(survey_tree, capsys):
    hvsc, tunes, work, tune = survey_tree
    key = f"{tune['md5']}_2"
    args = ["leads", str(hvsc), str(tunes), key, "--work", str(work)]
    with pytest.raises(ValueError, match="no --render"):
        survey.main(args)
    survey.main(args + ["--render", f"{__name__}:render_frame0"])
    out = capsys.readouterr().out.splitlines()
    assert "lead 0 (*" in out[0] and "min_lead 2 (<" in out[0]
    table = {int(ln[:4]): ln[4:].split() for ln in out[2:]}
    assert sorted(table) == list(range(-8, 9))
    assert table[0][:2] == ["*", "1"] and table[2] == ["<", "0", "1", "mode_vol"]
    assert (work / "library" / f"{key}.csv.zst").exists()
    survey.main(args)
    assert capsys.readouterr().out.splitlines() == out


def test_leads_all_and_verdict(survey_tree, capsys):
    hvsc, tunes, work, tune = survey_tree
    survey.run(hvsc, tunes, work, render_frame0, jobs=1)
    base = ["leads", str(hvsc), str(tunes), "--work", str(work), "--jobs", "1"]
    capsys.readouterr()
    survey.main(base + ["--all"])
    out = capsys.readouterr().out
    assert "lead != min_lead: [((0, 2), 1)]" in out
    assert _near_lines(out)[0].endswith(": 1")
    rows = survey.load_tunes(work / "leads.csv")
    assert {r["key"] for r in rows} == {f"{tune['md5']}_2"}
    assert [int(r["lead"]) for r in rows] == list(range(-8, 9))
    by = {int(r["lead"]): r for r in rows}
    assert (by[2]["first_diff"], by[2]["bad"], by[2]["regs"]) == ("0", "1", "mode_vol")
    assert (by[0]["first_diff"], by[0]["frames"]) == ("1", str(NFRAMES))
    out_csv = work / "x.csv"
    survey.main(base + ["--verdict", "exact", "--out", str(out_csv)])
    assert survey.load_tunes(out_csv) == []
    with pytest.raises(SystemExit):
        survey.main(base)


# --- survey match -------------------------------------------------------------


def match_body(raw):
    """A raw-byte body hook: the whole input as the (96-wide) vector."""
    vec = np.zeros(96, np.uint8)
    vec[: len(raw)] = np.frombuffer(raw, np.uint8)
    return vec, len(raw)


def _rawbody(width):
    def body(raw):
        vec = np.zeros(width, np.uint8)
        vec[: len(raw)] = np.frombuffer(raw, np.uint8)
        return vec, len(raw)

    return body


def test_image_body_rebasing_to_load():
    img = bytes(range(256))
    for sid in (build_psid(img, 0x2000), build_psid_embedded_load(img, 0x2000)):
        vec, length = survey.image_body(sid)
        assert length == 256 and vec[:256].tobytes() == img
        assert vec.shape == (survey.IMAGE_WIDTH,) and vec[256:].sum() == 0


def test_match_images_requires_candidates():
    with pytest.raises(ValueError, match="no candidate"):
        survey.match_images([], [b"xy"], body=match_body)


def test_match_images_exact_and_mutated():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, 100).astype(np.uint8).tobytes()
    other = rng.integers(0, 256, 64).astype(np.uint8).tobytes()
    body = _rawbody(128)
    rows = survey.match_images([img, other], [img], body=body)
    assert rows == [dict(candidate=0, length=100, ndiff=0, offset=0)]
    mut = bytearray(img)
    mut[3] ^= 0x21
    mut[77] ^= 0xC0
    mut[99] ^= 0x41
    rows = survey.match_images([img, other], [bytes(mut)], body=body)
    assert rows == [dict(candidate=0, length=100, ndiff=3, offset=0)]


def test_match_images_relocates_candidates():
    rng = np.random.default_rng(1)
    img = rng.integers(1, 256, 90, dtype=np.uint8).tobytes()
    body = _rawbody(128)
    rows = survey.match_images([b"\x00" * 7 + img], [img], body=body)
    assert rows == [dict(candidate=0, length=90, ndiff=0, offset=7)]
    rows = survey.match_images([img[5:]], [img], body=body)
    assert rows == [dict(candidate=0, length=90, ndiff=0, offset=-5)]


def test_match_images_length_guard():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, 90).astype(np.uint8)
    long_ = (
        img.astype(np.uint8).tobytes()
        + rng.integers(0, 256, 600).astype(np.uint8).tobytes()
    )
    close = img.copy()
    close[:40] = (close[:40].astype(np.int16) + 1) % 256
    rows = survey.match_images([long_, close], [img], body=_rawbody(0x400))
    assert rows[0] == dict(candidate=1, length=90, ndiff=40, offset=0)


def _match_tree(tmp_path):
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, 150).astype(np.uint8).tobytes()
    other = rng.integers(0, 256, 150).astype(np.uint8).tobytes()
    hvsc = tmp_path / "hvsc"
    hvsc.mkdir()
    tunes = {}
    for rel, blob in [("MUSICIANS/A/s1.sid", img), ("MUSICIANS/B/s2.sid", other)]:
        data = build_psid(blob, 0x1000)
        (hvsc / rel).parent.mkdir(parents=True, exist_ok=True)
        (hvsc / rel).write_bytes(data)
        tunes[rel] = data
    tunes_csv = tmp_path / "tunes.csv"
    with open(tunes_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, survey.TUNE_FIELDS)
        w.writeheader()
        for rel, data in tunes.items():
            w.writerow(
                dict(
                    path=rel,
                    md5=hashlib.md5(data).hexdigest(),
                    player="x",
                    songs="1",
                    clock="PAL",
                    lengths="1",
                )
            )
    return hvsc, tunes_csv, img, other


def test_match_cli(tmp_path, capsys):
    hvsc, tunes_csv, img, other = _match_tree(tmp_path)
    native = tmp_path / "native.sid"
    native.write_bytes(build_psid(img, 0x7000))
    foreign = tmp_path / "foreign.sid"
    foreign.write_bytes(build_psid(other, 0x7000))
    assert (
        survey.main(["match", str(hvsc), str(tunes_csv), str(native), str(foreign)])
        == 0
    )
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "native\thvsc\tlength\tndiff\toffset"
    assert lines[1].split("\t")[:2] == [str(native), "MUSICIANS/A/s1.sid"]
    assert lines[1].split("\t")[2:] == ["150", "0", "0"]
    assert lines[2].split("\t")[1] == "MUSICIANS/B/s2.sid"


def test_match_cli_out_and_body(tmp_path, capsys):
    hvsc = tmp_path / "hvsc"
    hvsc.mkdir()
    hvsc.joinpath("a.sid").write_bytes(b"\x11" * 80)
    hvsc.joinpath("b.sid").write_bytes(b"\x22" * 70)
    tunes_csv = tmp_path / "tunes.csv"
    with open(tunes_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, survey.TUNE_FIELDS)
        w.writeheader()
        for rel in ("a.sid", "b.sid"):
            w.writerow(
                dict(path=rel, md5="m", player="p", songs="1", clock="PAL", lengths="1")
            )
    native = tmp_path / "n.sid"
    native.write_bytes(b"\x11" * 80)
    out = tmp_path / "work" / "match.tsv"
    assert (
        survey.main(
            [
                "match",
                str(hvsc),
                str(tunes_csv),
                str(native),
                "--body",
                f"{__name__}:match_body",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    lines = out.read_text("utf-8").splitlines()
    assert lines[0] == "native\thvsc\tlength\tndiff\toffset"
    assert lines[1].split("\t")[1:] == ["a.sid", "80", "0", "0"]
    assert capsys.readouterr().out == ""


def test_run_kills_a_render_spinning_in_c_at_its_deadline(survey_tree):
    hvsc, tunes, work, _ = survey_tree
    tally = survey.run(hvsc, tunes, work, render_spin, jobs=2, timeout=1)
    rows = survey.load_tunes(work / "results.csv")
    assert tally == {"library_error": len(rows)}
    assert {r["error"] for r in rows} == {"TimeoutError: exceeded 1s"}
    assert all(r["frames"] for r in rows)


def _scope_rows(out):
    with open(out, newline="", encoding="utf-8") as f:
        return [(r["song"], r["verdict"], r["error"]) for r in csv.DictReader(f)]


def test_scope_classifies_each_subtune_without_the_oracle(survey_tree, tmp_path):
    hvsc, tunes, _, tune = survey_tree
    out = tmp_path / "scope.csv"
    tally = survey.scope(hvsc, tunes, render_exact, out=out, frames=3, jobs=1)
    assert tally == {"ok": 1, "library_error": 1}
    with open(out, newline="", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["song"])
    assert [r["key"] for r in rows] == [f"{tune['md5']}_{s}" for s in (1, 2)]
    assert "AssertionError" in rows[0]["error"] and rows[1]["verdict"] == "ok"
    assert json.loads(rows[1]["info"]) == {"variant": "test", "nframes": 3}
    survey.scope(hvsc, tunes, render_parse_error, out=out, jobs=1, player="nomatch")
    assert not _scope_rows(out)


@pytest.mark.parametrize(
    "render,verdict,error",
    [
        (render_parse_error, "not_decompiled", "cannot decompile"),
        (render_crash, "library_error", "KeyError"),
        (render_short, "library_error", "ValueError: render grid shape"),
    ],
)
def test_scope_check_failures(survey_tree, render, verdict, error):
    hvsc, _, _, tune = survey_tree
    job = (hvsc / tune["path"], tune, 2, 4, render, 0)
    row = survey.scope_check(job)
    assert row["verdict"] == verdict and error in row["error"]
    assert row["key"] == f"{tune['md5']}_2" and "info" not in row


def test_scope_cli_loads_a_renderer_file_and_times_out(survey_tree, tmp_path, capsys):
    hvsc, tunes, _, _ = survey_tree
    spec = tmp_path / "oracle_hvsc.py"
    spec.write_text(
        "def render(data, song, nframes):\n"
        "    while song == 2:\n"
        "        pass\n"
        "    return [[0] * 25] * nframes, {}\n",
        "utf-8",
    )
    out = tmp_path / "scope.csv"
    args = ["scope", str(hvsc), str(tunes), "--render", str(spec), "--out", str(out)]
    assert survey.main(args + ["--jobs", "2", "--timeout", "0.5", "--frames", "5"]) == 0
    assert sorted(_scope_rows(out)) == [
        ("1", "ok", ""),
        ("2", "library_error", "TimeoutError: exceeded 0.5s"),
    ]
    assert "'ok': 1" in capsys.readouterr().err


def test_scope_records_a_vanished_tune_as_a_library_error(survey_tree, tmp_path):
    hvsc, tunes, _, tune = survey_tree
    (hvsc / tune["path"]).unlink()
    out = tmp_path / "scope.csv"
    survey.scope(hvsc, tunes, render_exact, out=out, jobs=2)
    assert {v for _, v, _ in _scope_rows(out)} == {"library_error"}
    assert all("FileNotFoundError" in e for _, _, e in _scope_rows(out))


def _results(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text(
        "key,path,song,verdict,min_bad\n"
        + "".join(f"{k},p{k},1,{v},{b}\n" for k, v, b in rows),
        encoding="utf-8",
    )
    return path


def test_compare_moves_verdicts_and_lists_lost_exacts(tmp_path, capsys):
    base = _results(
        tmp_path,
        "base.csv",
        [("a", "exact", ""), ("b", "differs", "3"), ("c", "exact", "")],
    )
    new = _results(
        tmp_path,
        "new.csv",
        [
            ("a", "exact", ""),
            ("b", "exact", ""),
            ("c", "differs", "1"),
            ("d", "exact", ""),
            ("e", "differs", "9"),
        ],
    )
    keys = tmp_path / "keys.txt"
    keys.write_text("b c e z\n", encoding="utf-8")
    assert survey.main(["compare", str(new), str(keys), "--base", str(base)]) == 0
    out = capsys.readouterr().out
    assert "| differs | exact | 1 |" in out
    assert "| missing | exact | 1 |" in out
    assert "exact -> non-exact: 1" in out and "  c pc 1 differs" in out
    assert "{'differs': 2, 'exact': 1, 'missing': 1}" in out
    assert "differs by min_bad (4 = more): {1: 1, 4: 1}" in out
    moves, lost = survey.transitions(
        survey.read_results(base), survey.read_results(new)
    )
    assert moves[("exact", "exact")] == 1 and lost == ["c"]
    verdicts, bad = survey.tally(survey.read_results(new), ["e"], near=9)
    assert verdicts == {"differs": 1} and bad == {9: 1}


def test_compare_without_baseline_only_tallies(tmp_path, capsys):
    new = _results(tmp_path, "new.csv", [("a", "exact", "")])
    keys = tmp_path / "keys.txt"
    keys.write_text("a\n", encoding="utf-8")
    assert not survey.compare_results(new, [keys])
    out = capsys.readouterr().out
    assert "exact -> non-exact" not in out and "{'exact': 1}" in out


def _clear_cache(work, barrier, queue):
    barrier.wait()
    try:
        survey.oracle_cache(work)
        queue.put(None)
    except Exception as exc:  # pylint: disable=broad-except
        queue.put(repr(exc))


def test_oracle_cache_concurrent_clear(tmp_path, monkeypatch):
    work, workers = tmp_path / "work", 8
    stale = work / "oracle"
    stale.mkdir(parents=True)
    for k in range(500):
        (stale / f"{k}_1.csv.zst").write_bytes(b"")
    monkeypatch.setattr(survey, "sidtrace_image_id", lambda: "sha256:new")
    ctx = multiprocessing.get_context("fork")
    barrier, queue = ctx.Barrier(workers), ctx.Queue()
    procs = [
        ctx.Process(target=_clear_cache, args=(work, barrier, queue))
        for _ in range(workers)
    ]
    for proc in procs:
        proc.start()
    errors = [queue.get(timeout=60) for _ in procs]
    for proc in procs:
        proc.join()
    assert errors == [None] * workers
    assert sorted(p.name for p in stale.iterdir()) == ["STAMP"]
    assert json.loads((stale / "STAMP").read_text("utf-8"))["image"] == "sha256:new"


def _write_results(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, survey.RESULT_FIELDS)
        w.writeheader()
        for key, verdict, first in rows:
            w.writerow(
                dict(key=key, path=f"{key}.sid", verdict=verdict, first_diff=first)
            )
    return survey.read_results(path)


def test_changes_lists_moved_verdicts_and_first_diffs(tmp_path, capsys):
    base = tmp_path / "base.csv"
    head = tmp_path / "head.csv"
    _write_results(
        base,
        [("a_1", "exact", ""), ("b_1", "differs", "5"), ("c_1", "differs", "7")],
    )
    _write_results(
        head,
        [("a_1", "differs", "3"), ("b_1", "differs", "9"), ("c_1", "differs", "7")],
    )
    rows = survey.changes_cli(base, head)
    assert rows == [
        ("a_1", "exact", "differs", "", "3"),
        ("b_1", "differs", "differs", "5", "9"),
    ]
    out = capsys.readouterr().out
    assert "a_1 a_1.sid: exact -> differs" in out
    assert "b_1 b_1.sid: differs -> differs first_diff 5 -> 9" in out
    assert "2 of 3 subtunes changed" in out
