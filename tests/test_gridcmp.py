"""Grid-cmp core: serialization, lead scan, onset/carry cell diff, verdicts, reports, CLI."""

import json

import numpy as np
import pytest

from pysidtracker import gridcmp, survey
from pysidtracker.errors import IncompleteComparison

_U42_NAMES = [
    f"v{n} {s}"
    for n in (1, 2, 3)
    for s in ("freq lo", "freq hi", "pw lo", "pw hi", "ctrl", "ad", "sr")
] + ["cut lo", "cut hi", "res/route", "mode/vol"]


def test_carry_exempts_only_the_1lsb_freq_and_cutoff_pairs():
    ours = np.zeros((3, 25), np.uint8)
    theirs = ours.copy()
    ours[0, 0], theirs[0, 0] = 0x00, 0x01  # v1 freq +1
    ours[0, 1] = theirs[0, 1] = 0x05
    ours[0, 21], theirs[0, 21] = 0x00, 0x01  # cutoff +1
    ours[0, 22] = theirs[0, 22] = 0x03
    ours[1, 0], theirs[1, 0] = 0x00, 0x02  # a +2 delta is not the carry
    ours[2, 2], theirs[2, 2] = 0x00, 0x05  # a plain value difference, off the pairs
    ours[2, 7], theirs[2, 7] = 0x05, 0x04  # the other carry direction
    ours[2, 21], theirs[2, 21] = 0x04, 0x03  # the other cutoff direction
    mask = gridcmp.carry(ours, theirs)
    want = np.zeros((3, 25), bool)
    want[0, [0, 1, 21, 22]] = True
    want[2, [7, 8, 21, 22]] = True
    assert (mask == want).all()
    assert not gridcmp.carry(
        np.zeros((2, 14), np.uint8), np.zeros((2, 14), np.uint8)
    ).any()


def test_cell_diff_orders_cells_and_applies_onset_and_tolerance():
    ours, theirs = np.zeros((4, 25), np.uint8), np.zeros((4, 25), np.uint8)
    ours[1, 3], theirs[1, 3] = 1, 2
    ours[2, 0], theirs[2, 0] = (
        5,
        6,
    )  # a carry pair, exempted only with the carry tolerance
    assert gridcmp.cell_diff(ours, theirs).tolist() == [[1, 3, 1, 2], [2, 0, 5, 6]]
    assert gridcmp.cell_diff(ours, theirs, tol="carry").tolist() == [[1, 3, 1, 2]]
    assert gridcmp.cell_diff(ours, theirs, starts=[None, None, None]).size == 0
    skip = np.zeros((4, 25), bool)
    skip[2, 0] = True
    assert gridcmp.cell_diff(ours, theirs, tol=skip).tolist() == [[1, 3, 1, 2]]
    assert gridcmp.cell_diff(np.zeros((0, 25)), np.zeros((0, 25))).shape == (0, 4)


def test_first_mismatch_applies_the_onset_criterion_with_carry():
    ours, theirs = np.zeros((4, 25), np.uint8), np.zeros((4, 25), np.uint8)
    ours[1, 3], theirs[1, 3] = 9, 0
    ours[2, 0], theirs[2, 0] = 5, 6
    assert gridcmp.first_mismatch(ours, theirs, [0, None, None]) == (1, 3)
    assert gridcmp.first_mismatch(ours, theirs, [2, None, None]) is None
    assert gridcmp.first_mismatch(ours, theirs) == (1, 3)
    assert gridcmp.first_mismatch(ours, theirs, [2, None, None], tol=None) == (2, 0)


def test_runs_groups_registers_runs_and_value_segments():
    cells = sorted(
        [
            (1, 3, 9, 1),
            (2, 3, 9, 1),
            (3, 5, 1, 9),
            (7, 5, 8, 0),
            (7, 8, 0, 2),
            (8, 8, 4, 2),
            (2, 11, 0, 7),
            (3, 11, 5, 7),
        ]
    )
    out = gridcmp.runs(cells, [f"r{r}" for r in range(25)])
    assert [e["reg"] for e in out] == [3, 5, 8, 11]
    e3, e5, e8, e11 = out[0], out[1], out[2], out[3]
    assert (e3["from"], e3["to"], e3["cells"]) == (1, 2, 2)
    assert e3["runs"] == [{"from": 1, "to": 2, "values": [[1, 2, 9, 1]]}]
    assert (e5["name"], e5["from"], e5["to"], e5["cells"]) == ("r5", 3, 7, 2)
    assert e5["runs"] == [
        {"from": 3, "to": 3, "values": [[3, 3, 1, 9]]},
        {"from": 7, "to": 7, "values": [[7, 7, 8, 0]]},
    ]
    assert (e8["from"], e8["to"], e8["cells"]) == (7, 8, 2)
    assert e8["runs"] == [{"from": 7, "to": 8, "values": [[7, 7, 0, 2], [8, 8, 4, 2]]}]
    assert (e11["name"], e11["from"], e11["to"]) == ("r11", 2, 3)
    assert gridcmp.runs([]) == []


def test_verdict_onset_reports_onset_whole_and_early_end_like_the_lift_proof():
    theirs = (
        ((np.arange(8 * 25) + np.tile(np.arange(25) * 7, 8)) % 250)
        .astype(np.uint8)
        .reshape(8, 25)
    )
    starts = (2, 3, 4)
    ok = gridcmp.verdict_onset(theirs, starts, theirs)
    assert ok.ok and ok.played == 8 and ok.onset is None and ok.whole is None
    assert ok.line() == "verified: 8 ticks play the reference grid"
    assert (
        gridcmp.runs_report(ok, unit="tick", subject="document", a="doc", b="native")
        == "verified: 8 ticks play the native grid"
    )
    early = gridcmp.verdict_onset(theirs[:6], starts, theirs)
    assert not early.ok and early.onset is None and early.played == 6
    assert (
        gridcmp.runs_report(early, unit="tick", subject="document", a="doc", b="native")
        == "mismatch: the document ends at tick 6 of 8"
    )
    assert early.line() == "mismatch: the render ends at tick 6 of 8"

    ours = theirs.copy()
    ours[4, 5] = (ours[4, 5] + 1) & 0xFF  # v1 ad, inside voice 0's onset
    ours[1, 21] = (ours[1, 21] + 2) & 0xFF  # a global before the earliest note, delta 2
    ver = gridcmp.verdict_onset(ours, starts, theirs)
    assert (ver.onset, ver.whole) == ((4, 5), (1, 21)) and not ver.ok
    assert ver.cells == ((4, 5, int(ours[4, 5]), int(theirs[4, 5])),)
    assert ver.line() == "mismatch: tick 4, register $05 (from tick 0: tick 1, $15)"
    assert (
        gridcmp.runs_report(
            ver, names=_U42_NAMES, unit="tick", subject="document", a="doc", b="native"
        )
        == "mismatch: tick 4, register $05 v1 ad (from tick 0: tick 1, $15 cut lo)\n"
        "  $05 v1 ad: 1 cell, tick 4\n"
        f"    4: doc {ours[4, 5]:02x}, native {theirs[4, 5]:02x}"
    )
    summary = ver.summary(names=_U42_NAMES)
    assert (summary["ok"], summary["played"], summary["ended"]) == (False, 8, 0)
    assert (summary["starts"], summary["first"]) == ([2, 3, 4], None)
    assert (summary["onset"], summary["whole"], summary["cells"]) == (
        [4, 5],
        [1, 21],
        [[4, 5, int(ours[4, 5]), int(theirs[4, 5])]],
    )
    assert summary["registers"][0]["name"] == "v1 ad"


def test_runs_report_counts_the_value_lines_it_omits():
    theirs = np.zeros((10, 25), np.uint8)
    ours = np.zeros((10, 25), np.uint8)
    ours[np.arange(10), 0] = np.arange(
        2, 12
    )  # deltas of 2 or more: past the 1-LSB carry
    grid = gridcmp.verdict_onset(ours, [0], theirs)
    assert not grid.ok and grid.onset == (0, 0)
    text = gridcmp.runs_report(grid, shown=6)
    assert "  $00 v1_freq_lo: 10 cells, ticks 0-9" in text
    assert "    0: render 02, reference 00" in text
    assert "    5: render 07, reference 00" in text
    assert "    6:" not in text
    assert "    ... 4 more value lines" in text


def test_compare_grids_short_render_is_an_error():
    a = (np.arange(10 * 25) % 250 + 1).astype(np.uint8).reshape(10, 25)
    with pytest.raises(ValueError, match="shift 8"):
        gridcmp.compare_grids(a, np.zeros((8, 25)))
    with pytest.raises(ValueError, match="shift 8"):
        gridcmp.verdict_lead(np.zeros((4, 25), np.int16), np.zeros((4, 25), np.int16))
    with pytest.raises(IncompleteComparison, match="of a 11-frame song"):
        gridcmp.verdict_lead(a, np.vstack([a, a]), whole=11)
    with pytest.raises(IncompleteComparison, match="render 17"):
        gridcmp.verdict_lead(a, a[np.arange(17) % 10], whole=10)
    assert gridcmp.verdict_lead(a, np.vstack([a, a]), whole=10).ok
    b = np.vstack([np.zeros((2, 25)), a, a[:6]])
    out = gridcmp.compare_grids(a, b)
    assert (out["verdict"], out["lead"], out["frames"]) == ("exact", 2, 10)
    assert (out["min_lead"], out["min_bad"]) == (2, 0)


def test_read_grid_infers_frames_and_accepts_plain_csv(tmp_path):
    grid = (np.arange(4 * 25) % 250 + 1).astype(np.int16).reshape(4, 25)
    ref = tmp_path / "g.csv.zst"
    gridcmp.write_grid(grid, ref)
    assert gridcmp.read_grid(ref).dtype == np.int16
    assert (gridcmp.read_grid(ref) == grid).all()
    assert (
        gridcmp.read_grid(ref, 6) == np.vstack([grid, np.repeat(grid[-1:], 2, 0)])
    ).all()
    plain = tmp_path / "plain.csv"
    plain.write_text("frame,reg,value\n0,24,15\n3,24,14\n", encoding="utf-8")
    want = np.zeros((4, 25), np.int16)
    want[:3, 24] = 15
    want[3, 24] = 14
    assert (gridcmp.read_grid(plain) == want).all()
    empty = tmp_path / "empty.csv"
    empty.write_text("frame,reg,value\n", encoding="utf-8")
    assert gridcmp.read_grid(empty).shape == (1, 25)
    assert gridcmp.read_grid(empty, 5).shape == (5, 25)


@pytest.mark.strain_exempt(reason="exercises code outside the measured package")
def test_survey_reexports_the_shared_core():
    for name in (
        "LEAD",
        "NEAR",
        "NREGS",
        "ROLES",
        "REG_NAMES",
        "aligned",
        "classify",
        "compare",
        "compare_grids",
        "gated",
        "min_lead",
        "read_grid",
        "reg_names",
        "scan_leads",
        "verdict_fields",
        "write_grid",
    ):
        assert getattr(survey, name) is getattr(gridcmp, name)


def test_main_diffs_the_two_grid_files_in_lead_and_onset_modes(tmp_path, capsys):
    ref = (np.arange(12 * 25) % 200).astype(np.int16).reshape(12, 25)
    ref_path = tmp_path / "ref.csv.zst"
    gridcmp.write_grid(ref, ref_path)
    rend_path = tmp_path / "rend.csv.zst"
    gridcmp.write_grid(
        np.vstack([np.zeros((2, 25), np.int16), ref, ref[:6]]), rend_path
    )
    assert gridcmp.main([str(ref_path), str(rend_path)]) == 0
    assert "verified: 12 frames agree at lead 2" in capsys.readouterr().out

    rend_bad = np.vstack([np.zeros((2, 25), np.int16), ref, ref[:6]])
    rend_bad[9, 0] ^= 0x40
    bad_path = tmp_path / "bad.csv.zst"
    gridcmp.write_grid(rend_bad, bad_path)
    assert gridcmp.main([str(ref_path), str(bad_path)]) == 1
    out = capsys.readouterr().out
    assert "differs: frame 7, register $00 v1_freq_lo" in out and "^^" in out

    json_path = tmp_path / "work" / "d.json"
    assert gridcmp.main([str(ref_path), str(bad_path), "--json", str(json_path)]) == 1
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert (summary["ok"], summary["steps"], summary["first"]) == (False, 12, 7)
    assert summary["regs"] == ["v1_freq_lo"] and summary["min_bad"] == 1
    assert gridcmp.main([str(ref_path), str(bad_path), "--json", "-"]) == 1
    assert json.loads(capsys.readouterr().out)["steps"] == 12

    onset_bad = ref.copy()
    onset_bad[5, 0] ^= 0x40
    onset_path = tmp_path / "onset.csv.zst"
    gridcmp.write_grid(onset_bad, onset_path)
    args = [
        str(ref_path),
        str(onset_path),
        "--mode",
        "onset",
        "--onset",
        "1",
        "--onset",
        "1",
        "--onset",
        "1",
    ]
    assert gridcmp.main(args) == 1
    out = capsys.readouterr().out
    assert "mismatch: frame 5, register $00 v1_freq_lo" in out
    assert f"render {onset_bad[5, 0]:02x}, reference {ref[5, 0]:02x}" in out
