"""Unit tests for the sidtrace oracle helpers (no Docker required)."""

import subprocess
from pathlib import Path

import pytest
import zstandard

from pysidtracker import (
    SidtraceRow,
    SidtraceUnavailable,
    grid_from_writes,
    read_sidtrace,
    register_grid,
    run_sidtrace,
    sidtrace_cadence,
    sidtrace_grid,
)
from pysidtracker import oracle, registers
from pysidtracker.testing import make_oracle_fixtures, oracle_grid
from pysidtracker.testing import whole_song

from .helpers import build_psid, event, fake_sidtrace, handled, paced_trace, trace_csv

_HEADER = ",".join(oracle.SIDTRACE_COLUMNS)

# A minimal CSV: a pre-IRQ volume write, then one handled CIA raise per reg-0 write.
_CSV = (
    f"{_HEADER}\n"
    "100,,,,0,24,15\n"
    "19706,,,0,,,cia_irq\n"
    "19713,,,7,,,cia_irq_entry\n"
    "19756,,,50,0,0,10\n"
    "19800,,,94,,,rti\n"
    "39352,,,0,,,cia_irq\n"
    "39359,,,7,,,cia_irq_entry\n"
    "39412,,,60,0,0,20\n"
    "39450,,,98,,,rti\n"
    "59013,,,0,,,cia_irq\n"
    "59020,,,7,,,cia_irq_entry\n"
    "59068,,,55,0,0,30\n"
    "59100,,,87,,,rti\n"
)


def _irqs(rows):
    """``rows`` with each timed write's interrupt raised, entered and returned from.

    The handler returns after the last write timed by its raise.
    """
    ends = {}
    for r in rows:
        since = r.since_video_irq if r.since_video_irq is not None else r.since_cia_irq
        if since is not None:
            key = (r.cycle - since, "video_irq" if r.since_video_irq else "cia_irq")
            ends[key] = max(ends.get(key, 0), r.cycle)
    out = list(rows)
    for (at, source), end in ends.items():
        out += [event(at, source), event(at, f"{source}_entry"), event(end, "rti")]
    return sorted(out, key=lambda r: (r.cycle, r.value == "rti", r.chip is not None))


# Three clean CIA frames (raise cycles 19656, 39312, 58968 -> cadence 19656).
_ROWS = _irqs(
    [
        SidtraceRow(19656, None, None, 0, 0, 0, 10),
        SidtraceRow(39312, None, None, 0, 0, 0, 20),
        SidtraceRow(58968, None, None, 0, 0, 0, 30),
    ]
)


def _zst(text: str) -> bytes:
    return zstandard.ZstdCompressor().compress(text.encode("utf-8"))


def test_read_sidtrace_plain_csv(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text(_CSV)
    rows = read_sidtrace(path)
    assert rows[0] == SidtraceRow(100, None, None, None, 0, 24, 15)
    assert rows[1] == SidtraceRow(19706, None, None, 0, None, None, "cia_irq")
    assert rows[2] == SidtraceRow(19713, None, None, 7, None, None, "cia_irq_entry")
    assert rows[3] == SidtraceRow(19756, None, None, 50, 0, 0, 10)
    assert rows[4] == SidtraceRow(19800, None, None, 94, None, None, "rti")
    assert len(rows) == 13


def test_read_sidtrace_zst_matches_plain(tmp_path):
    zpath = tmp_path / "t.csv.zst"
    zpath.write_bytes(_zst(_CSV))
    ppath = tmp_path / "t.csv"
    ppath.write_text(_CSV)
    assert read_sidtrace(zpath) == read_sidtrace(ppath)


def test_read_sidtrace_skips_blank_and_header(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text(_HEADER + "\n\n100,,,,0,0,1\n")
    assert read_sidtrace(path) == [SidtraceRow(100, None, None, None, 0, 0, 1)]


def test_sidtrace_cadence_median_irq_interval():
    assert sidtrace_cadence(_ROWS) == 19656


def test_sidtrace_cadence_needs_interrupt_rows():
    with pytest.raises(ValueError, match="no interrupt rows"):
        sidtrace_cadence([SidtraceRow(0, None, None, None, 0, 0, 1)])
    assert sidtrace_cadence(_irqs([SidtraceRow(9, None, None, 2, 0, 0, 1)])) is None


def test_sidtrace_cadence_prefers_video_over_cia():
    rows = _irqs(
        [
            SidtraceRow(20000, None, 100, 999, 0, 0, 1),
            SidtraceRow(39656, None, 100, 999, 0, 0, 2),
        ]
    ) + [SidtraceRow(n * 7000, None, None, 0, None, None, "cia_irq") for n in (3, 4)]
    rows.sort(key=lambda r: r.cycle)
    # video-raise 19900, 39556 -> 19656 (cia raises ignored).
    assert sidtrace_cadence(rows) == 19656


def _row(cycle, since, register, value):
    return SidtraceRow(cycle, None, None, since, 0, register, value)


def test_sidtrace_grid_frames_by_interrupt():
    assert [row[0] for row in sidtrace_grid(_ROWS)] == [10, 20, 30]


def test_init_writes_are_the_baseline_however_close_the_first_interrupt():
    rows = _irqs(
        [_row(100, None, 1, 7), _row(6000, 200, 0, 10), _row(25856, 200, 0, 20)]
    )
    assert sidtrace_grid(rows, reg_count=2, cycles_per_frame=19656) == [
        [10, 7],
        [20, 7],
    ]


def test_fixed_cadence_counts_calls_between_raises():
    rows = _irqs(
        [
            _row(1000, 0, 0, 1),
            _row(2000, 0, 0, 2),
            _row(2000 + 3 * 19656, 0, 0, 3),
        ]
    )
    assert sidtrace_grid(rows, reg_count=1, cycles_per_frame=19656) == [
        [1],
        [2],
        [2],
        [2],
        [3],
    ]


def test_a_write_outside_an_interrupt_joins_the_last_one_raised():
    rows = _irqs([_row(1000, 0, 0, 1), _row(1500, None, 0, 2), _row(20656, 0, 0, 3)])
    assert sidtrace_grid(rows, reg_count=1, cycles_per_frame=19656) == [[2], [3]]


def test_grid_from_writes_anchors_after_the_init_gap():
    writes = [(0, 0, 9), (50000, 0, 1), (69656, 0, 2)]
    assert grid_from_writes(writes, reg_count=1) == [[1], [2]]


def test_sidtrace_grid_filters_by_chip():
    rows = _irqs(
        [
            SidtraceRow(20000, None, 100, None, 0, 0, 11),
            SidtraceRow(20000, None, 100, None, 1, 0, 99),
            SidtraceRow(39656, None, 100, None, 0, 0, 22),
        ]
    )
    assert all(row[0] != 99 for row in sidtrace_grid(rows, chip=0))


def test_register_grid_seeds_driver_volume():
    # init at $1000: LDA #$0A / STA $D400 / RTS ; play just RTS (never sets vol).
    image = bytes([0xA9, 0x0A, 0x8D, 0x00, 0xD4, 0x60, 0x60])
    tune = build_psid(image, load=0x1000, init=0x1000, play=0x1006)
    grid = register_grid(tune, whole_song(2))
    assert grid[0][24] == 0x0F  # PSID driver max volume, not 0
    assert grid[0][0] == 0x0A


def _fake_docker(csv_text):
    def _run(cmd, check, stdout, stderr):  # pylint: disable=unused-argument
        work = Path(cmd[cmd.index("-v") + 1].split(":")[0])
        (work / "trace.csv.zst").write_bytes(_zst(csv_text))
        return subprocess.CompletedProcess(cmd, 0)

    return _run


def test_run_sidtrace_moves_output(tmp_path, monkeypatch):
    monkeypatch.setattr(oracle.subprocess, "run", _fake_docker(_CSV))
    tune = tmp_path / "song.sid"
    tune.write_bytes(b"PSID" + b"\x00" * 60)
    out = run_sidtrace(tune, tmp_path / "cache" / "song.csv.zst", seconds=2)
    assert out.exists() and read_sidtrace(out)[0].value == 15
    # The private mount dir is cleaned up.
    assert not list((tmp_path / "cache").glob(".sidtrace-*"))


def test_run_sidtrace_missing_docker(tmp_path, monkeypatch):
    def _boom(*_a, **_k):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(oracle.subprocess, "run", _boom)
    tune = tmp_path / "song.sid"
    tune.write_bytes(b"PSID" + b"\x00" * 60)
    with pytest.raises(SidtraceUnavailable, match="not found"):
        run_sidtrace(tune, tmp_path / "o.csv.zst")


def test_run_sidtrace_render_failure(tmp_path, monkeypatch):
    def _fail(cmd, check, stdout, stderr):  # pylint: disable=unused-argument
        raise subprocess.CalledProcessError(1, cmd, stderr=b"boom")

    monkeypatch.setattr(oracle.subprocess, "run", _fail)
    tune = tmp_path / "song.sid"
    tune.write_bytes(b"PSID" + b"\x00" * 60)
    with pytest.raises(SidtraceUnavailable, match="boom"):
        run_sidtrace(tune, tmp_path / "o.csv.zst")


def test_oracle_grid_uses_cache(tmp_path, monkeypatch):
    cache = tmp_path / "csv"
    cache.mkdir()
    (cache / "song.csv.zst").write_bytes(_zst(_CSV))
    tune = tmp_path / "song.sid"
    tune.write_bytes(b"PSID" + b"\x00" * 60)

    def _no_render(*_a, **_k):
        raise AssertionError("should not re-render on a cache hit")

    monkeypatch.setattr(oracle, "run_sidtrace", _no_render)
    rows = read_sidtrace(cache / "song.csv.zst")
    calls = oracle.sidtrace_calls(rows)
    want = sidtrace_grid(rows)[:calls]
    assert oracle_grid(tune, oracle_cache=cache, frames=calls) == want


def test_oracle_grid_reuses_a_trace_without_rti_rows(tmp_path, monkeypatch):
    # A raise with entry but no `rti` is a valid never-returning tune, not a
    # stale trace: it is reused, never re-rendered (which cannot add an `rti`).
    cache = tmp_path / "csv"
    cache.mkdir()
    head, *body = _CSV.splitlines()
    kept = [
        l for l in body if l[-1].isdigit() or l.endswith("irq") or l.endswith("entry")
    ]
    stale = "\n".join([head] + kept) + "\n"
    (cache / "song.csv.zst").write_bytes(_zst(stale))
    tune = tmp_path / "song.sid"
    tune.write_bytes(b"PSID" + b"\x00" * 60)

    def _no_render(*_a, **_k):
        raise AssertionError("a never-returning trace must be reused, not re-rendered")

    monkeypatch.setattr(oracle.subprocess, "run", _no_render)
    rows = read_sidtrace(cache / "song.csv.zst")
    calls = oracle.sidtrace_calls(rows)
    assert not oracle.nested(rows)
    want = sidtrace_grid(rows)[:calls]
    assert oracle_grid(tune, oracle_cache=cache, frames=calls) == want


def test_oracle_grid_still_raises_without_interrupt_rows(tmp_path):
    # A trace with no interrupt at all is unplayable, not merely un-nested.
    cache = tmp_path / "csv"
    cache.mkdir()
    head, *body = _CSV.splitlines()
    writes = "\n".join([head] + [l for l in body if l[-1].isdigit()]) + "\n"
    (cache / "song.csv.zst").write_bytes(_zst(writes))
    tune = tmp_path / "song.sid"
    tune.write_bytes(b"PSID" + b"\x00" * 60)
    with pytest.raises(ValueError, match="no interrupt rows"):
        oracle_grid(tune, oracle_cache=cache, frames=whole_song(0))


def test_oracle_grid_renders_on_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(oracle.subprocess, "run", _fake_docker(_CSV))
    tune = tmp_path / "song.sid"
    tune.write_bytes(b"PSID" + b"\x00" * 60)
    grid = oracle_grid(
        tune, oracle_cache=tmp_path / "csv", seconds=whole_song(2), frames=whole_song(2)
    )
    assert (tmp_path / "csv" / "song.csv.zst").exists() and len(grid) == 2


def test_oracle_grid_plays_the_given_song(tmp_path, monkeypatch):
    seen = []

    def _run(cmd, **kw):
        seen.append(cmd)
        return _fake_docker(_CSV)(cmd, **kw)

    monkeypatch.setattr(oracle.subprocess, "run", _run)
    tune = tmp_path / "song.sid"
    tune.write_bytes(b"PSID" + b"\x00" * 60)
    for song in (None, 1, 2):
        oracle_grid(
            tune,
            oracle_cache=tmp_path / "csv",
            seconds=whole_song(2),
            frames=whole_song(2),
            song=song,
        )
    names = sorted(p.name for p in (tmp_path / "csv").glob("*.csv.zst"))
    assert names == ["song.csv.zst", "song_1.csv.zst", "song_2.csv.zst"]
    songs = [[a for a in cmd if a.startswith("-o")] for cmd in seen]
    assert songs == [[], ["-o1"], ["-o2"]]


def test_make_oracle_fixtures_returns_two_fixtures():
    tune_id, oracle_match = make_oracle_fixtures(
        {"x": "A/x.sid"}, hvsc_cache="/tmp/h", oracle_cache="/tmp/o"
    )
    assert tune_id is not None and oracle_match is not None


def _jch_rows():
    """VIC-timed calls every PAL frame; only every third call writes."""
    frame = registers.PAL_CYCLES_PER_FRAME
    rows = []
    for n in range(1, 13):
        at = n * frame + 64
        rows.append(SidtraceRow(at, None, 0, None, None, None, "video_irq"))
        rows.append(SidtraceRow(at + 7, None, 7, None, None, None, "video_irq_entry"))
        if n % 3 == 0:
            rows.append(SidtraceRow(at + 40, None, 40, None, 0, 0, n))
        rows.append(SidtraceRow(at + 60, None, 60, None, None, None, "rti"))
    return rows


def test_read_sidtrace_interrupt_rows(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text(
        _HEADER + "\n100,,0,,,,video_irq\n107,,7,,,,video_irq_entry\n"
        "140,,40,,0,24,15\n141,,41,,,,brk_entry\n150,,50,,,,rti\n160,,60,,,,rti\n"
    )
    assert read_sidtrace(path) == [
        SidtraceRow(100, None, 0, None, None, None, "video_irq"),
        SidtraceRow(107, None, 7, None, None, None, "video_irq_entry"),
        SidtraceRow(140, None, 40, None, 0, 24, 15),
        SidtraceRow(141, None, 41, None, None, None, "brk_entry"),
        SidtraceRow(150, None, 50, None, None, None, "rti"),
        SidtraceRow(160, None, 60, None, None, None, "rti"),
    ]


def test_interrupt_rows_frame_every_call():
    frame = registers.PAL_CYCLES_PER_FRAME
    assert sidtrace_cadence(_jch_rows()) == frame
    grid = sidtrace_grid(_jch_rows(), reg_count=1)
    assert [g[0] for g in grid] == [0, 0, 3, 3, 3, 6, 6, 6, 9, 9, 9, 12]


def test_interrupt_rows_ignore_other_source():
    rows = _jch_rows() + [
        SidtraceRow(n * 7000, None, None, 0, None, None, "cia_irq")
        for n in range(1, 30)
    ]
    rows.sort(key=lambda r: r.cycle)
    assert sidtrace_cadence(rows) == registers.PAL_CYCLES_PER_FRAME


def _write(cycle, register, chip=0):
    return SidtraceRow(cycle, None, None, None, chip, register, 1)


def test_handler_sources_follow_entry_and_rti_nesting():
    rows = [
        _write(1, 0),
        event(10, "cia_irq"),
        event(17, "cia_irq_entry"),
        _write(20, 1),
        event(30, "nmi"),
        event(37, "nmi_entry"),
        _write(40, 24),
        event(50, "rti"),
        _write(55, 2),
        event(60, "brk_entry"),
        _write(65, 3),
        event(70, "rti"),
        _write(75, 4),
        event(80, "rti"),
        event(85, "rti"),
        _write(90, 5),
    ]
    owners = {
        r.reg: s for r, s in zip(rows, oracle.handler_sources(rows)) if r.chip == 0
    }
    assert owners == {
        0: None,
        1: "cia_irq",
        24: "nmi",
        2: "cia_irq",
        3: "cia_irq",
        4: "cia_irq",
        5: None,
    }


def test_handler_sources_unknown_without_rti_rows():
    # A handler that never returned leaves the source of every row unknown.
    rows = [
        event(10, "cia_irq"),
        event(17, "cia_irq_entry"),
        _write(20, 0),
        _write(30, 1),
    ]
    assert oracle.handler_sources(rows) == [None] * len(rows)
    assert oracle.play_source(rows) is None
    assert oracle.play_source(rows, min_raises=3) is None


def test_play_source_counts_the_chip_and_raise_floor():
    cia = handled("cia_irq", 100, [(110, 0, 1)])
    cia[-1:-1] = [_write(111 + r, r, chip=1) for r in range(3)]
    rows = handled("nmi", 10, [(20, 0, 1), (21, 1, 1)]) + cia
    rows += [event(200, "cia_irq"), event(300, "cia_irq")]
    assert oracle.play_source(rows) == "cia_irq"
    assert oracle.play_source(rows, chip=0) == "nmi"
    assert oracle.play_source(rows, chip=0, min_raises=2) == "cia_irq"
    assert oracle.play_source(rows, min_raises=4) is None
    assert oracle.raise_cycles(rows, "cia_irq") == [100, 200, 300]


def test_cached_sidtrace_force_rerenders(tmp_path, monkeypatch):
    cache = tmp_path / "song.csv.zst"
    cache.write_bytes(_zst(_CSV.replace("15\n", "14\n", 1)))
    monkeypatch.setattr(oracle.subprocess, "run", _fake_docker(_CSV))
    tune = tmp_path / "song.sid"
    tune.write_bytes(b"PSID" + b"\x00" * 60)
    assert oracle.cached_sidtrace(tune, cache)[0].value == 14
    assert oracle.cached_sidtrace(tune, cache, force=True)[0].value == 15


def test_sidtrace_calls_counts_complete_calls(tmp_path):
    (tmp_path / "t.csv").write_text(_CSV)
    rows = read_sidtrace(tmp_path / "t.csv")
    assert oracle.sidtrace_calls(rows) == 2
    assert oracle.sidtrace_calls(rows, cycles_per_frame=19656 // 2) == 4
    assert oracle.sidtrace_calls([r for r in rows if r.chip is not None]) == 0


def test_play_seconds_at_the_tune_cadence():
    data = build_psid(bytes([0x60, 0x60]), 0x1000, 0x1000, 0x1001, songs=2)
    pal = registers.PAL_CYCLES_PER_FRAME / registers.PAL_CLOCK_HZ
    assert oracle.play_seconds(data, 50) == pytest.approx(50 * pal)
    assert oracle.play_seconds(bytearray(data), 50, 2) == pytest.approx(50 * pal)
    assert oracle.render_seconds(50 * pal) == 2


def test_oracle_grid_repeats_the_last_write_but_never_pads_the_trace(
    tmp_path, monkeypatch
):
    frame = registers.PAL_CYCLES_PER_FRAME
    rows = [r for r in paced_trace(1, frame) if r.chip is None or r.cycle < 5 * frame]
    cache = tmp_path / "csv"
    cache.mkdir()
    (cache / "song.csv.zst").write_bytes(_zst(trace_csv(rows)))
    tune = tmp_path / "song.sid"
    tune.write_bytes(build_psid(bytes([0x60]), 0x1000))
    calls = oracle.sidtrace_calls(rows)
    grid = oracle_grid(tune, oracle_cache=cache, frames=calls)
    assert [g[0] for g in grid] == [0, 1, 2, 3, 4] + [4] * (calls - 5)
    renders = []
    monkeypatch.setattr(oracle, "run_sidtrace", fake_sidtrace(frame, renders, cap=1))
    with pytest.raises(oracle.ShortTrace, match=f"covers {calls} of {calls + 1}"):
        oracle_grid(tune, oracle_cache=cache, frames=calls + 1)
    assert renders == [oracle.render_seconds(oracle.play_seconds(tune, calls + 1))]


def test_oracle_grid_frames_a_skipped_call_only_at_a_fixed_cadence(tmp_path):
    frame = registers.PAL_CYCLES_PER_FRAME
    rows = paced_trace(1, frame)
    skipped = rows[40:44]
    rows = [r for r in rows if r not in skipped]
    cache = tmp_path / "csv"
    cache.mkdir()
    (cache / "song.csv.zst").write_bytes(_zst(trace_csv(rows)))
    tune = tmp_path / "song.sid"
    tune.write_bytes(build_psid(bytes([0x60]), 0x1000))
    calls = oracle.sidtrace_calls(rows, cycles_per_frame=frame)
    grid = oracle_grid(tune, oracle_cache=cache, frames=calls, cycles_per_frame=frame)
    assert [g[0] for g in grid[9:12]] == [9, 9, 11]
    with pytest.raises(oracle.ShortTrace, match=f"covers {calls - 1} of {calls}"):
        oracle_grid(tune, oracle_cache=cache, frames=calls)


def test_cached_sidtrace_without_interrupt_rows_covers_no_calls(tmp_path, monkeypatch):
    head, *body = _CSV.splitlines()
    writes = "\n".join([head] + [l for l in body if l[-1].isdigit()]) + "\n"
    monkeypatch.setattr(oracle.subprocess, "run", _fake_docker(writes))
    tune = tmp_path / "song.sid"
    tune.write_bytes(b"PSID" + b"\x00" * 60)
    with pytest.raises(oracle.ShortTrace, match="covers 0 of 1"):
        oracle.cached_sidtrace(tune, tmp_path / "song.csv.zst", calls=1)
