"""Unit tests for the sidtrace oracle helpers (no Docker required)."""

import subprocess
from pathlib import Path

import pytest
import zstandard

from pysidtracker import (
    SidtraceRow,
    SidtraceUnavailable,
    read_sidtrace,
    run_sidtrace,
)
from pysidtracker import oracle, registers
from pysidtracker.testing import oracle_grid
from pysidtracker.testing import whole_song

from .helpers import build_psid, event, fake_sidtrace, paced_trace, trace_csv

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


def test_read_sidtrace_skips_blank_and_header(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text(_HEADER + "\n\n100,,,,0,0,1\n")
    assert read_sidtrace(path) == [SidtraceRow(100, None, None, None, 0, 0, 1)]


def _row(cycle, since, register, value):
    return SidtraceRow(cycle, None, None, since, 0, register, value)


def _fake_docker(csv_text):
    def _run(cmd, check, stdout, stderr):  # pylint: disable=unused-argument
        work = Path(cmd[cmd.index("-v") + 1].split(":")[0])
        (work / "trace.csv.zst").write_bytes(_zst(csv_text))
        return subprocess.CompletedProcess(cmd, 0)

    return _run


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


def test_sidtrace_calls_counts_complete_calls(tmp_path):
    (tmp_path / "t.csv").write_text(_CSV)
    rows = read_sidtrace(tmp_path / "t.csv")
    assert oracle.sidtrace_calls(rows) == 2
    assert oracle.sidtrace_calls(rows, cycles_per_frame=19656 // 2) == 4
    assert oracle.sidtrace_calls([r for r in rows if r.chip is not None]) == 0


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
