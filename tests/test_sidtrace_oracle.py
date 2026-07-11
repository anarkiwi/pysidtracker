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
from pysidtracker import oracle, testing
from pysidtracker.testing import make_oracle_fixtures, oracle_grid

from .helpers import build_psid

_HEADER = ",".join(oracle.SIDTRACE_COLUMNS)

# A minimal CSV: a pre-IRQ volume write, then one reg-0 write per CIA frame.
_CSV = (
    f"{_HEADER}\n"
    "100,,,,0,24,15\n"
    "19756,,,50,0,0,10\n"
    "39412,,,60,0,0,20\n"
    "59068,,,55,0,0,30\n"
)

# Three clean CIA frames (raise cycles 19656, 39312, 58968 -> cadence 19656).
_ROWS = [
    SidtraceRow(19656, None, None, 0, 0, 0, 10),
    SidtraceRow(39312, None, None, 0, 0, 0, 20),
    SidtraceRow(58968, None, None, 0, 0, 0, 30),
]


def _zst(text: str) -> bytes:
    return zstandard.ZstdCompressor().compress(text.encode("utf-8"))


def test_read_sidtrace_plain_csv(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text(_CSV)
    rows = read_sidtrace(path)
    assert rows[0] == SidtraceRow(100, None, None, None, 0, 24, 15)
    assert rows[1] == SidtraceRow(19756, None, None, 50, 0, 0, 10)
    assert len(rows) == 4


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


def test_sidtrace_cadence_none_without_irqs():
    assert sidtrace_cadence([SidtraceRow(0, None, None, None, 0, 0, 1)]) is None


def test_sidtrace_cadence_prefers_video_over_cia():
    rows = [
        SidtraceRow(20000, None, 100, 999, 0, 0, 1),
        SidtraceRow(39656, None, 100, 999, 0, 0, 2),
    ]
    # video-raise 19900, 39556 -> 19656 (cia column ignored).
    assert sidtrace_cadence(rows) == 19656


def test_sidtrace_grid_matches_grid_from_writes():
    writes = [(r.cycle, r.reg, r.value) for r in _ROWS]
    assert sidtrace_grid(_ROWS) == grid_from_writes(writes, cycles_per_frame=19656)


def test_sidtrace_grid_filters_by_chip():
    rows = [
        SidtraceRow(20000, None, 100, None, 0, 0, 11),
        SidtraceRow(20000, None, 100, None, 1, 0, 99),
        SidtraceRow(39656, None, 100, None, 0, 0, 22),
    ]
    assert all(row[0] != 99 for row in sidtrace_grid(rows, chip=0))


def test_register_grid_seeds_driver_volume():
    # init at $1000: LDA #$0A / STA $D400 / RTS ; play just RTS (never sets vol).
    image = bytes([0xA9, 0x0A, 0x8D, 0x00, 0xD4, 0x60, 0x60])
    tune = build_psid(image, load=0x1000, init=0x1000, play=0x1006)
    grid = register_grid(tune, 2)
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

    monkeypatch.setattr(testing, "run_sidtrace", _no_render)
    grid = oracle_grid(tune, oracle_cache=cache)
    assert grid == sidtrace_grid(read_sidtrace(cache / "song.csv.zst"))


def test_oracle_grid_renders_on_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(oracle.subprocess, "run", _fake_docker(_CSV))
    tune = tmp_path / "song.sid"
    tune.write_bytes(b"PSID" + b"\x00" * 60)
    grid = oracle_grid(tune, oracle_cache=tmp_path / "csv", seconds=2, frames=2)
    assert (tmp_path / "csv" / "song.csv.zst").exists() and len(grid) == 2


def test_make_oracle_fixtures_returns_two_fixtures():
    tune_id, oracle_match = make_oracle_fixtures(
        {"x": "A/x.sid"}, hvsc_cache="/tmp/h", oracle_cache="/tmp/o"
    )
    assert tune_id is not None and oracle_match is not None
