"""Single-file oracle compare: ``survey.compare_file``/``compare_grids``/``file``."""

import csv
import hashlib
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import zstandard

from pysidtracker import formats, oracle, registers, survey
from pysidtracker.asm6502 import assemble
from pysidtracker.errors import SidFormatError
from pysidtracker.header import write_psid
from pysidtracker.testing import whole_song

from .helpers import build_prg, handled, trace_csv

LOAD = 0x1000
PLAYER = """
    jmp init
    jmp play
init: lda #0
    sta $fb
    rts
play: inc $fb
    lda $fb
    ldx #24
loop: sta $d400,x
    dex
    bpl loop
    rts
"""
CPF = registers.PAL_CYCLES_PER_FRAME
NFRAMES = 40


@pytest.fixture(name="prg", scope="module")
def _prg():
    return build_prg(assemble(PLAYER, LOAD)[0], LOAD)


def expected(nframes, shift=0):
    """The player's grid: every register holds the play count, PW highs masked."""
    grid = np.repeat(((np.arange(nframes) + 1 - shift) & 0xFF)[:, None], 25, 1)
    grid[:, list(registers.PW_HI_REGS)] &= 0x0F
    return grid.astype(np.int16)


def render_exact(data, song, nframes):
    """Fake renderer of the synthetic player."""
    assert data[:4] == b"PSID" and song == 1
    return expected(nframes), {"n": nframes}


def render_shift3(_data, _song, nframes):
    """:func:`render_exact` three frames late."""
    return expected(nframes, 3), {}


def render_shift_m2(_data, _song, nframes):
    """:func:`render_exact` two frames early."""
    return expected(nframes, -2), {}


def render_glitch(data, song, nframes):
    """:func:`render_exact` with voice 2's control register wrong on frame 7."""
    grid, info = render_exact(data, song, nframes)
    grid[7, 11] ^= 0x80
    return grid, info


def render_unknown(data, song, nframes):
    """Fake renderer that does not recognise the file."""
    raise SidFormatError(f"unknown {len(data)} {song} {nframes}")


def test_compare_file_covers_the_whole_song(prg, songlengths):
    songlengths.add(survey.as_sid(prg), 0.3)
    row = survey.compare_file(prg, render_exact)
    assert row["frames"] == 15 and row["verdict"] == "exact"


def test_compare_file_differs_and_errors(prg, tmp_path):
    path = tmp_path / "t.prg"
    path.write_bytes(prg)
    row = survey.compare_file(path, render_glitch, frames=whole_song(NFRAMES))
    assert (row["verdict"], row["first_diff"], row["diff_regs"]) == (
        "differs",
        7,
        "v2_ctrl",
    )
    assert (row["min_bad"], row["path"]) == (1, str(path))
    assert survey.compare_file(prg, render_unknown, frames=whole_song(4))[
        "verdict"
    ] == ("not_decompiled")
    row = survey.compare_file(b"\x00", render_exact, frames=whole_song(4))
    assert row["verdict"] == "oracle_error" and "SidParseError" in row["error"]


def _video_trace(nframes):
    """A video-IRQ sidtrace trace whose play writes the call count everywhere."""
    rows = []
    for k in range(nframes + 4):
        at = 1000 + k * CPF
        writes = [(at + 20 + r, r, (k + 1) & 0xFF) for r in range(25)]
        rows += handled("video_irq", at, writes)
    return zstandard.ZstdCompressor().compress(trace_csv(rows).encode())


def test_compare_file_cached_sidtrace(prg, tmp_path, monkeypatch):
    monkeypatch.setattr(survey, "sidtrace_image_id", lambda: "sha256:a")
    work = tmp_path / "work"
    md5 = hashlib.md5(survey.as_sid(prg)).hexdigest()
    (survey.oracle_cache(work) / f"{md5}_1.csv.zst").write_bytes(_video_trace(NFRAMES))
    row = survey.compare_file(prg, render_exact, frames=whole_song(NFRAMES), work=work)
    assert (row["oracle"], row["verdict"]) == ("sidtrace", "exact")
    assert row["seconds"] == pytest.approx(NFRAMES * CPF / registers.PAL_CLOCK_HZ)
    assert (work / "sid" / f"{md5}.sid").read_bytes() == survey.as_sid(prg)


def test_compare_file_sidtrace_fallback(prg, tmp_path, monkeypatch):
    monkeypatch.setattr(survey, "sidtrace_image_id", lambda: None)
    seen = []

    def _unavailable(tune_path, *_, **__):
        seen.append(tune_path)
        raise oracle.SidtraceUnavailable("no docker")

    monkeypatch.setattr(oracle, "run_sidtrace", _unavailable)
    sid_path = tmp_path / "t.sid"
    sid_path.write_bytes(survey.as_sid(prg))
    row = survey.compare_file(
        sid_path, render_exact, frames=whole_song(8), work=tmp_path / "w"
    )
    assert (row["oracle"], row["verdict"], row["seconds"]) == ("jennings", "exact", "")
    assert seen == [sid_path]


def _no_docker(*_, **__):
    raise oracle.SidtraceUnavailable("no docker")


def _hook(tmp_path):
    hook = tmp_path / "oracle_hvsc.py"
    hook.write_text(
        "from tests.test_survey_file import render_exact as render\n", "utf-8"
    )
    return hook


def test_file_cli_csv(prg, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(survey, "sidtrace_image_id", lambda: None)
    monkeypatch.setattr(oracle, "run_sidtrace", _no_docker)
    files = []
    for name in ("a.prg", "b.prg"):
        files.append(tmp_path / name)
        files[-1].write_bytes(prg)
    out = tmp_path / "out.csv"
    args = ["file", str(_hook(tmp_path)), *map(str, files), "--frames", "16"]
    assert survey.main(args + ["--jobs", "1", "--out", str(out)]) == 0
    rows = survey.load_tunes(out)
    assert list(rows[0]) == survey.FILE_FIELDS
    assert [(r["path"], r["verdict"], r["min_bad"]) for r in rows] == [
        (str(f), "exact", "0") for f in files
    ]
    capsys.readouterr()
    work = ["--work", str(tmp_path / "w")]
    survey.main(args[:3] + ["--frames", "16", "--jobs", "1", "--init", "$1000"] + work)
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert [(r["verdict"], r["lead"], r["oracle"]) for r in rows] == [
        ("exact", "0", "jennings")
    ]


def render_by_song(_data, song, nframes):
    """The grid of a player whose count starts at the 0-based subtune."""
    return expected(nframes, 1 - song), {}


def test_file_cli_songs(prg, tmp_path, monkeypatch):
    monkeypatch.setattr(survey, "sidtrace_image_id", lambda: None)
    monkeypatch.setattr(oracle, "run_sidtrace", _no_docker)
    code = assemble(PLAYER.replace("init: lda #0", "init: nop"), LOAD)[0]
    sid, bare = tmp_path / "s.sid", tmp_path / "b.prg"
    sid.write_bytes(
        write_psid(load=LOAD, init=LOAD, play=LOAD + 3, image=code, songs=3)
    )
    bare.write_bytes(prg)
    out = tmp_path / "out.csv"
    for files, spec, picks in (
        ([sid, bare], "all", [(sid, 1), (sid, 2), (sid, 3), (bare, 1)]),
        ([sid], "3,1-2", [(sid, 3), (sid, 1), (sid, 2)]),
    ):
        tally = survey.file_cli(
            f"{__name__}:render_by_song",
            files,
            song=spec,
            out=out,
            frames=whole_song(8),
        )
        rows = survey.load_tunes(out)
        assert sorted((r["path"], r["song"]) for r in rows) == sorted(
            (str(f), str(n)) for f, n in picks
        )
        assert tally == {"exact": len(picks)}


class _GridPlayer:
    """A format player of the synthetic tune: its expected grid."""

    def __init__(self, subtune):
        self.subtune = subtune

    def render_grid(self, nframes):
        return expected(nframes)


FAKE_FORMAT = formats.SidFormat(
    name="fake",
    parser=SimpleNamespace(
        parse=lambda data, subtune: subtune, recognize=lambda image: True
    ),
    player=_GridPlayer,
)


def test_file_cli_format(prg, tmp_path, monkeypatch):
    monkeypatch.setattr(formats, "discover_formats", lambda: [FAKE_FORMAT])
    files = [tmp_path / f"{i}.prg" for i in range(2)]
    for f in files:
        f.write_bytes(prg)
    out = tmp_path / "out.csv"
    args = ["file", *map(str, files), "--frames", "16", "--jobs", "1"]
    for name in ("auto", "fake"):
        assert survey.main(args + ["--out", str(out), "--format", name]) == 0
        rows = survey.load_tunes(out)
        assert [(r["path"], r["verdict"]) for r in rows] == [
            (str(f), "exact") for f in files
        ]
        assert {json.loads(r["info"])["format"] for r in rows} == {"fake"}
    assert survey.main(args + ["--out", str(out), "--format", "other"]) == 0
    assert {r["verdict"] for r in survey.load_tunes(out)} == {"not_decompiled"}
    with pytest.raises(SystemExit):
        survey.main(["file", str(files[0]), "--frames", "16"])


def test_fidelity_row(songlengths):
    data, other = b"\x00\x10\x01", b"\x00\x10\x02"
    songlengths.add(data, 0.06)

    def render(data, nframes, _song):
        return [[data[2]] * 25 for _ in range(nframes)]

    ok = survey.fidelity_row({}, data, lambda d: d, render=render)
    assert ok["fidelity"] == "exact"
    bad = survey.fidelity_row({}, data, lambda d: other, render=render)
    assert bad["fidelity"] == "differs" and bad["fidelity_first"] == 0
    assert survey.fidelity_row({}, b"\x01", lambda d: d)["fidelity"] == "not_decompiled"

    def boom(_):
        raise ValueError("x")

    assert survey.fidelity_row({}, data, boom)["fidelity"] == "rebuild_error"
