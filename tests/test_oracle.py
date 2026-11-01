"""Tests for the register-grid oracle and its framers."""

import pytest

from pysidtracker import (
    EmuPlayer,
    SidParseError,
    aligned_match,
    grid_from_writes,
    read_sidwr,
    register_grid,
    run_until,
)
from pysidtracker.image import SidImage
from pysidtracker.oracle import _SIDWR_REC
from pysidtracker.testing import whole_song

from .helpers import build_prg, build_psid

# init at $1000: LDA #$0A / STA $D400 / RTS
# play at $1005: LDA #$05 / STA $D401 / RTS
_SIMPLE = bytes(
    [0xA9, 0x0A, 0x8D, 0x00, 0xD4, 0x60, 0xA9, 0x05, 0x8D, 0x01, 0xD4, 0x60]
)

# A play routine exercising the NMOS illegal opcodes.
_ILLEGAL_PLAY = bytes(
    [
        0xCB,
        0x01,  # SBX #$01
        0x0B,
        0x03,  # ANC #$03
        0x4B,
        0x0F,  # ALR #$0F
        0x6B,
        0x0F,  # ARR #$0F
        0xEB,
        0x01,  # SBC #$01
        0xAB,
        0x05,  # LAX #$05
        0x8F,
        0x00,
        0x20,  # SAX $2000
        0xAF,
        0x00,
        0x20,  # LAX $2000
        0x1A,  # NOP (implied illegal)
        0x80,
        0x01,  # NOP #imm
        0x04,
        0x10,  # NOP $10
        0x14,
        0x10,  # NOP $10,X
        0x0C,
        0x00,
        0x20,  # NOP $2000
        0x1C,
        0x00,
        0x20,  # NOP $2000,X
        0x60,  # RTS
    ]
)


_PAL_READ = bytes([0xAD, 0xA6, 0x02, 0x8D, 0x00, 0xD4, 0x60])  # LDA $02A6/STA $D400


def test_emuplayer_tune_bytes_over_the_pal_flag():
    player = EmuPlayer(b"\x7e" + _PAL_READ, load=0x02A6, init=0x02A7, play=0x02AD)
    assert player.regs[0] == 0x7E


def test_register_grid_no_header():
    image = SidImage.from_prg(build_prg(_SIMPLE, load=0x1000))
    with pytest.raises(SidParseError):
        register_grid(image, whole_song(1))


def test_grid_from_writes_empty():
    assert grid_from_writes([]) == []


def test_grid_from_writes_out_of_range_reg_ignored():
    grid = grid_from_writes([(0, 99, 0x10), (0, 0, 0x20)], reg_count=25)
    assert grid[0][0] == 0x20


def test_read_sidwr(tmp_path):
    path = tmp_path / "t.sidwr.bin"
    records = [
        (100, 0xD400, 0, 0x11),
        (200, 0xD401, 1, 0x22),
        (300, 0xD500, 30, 0x33),  # reg >= 25, filtered out
    ]
    blob = b"".join(_SIDWR_REC.pack(*rec) for rec in records)
    path.write_bytes(blob)
    assert read_sidwr(path) == [(100, 0, 0x11), (200, 1, 0x22)]


def test_aligned_match_exact():
    oracle = [[1, 2], [3, 4]]
    assert aligned_match(oracle, [[1, 2], [3, 4]])


def test_aligned_match_with_silent_lead():
    oracle = [[1, 2], [3, 4]]
    rendered = [[0, 0], [1, 2], [3, 4]]
    assert aligned_match(oracle, rendered, max_lead=2)


def test_aligned_match_mismatch_and_empty():
    assert not aligned_match([[1]], [[2]])
    assert not aligned_match([[1]], [])
    # Too short to align.
    assert not aligned_match([[1], [2], [3]], [[1]])


def test_emuplayer_loses_the_interrupt_a_call_outlasts():
    # play: INC $D400, then a ~1280-cycle delay loop -- longer than the period.
    play = bytes([0xEE, 0x00, 0xD4, 0xA2, 0x00, 0xCA, 0xD0, 0xFD, 0x60])
    player = EmuPlayer(
        _SIMPLE[:6] + play, load=0x1000, init=0x1000, play=0x1006, cycles_per_frame=1000
    )
    assert [row[0] for row in player.render_grid(whole_song(6))] == [
        11,
        11,
        12,
        12,
        13,
        13,
    ]


def _simple_image():
    return SidImage.from_bytes(
        build_psid(_SIMPLE, load=0x1000, init=0x1000, play=0x1006)
    )


def test_run_until_stops_in_init_before_the_instruction():
    entry = run_until(_simple_image(), [0x1002])
    assert (entry.pc, entry.a, entry.phase, entry.call) == (0x1002, 0x0A, "init", 0)
    assert entry.memory[0xD400] == 0 and entry.memory[0x1000] == 0xA9


def test_run_until_finds_play_and_skips_snapshot():
    entry = run_until(_simple_image(), {0x1008}, play_calls=2, snapshot=False)
    assert (entry.pc, entry.phase, entry.call, entry.memory) == (
        0x1008,
        "play",
        1,
        None,
    )


def test_run_until_none_when_never_entered():
    assert run_until(_simple_image(), [0x3000], play_calls=2) is None
    assert run_until(_simple_image(), [0x1008]) is None
