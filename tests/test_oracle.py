"""Tests for the register-grid oracle and its framers."""

import struct

import pytest

from pysidtracker import (
    PAL_CYCLES_PER_FRAME,
    EmuPlayer,
    SidParseError,
    aligned_match,
    grid_from_writes,
    read_sidwr,
    register_grid,
)
from pysidtracker.image import SidImage
from pysidtracker.oracle import _SIDWR_REC

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


def test_register_grid_from_bytes():
    tune = build_psid(_SIMPLE, load=0x1000, init=0x1000, play=0x1006)
    grid = register_grid(tune, 3)
    assert len(grid) == 3
    for row in grid:
        assert len(row) == 25
        assert row[0] == 0x0A
        assert row[1] == 0x05


def test_register_grid_from_image():
    tune = build_psid(_SIMPLE, load=0x1000, init=0x1000, play=0x1006)
    image = SidImage.from_bytes(tune)
    grid = register_grid(image, 1)
    assert grid[0][0] == 0x0A


def test_register_grid_illegal_opcodes():
    image = _SIMPLE[:6] + _ILLEGAL_PLAY  # init then illegal play
    tune = build_psid(image, load=0x1000, init=0x1000, play=0x1006)
    grid = register_grid(tune, 2, illegal_opcodes=True)
    assert len(grid) == 2
    assert grid[0][0] == 0x0A


def test_register_grid_no_header():
    image = SidImage.from_prg(build_prg(_SIMPLE, load=0x1000))
    with pytest.raises(SidParseError):
        register_grid(image, 1)


def test_emuplayer_runs_tunes_own_code():
    player = EmuPlayer(_SIMPLE, load=0x1000, init=0x1000, play=0x1006)
    assert player.regs[0] == 0x0A  # init wrote $D400
    grid = player.render_grid(3)
    assert len(grid) == 3
    for row in grid:
        assert len(row) == 25 and row[0] == 0x0A and row[1] == 0x05


def test_emuplayer_matches_register_grid():
    tune = build_psid(_SIMPLE, load=0x1000, init=0x1000, play=0x1006)
    player = EmuPlayer(_SIMPLE, load=0x1000, init=0x1000, play=0x1006)
    assert player.render_grid(3) == register_grid(tune, 3)


def test_emuplayer_masks_pw_hi():
    # play writes $7D to $D403 (a PW-high register); only the low nibble is real.
    play = bytes([0xA9, 0x7D, 0x8D, 0x03, 0xD4, 0x60])
    player = EmuPlayer(_SIMPLE[:6] + play, load=0x1000, init=0x1000, play=0x1006)
    player.play_frame()
    assert player.regs[3] == 0x0D


def test_emuplayer_play_frame_diffs():
    player = EmuPlayer(_SIMPLE, load=0x1000, init=0x1000, play=0x1006)
    assert len(player.play_frame()) == 25  # first frame: full register file
    assert player.play_frame() == []  # steady state: no register changed


def test_emuplayer_illegal_opcodes():
    player = EmuPlayer(
        _SIMPLE[:6] + _ILLEGAL_PLAY,
        load=0x1000,
        init=0x1000,
        play=0x1006,
        illegal_opcodes=True,
    )
    assert player.regs[0] == 0x0A
    assert len(player.render_grid(2)) == 2


def test_grid_from_writes_empty():
    assert grid_from_writes([]) == []


def test_grid_from_writes_gap_anchor_forward_fill_and_pw_mask():
    cpf = PAL_CYCLES_PER_FRAME
    writes = [
        (0, 0, 0xAA),  # init baseline
        (16, 3, 0xFF),  # PW-hi reg -> masked to 0x0F
        (cpf, 1, 0x22),  # first play (after > gap)
        (2 * cpf, 1, 0x33),
    ]
    grid = grid_from_writes(writes)
    assert grid == [
        [0xAA, 0x22, 0, 0x0F] + [0] * 21,
        [0xAA, 0x33, 0, 0x0F] + [0] * 21,
    ]


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


def test_read_sidwr_ignores_trailing_partial(tmp_path):
    path = tmp_path / "t.sidwr.bin"
    blob = _SIDWR_REC.pack(1, 0xD400, 0, 0x05) + b"\x00\x01\x02"
    path.write_bytes(blob)
    assert read_sidwr(path) == [(1, 0, 0x05)]
    assert struct.calcsize("<qHBB") == _SIDWR_REC.size


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
