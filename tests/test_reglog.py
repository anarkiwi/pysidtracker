"""Tests for the shared register-log convention."""

import io

import pytest

from pysidtracker import (
    DEFAULT_WRITE_SPACING,
    REGLOG_HEADER,
    RegWrite,
    SidParseError,
    frame_writes,
    read_reglog,
    write_reglog,
)


def test_round_trip_path(tmp_path):
    writes = [RegWrite(0, 0, 0x11), RegWrite(16, 4, 0x21), RegWrite(19656, 24, 0x0F)]
    path = tmp_path / "log.txt"
    write_reglog(writes, path)
    assert read_reglog(path) == writes
    assert path.read_text(encoding="utf-8").splitlines()[0] == REGLOG_HEADER


def test_round_trip_filelike_no_header():
    writes = [RegWrite(0, 1, 2), RegWrite(32, 3, 4)]
    buf = io.StringIO()
    write_reglog(writes, buf, header=False)
    text = buf.getvalue()
    assert not text.startswith("#")
    assert read_reglog(io.StringIO(text)) == writes


def test_custom_string_header():
    writes = [RegWrite(0, 0, 1)]
    buf = io.StringIO()
    custom = "# pygoattracker register log: clock reg val"
    write_reglog(writes, buf, header=custom)
    lines = buf.getvalue().splitlines()
    assert lines[0] == custom
    assert lines[0] != REGLOG_HEADER
    assert read_reglog(io.StringIO(buf.getvalue())) == writes


def test_header_true_uses_default():
    buf = io.StringIO()
    write_reglog([RegWrite(0, 0, 1)], buf, header=True)
    assert buf.getvalue().splitlines()[0] == REGLOG_HEADER


def test_comments_and_blank_lines_ignored():
    text = "# a comment\n\n0 0 10  # inline\n  \n16 1 20\n"
    assert read_reglog(io.StringIO(text)) == [RegWrite(0, 0, 10), RegWrite(16, 1, 20)]


def test_malformed_line_raises_sidparseerror():
    with pytest.raises(SidParseError):
        read_reglog(io.StringIO("0 0\n"))
    with pytest.raises(SidParseError):
        read_reglog(io.StringIO("0 x 1\n"))


def test_read_reglog_bad_type():
    with pytest.raises(TypeError):
        read_reglog(12345)


def test_frame_writes_math_rebasing_and_offsets():
    frames = [
        [(0xD400, 0xAA), (0xD404, 0x11)],
        [(0xD418, 0xBB)],
    ]
    out = list(frame_writes(frames, cycles_per_frame=19656))
    assert out == [
        RegWrite(0, 0x00, 0xAA),
        RegWrite(DEFAULT_WRITE_SPACING, 0x04, 0x11),
        RegWrite(19656, 0x18, 0xBB),
    ]


def test_frame_writes_skips_out_of_band_and_masks():
    frames = [[(0xD400, 0x1AA), (0xD500, 0x05), (0xD418, 0x42)]]
    out = list(frame_writes(frames, cycles_per_frame=19656))
    # $D500 rebases to 0x100, out of 0..0x18, dropped; val masked to 0xFF.
    assert out == [RegWrite(0, 0x00, 0xAA), RegWrite(16, 0x18, 0x42)]


def test_frame_writes_start_frame_and_reg_base_zero():
    frames = [[(0, 0x10), (24, 0x20)]]
    out = list(
        frame_writes(frames, cycles_per_frame=19656, sid_reg_base=0, start_frame=2)
    )
    assert out == [RegWrite(2 * 19656, 0, 0x10), RegWrite(2 * 19656 + 16, 24, 0x20)]


def test_frame_writes_spacing_guard():
    with pytest.raises(SidParseError):
        list(frame_writes([[(0, 0)]], cycles_per_frame=100, write_spacing=16))
