"""Tests for the shared register-log convention."""

import io

import pytest

from pysidtracker import (
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


def test_malformed_line_raises_sidparseerror():
    with pytest.raises(SidParseError):
        read_reglog(io.StringIO("0 0\n"))
    with pytest.raises(SidParseError):
        read_reglog(io.StringIO("0 x 1\n"))


def test_read_reglog_bad_type():
    with pytest.raises(TypeError):
        read_reglog(12345)


def test_frame_writes_skips_out_of_band_and_masks():
    frames = [[(0xD400, 0x1AA), (0xD500, 0x05), (0xD418, 0x42)]]
    out = list(frame_writes(frames, cycles_per_frame=19656))
    # $D500 rebases to 0x100, out of 0..0x18, dropped; val masked to 0xFF.
    assert out == [RegWrite(0, 0x00, 0xAA), RegWrite(16, 0x18, 0x42)]


def test_frame_writes_spacing_guard():
    with pytest.raises(SidParseError):
        list(frame_writes([[(0, 0)]], cycles_per_frame=100, write_spacing=16))
