"""Tests for ByteCursor and the validation helpers."""

import pytest

from pysidtracker import ByteCursor, SidParseError, byte_range, check


def test_cursor_reads():
    cur = ByteCursor(b"\x01\x02\x03\x04\x05")
    assert cur.u8("a") == 0x01
    assert cur.u16le("w") == 0x0302
    assert cur.take(2, "rest") == b"\x04\x05"
    assert len(cur) == 0


def test_cursor_truncation():
    cur = ByteCursor(b"\x01")
    with pytest.raises(SidParseError, match="truncated"):
        cur.u16le("word")


def test_check():
    check(True, "ok")
    with pytest.raises(SidParseError, match="bad"):
        check(False, "bad")

    class Custom(Exception):
        pass

    with pytest.raises(Custom):
        check(False, "x", Custom)


def test_byte_range():
    assert byte_range(0xFF, "v") == 0xFF
    with pytest.raises(SidParseError, match="byte range"):
        byte_range(256, "v")
    with pytest.raises(SidParseError):
        byte_range(-1, "v")
