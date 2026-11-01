"""Tests for raw-buffer code scanning and Match operand accessors."""

from pysidtracker import find_code_first


def test_match_u8_u16():
    buf = bytes([0xBD, 0x34, 0x12, 0x99])
    match = find_code_first(buf, "BD {t:w}")
    assert match.u8(0) == 0xBD
    assert match.u8(3) == 0x99
    assert match.u16(1) == 0x1234
