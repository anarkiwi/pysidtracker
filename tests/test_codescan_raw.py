"""Tests for raw-buffer code scanning and Match operand accessors."""

from pysidtracker import find_code_all, find_code_first
from pysidtracker.codescan import Match


def test_find_on_raw_bytes():
    buf = bytes([0x00, 0xA9, 0x2A, 0x8D, 0x00, 0xD4])
    match = find_code_first(buf, "A9 {imm}")
    assert match.addr == 1
    assert match.captures["imm"] == 0x2A


def test_find_on_bytearray_and_memoryview():
    data = bytearray([0xBD, 0x34, 0x12])
    for buf in (data, memoryview(bytes(data))):
        match = find_code_first(buf, "BD {t:w}")
        assert match.captures["t"] == 0x1234


def test_match_u8_u16():
    buf = bytes([0xBD, 0x34, 0x12, 0x99])
    match = find_code_first(buf, "BD {t:w}")
    assert match.u8(0) == 0xBD
    assert match.u8(3) == 0x99
    assert match.u16(1) == 0x1234


def test_match_equality_ignores_buf():
    buf = bytes([0xA9, 0x01])
    match = find_code_first(buf, "A9 {imm}")
    assert match == Match(0, {"imm": 0x01})


def test_find_all_raw():
    buf = bytes([0xA9, 0x01, 0xA9, 0x02])
    matches = find_code_all(buf, "A9 {imm}")
    assert [m.captures["imm"] for m in matches] == [0x01, 0x02]
