"""Tests for the loaded memory-image model."""

import pytest

from pysidtracker import SidImage, SidParseError

from .helpers import build_psid


def test_overrun_rejected():
    with pytest.raises(SidParseError, match="overruns memory"):
        SidImage.from_sid(build_psid(b"\x00" * 0x10, load=0xFFF8))


def test_byte_and_word_and_peek():
    img = SidImage.from_sid(build_psid(b"\x34\x12", load=0x1000))
    assert img.byte(0x1000) == 0x34
    assert img.word(0x1000) == 0x1234
    assert img.peek(0x1000) == 0x34
    assert img.peek(0x1_0000, default=0xEE) == 0xEE
    assert img.peek(-1, default=7) == 7


def test_slice_out_of_range_raises():
    img = SidImage.from_sid(build_psid(b"\x00", load=0x1000))
    with pytest.raises(SidParseError, match="out of range"):
        img.slice(0xFFFF, 4)


def test_find_split_table_anchor():
    lo = bytes(range(0, 16))
    hi = bytes(range(0x80, 0x90))
    image = lo + hi
    img = SidImage.from_sid(build_psid(image, load=0x1000))
    addr, first, length = img.find_split_table(lo, hi, min_length=8)
    assert addr == 0x1000
    assert first == 0
    assert length == 16
