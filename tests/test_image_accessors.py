"""Tests for the absolute-address image accessors, to_prg and parse_prg."""

import pytest

from pysidtracker import SidImage, SidParseError, parse_prg

from .helpers import build_prg


def _image():
    return SidImage.from_prg(build_prg(b"\x11\x22\x33\x44", load=0x1000))


def test_contains():
    img = _image()
    assert img.contains(0x1000)
    assert img.contains(0x1003)
    assert not img.contains(0x1004)
    assert not img.contains(0x0FFF)


def test_byte_and_word_at():
    img = _image()
    assert img.byte_at(0x1000) == 0x11
    assert img.word_at(0x1000) == 0x2211
    with pytest.raises(SidParseError):
        img.byte_at(-1)


def test_poke_grows_end():
    img = _image()
    assert img.end == 0x1004
    img.poke(0x1004, 0x55)
    assert img.end == 0x1005
    assert img.byte_at(0x1004) == 0x55
    img.poke_bytes(0x1005, b"\xaa\xbb")
    assert img.end == 0x1007
    assert img.slice(0x1005, 2) == b"\xaa\xbb"


def test_poke_before_load_guarded():
    img = _image()
    with pytest.raises(SidParseError, match="before the load"):
        img.poke(0x0FFF, 0)


def test_poke_out_of_range():
    img = _image()
    with pytest.raises(SidParseError, match="out of range"):
        img.poke(0x10000, 0)


def test_to_prg_round_trips():
    img = _image()
    prg = img.to_prg()
    assert prg == build_prg(b"\x11\x22\x33\x44", load=0x1000)
    again = SidImage.from_prg(prg)
    assert again.load == 0x1000
    assert again.slice(0x1000, 4) == b"\x11\x22\x33\x44"


def test_parse_prg():
    load, body = parse_prg(build_prg(b"xyz", load=0x0801))
    assert load == 0x0801
    assert body == b"xyz"


def test_parse_prg_expected_load_mismatch():
    with pytest.raises(SidParseError, match="unexpected load"):
        parse_prg(build_prg(b"z", load=0x1000), expected_load=0x0801)


def test_parse_prg_too_short():
    with pytest.raises(SidParseError, match="too short"):
        parse_prg(b"\x00")
