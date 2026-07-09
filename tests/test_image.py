"""Tests for the loaded memory-image model."""

import pytest

from pysidtracker import SidImage, SidParseError

from .helpers import build_prg, build_psid, build_psid_embedded_load


def test_from_sid_places_data():
    img = SidImage.from_sid(build_psid(b"ABCD", load=0x1000))
    assert img.load == 0x1000
    assert img.end == 0x1004
    assert img.slice(0x1000, 4) == b"ABCD"
    assert img.header.is_psid
    assert img.image == b"ABCD"
    assert img.container[:4] == b"PSID"


def test_from_bytes_dispatches_sid_and_prg():
    assert SidImage.from_bytes(build_psid(b"XY", load=0x1000)).header is not None
    prg = SidImage.from_bytes(build_prg(b"XY", load=0x0801))
    assert prg.header is None
    assert prg.load == 0x0801
    assert prg.slice(0x0801, 2) == b"XY"
    assert prg.container == b"\x01\x08"


def test_from_prg_too_short():
    with pytest.raises(SidParseError, match="PRG too short"):
        SidImage.from_prg(b"\x00")


def test_embedded_load():
    img = SidImage.from_sid(build_psid_embedded_load(b"\xde\xad", load=0x4000))
    assert img.load == 0x4000
    assert img.slice(0x4000, 2) == b"\xde\xad"


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


def test_byte_out_of_range_raises():
    img = SidImage.from_sid(build_psid(b"\x00", load=0x1000))
    with pytest.raises(SidParseError, match="out of range"):
        img.byte(0x10000)


def test_slice_out_of_range_raises():
    img = SidImage.from_sid(build_psid(b"\x00", load=0x1000))
    with pytest.raises(SidParseError, match="out of range"):
        img.slice(0xFFFF, 4)


def test_ptr_reads_split_table():
    # lo table at 0x2000, hi table at 0x2010.
    image = bytearray(0x20)
    image[0x00:0x03] = bytes([0x11, 0x22, 0x33])  # lo[0..2]
    image[0x10:0x13] = bytes([0xAA, 0xBB, 0xCC])  # hi[0..2]
    img = SidImage.from_sid(build_psid(bytes(image), load=0x2000))
    assert img.ptr(0x2000, 0x2010, 1) == 0xBB22


def test_find_and_find_all():
    img = SidImage.from_sid(build_psid(b"..SIG..SIG..", load=0x1000))
    assert img.find(b"SIG") == 0x1002
    assert img.find_all(b"SIG") == [0x1002, 0x1007]
    assert img.find(b"NOPE") == -1


def test_find_defaults_to_load_start():
    # The container header is not in the memory image at all; find() defaults
    # its start to the load address (skipping zero-page/BASIC memory below it).
    img = SidImage.from_sid(build_psid(b"payload", load=0x1000))
    assert img.find(b"PSID") == -1  # container magic never lands in memory
    # A needle below the load address is not found with the default start.
    img.mem[0x0800:0x0807] = b"payload"
    assert img.find(b"payload") == 0x1000
    assert img.find(b"payload", start=0) == 0x0800


def test_find_split_table_anchor():
    lo = bytes(range(0, 16))
    hi = bytes(range(0x80, 0x90))
    image = lo + hi
    img = SidImage.from_sid(build_psid(image, load=0x1000))
    addr, first, length = img.find_split_table(lo, hi, min_length=8)
    assert addr == 0x1000
    assert first == 0
    assert length == 16
