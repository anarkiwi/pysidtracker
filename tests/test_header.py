"""Tests for PSID/RSID container header parsing."""

import pytest

from pysidtracker import SidFormatError, parse_sid_header
from pysidtracker.header import _decode_str

from .helpers import build_psid, build_psid_embedded_load


def test_basic_fields():
    sid = build_psid(
        b"\x01\x02\x03",
        load=0x1000,
        init=0x1003,
        play=0x1006,
        songs=3,
        start_song=2,
        name="TUNE",
        author="ME",
        released="2026",
        flags=0x0004,
    )
    h = parse_sid_header(sid)
    assert h.magic == b"PSID"
    assert h.is_psid and not h.is_rsid
    assert h.version == 2
    assert h.load_address == 0x1000
    assert h.real_load_address == 0x1000
    assert h.data_start == 0x7C
    assert h.init_address == 0x1003
    assert h.play_address == 0x1006
    assert h.songs == 3
    assert h.start_song == 2
    assert h.name == "TUNE"
    assert h.author == "ME"
    assert h.released == "2026"
    assert h.flags == 0x0004
    assert not h.is_multi_sid


def test_rsid_magic():
    h = parse_sid_header(build_psid(b"\x00", load=0x2000, magic=b"RSID"))
    assert h.is_rsid and not h.is_psid


def test_embedded_load_address():
    sid = build_psid_embedded_load(b"\xaa\xbb", load=0x1234)
    h = parse_sid_header(sid)
    assert h.load_address == 0
    assert h.real_load_address == 0x1234
    assert h.data_start == 0x7C + 2


def test_multi_sid_detected():
    sid = build_psid(b"\x00", load=0x1000, version=3, second_sid=0x42)
    h = parse_sid_header(sid)
    assert h.is_multi_sid


def test_version1_has_no_flags():
    sid = build_psid(b"\x00", load=0x1000, version=1, flags=0x1234)
    h = parse_sid_header(sid)
    assert h.flags == 0


def test_bad_magic():
    with pytest.raises(SidFormatError, match="not a SID file"):
        parse_sid_header(build_psid(b"\x00", load=0x1000, magic=b"MP3 "))


def test_truncated_header():
    with pytest.raises(SidFormatError, match="too short"):
        parse_sid_header(b"PSID\x00")


def test_data_offset_past_end():
    sid = bytearray(build_psid(b"", load=0x1000))
    # Point dataOffset far past the (now short) file.
    sid[6:8] = (0x9000).to_bytes(2, "big")
    with pytest.raises(SidFormatError, match="past the end"):
        parse_sid_header(bytes(sid))


def test_decode_str_stops_at_nul():
    assert _decode_str(b"HI\x00\x00garbage") == "HI"
