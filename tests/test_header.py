"""Tests for PSID/RSID container header parsing."""

import pytest

from pysidtracker import SidFormatError, parse_sid_header, write_psid

from .helpers import build_psid


def test_multi_sid_detected():
    sid = build_psid(b"\x00", load=0x1000, version=3, second_sid=0x42)
    h = parse_sid_header(sid)
    assert h.is_multi_sid


def test_data_offset_past_end():
    sid = bytearray(build_psid(b"", load=0x1000))
    # Point dataOffset far past the (now short) file.
    sid[6:8] = (0x9000).to_bytes(2, "big")
    with pytest.raises(SidFormatError, match="past the end"):
        parse_sid_header(bytes(sid))


def test_speed_bits_select_each_song_s_timer():
    header = parse_sid_header(
        write_psid(load=0x1000, init=0, play=0, image=b"", speed=1 << 31 | 2)
    )
    assert header.speed == 1 << 31 | 2
    assert [header.cia_speed(s) for s in (0, 1, 30, 31, 40)] == [
        False,
        True,
        False,
        True,
        True,
    ]
    assert parse_sid_header(header.to_bytes()).speed == header.speed
