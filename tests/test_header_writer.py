"""Tests for the PSID writer, header serialiser, and C64 string codec."""

import pytest

from pysidtracker import (
    SidFormatError,
    decode_cstr,
    encode_cstr,
    parse_sid_header,
    write_psid,
)


def test_write_psid_round_trips():
    raw = write_psid(
        load=0x1000,
        init=0x1003,
        play=0x1006,
        image=b"\x01\x02\x03",
        name="TUNE",
        author="ME",
        released="2026 X",
        songs=3,
        start_song=2,
        flags=0x0004,
    )
    h = parse_sid_header(raw)
    assert h.magic == b"PSID"
    assert h.version == 2
    assert h.real_load_address == 0x1000
    assert h.load_address == 0x1000
    assert h.init_address == 0x1003
    assert h.play_address == 0x1006
    assert h.songs == 3
    assert h.start_song == 2
    assert h.name == "TUNE"
    assert h.author == "ME"
    assert h.released == "2026 X"
    assert h.flags == 0x0004
    assert raw[h.data_start :] == b"\x01\x02\x03"


def test_write_psid_rsid_kind():
    raw = write_psid(load=0x2000, init=0x2000, play=0x2000, image=b"\x00", kind="RSID")
    assert parse_sid_header(raw).is_rsid


def test_write_psid_bad_kind():
    with pytest.raises(SidFormatError, match="magic"):
        write_psid(load=0x1000, init=0, play=0, image=b"", kind="MP3 ")


def test_header_to_bytes_round_trips():
    raw = write_psid(load=0x1234, init=0x1240, play=0x1250, image=b"abc", name="hi")
    h = parse_sid_header(raw)
    again = parse_sid_header(h.to_bytes() + b"abc")
    assert again.real_load_address == h.real_load_address
    assert again.init_address == h.init_address
    assert again.name == h.name


def test_encode_cstr_pads_and_checks():
    assert encode_cstr("AB", 4) == b"AB\x00\x00"
    with pytest.raises(SidFormatError):
        encode_cstr("TOOLONG", 4)


def test_decode_cstr_stops_at_nul():
    assert decode_cstr(b"HI\x00rest") == "HI"
    assert decode_cstr(encode_cstr("round", 32)) == "round"
