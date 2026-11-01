"""Tests for the PSID writer, header serialiser, and C64 string codec."""

import pytest

from pysidtracker import (
    SidFormatError,
    SidImage,
    encode_cstr,
    parse_sid_header,
    write_psid,
)


def test_write_psid_embed_load_in_body():
    image = b"\xea\x20\x60"
    raw = write_psid(
        load=0x1800, init=0x1800, play=0x1806, image=image, name="TUNE", embed_load=True
    )
    header = raw[:0x7C]
    assert header[0:4] == b"PSID"
    assert header[8:10] == b"\x00\x00"  # loadAddress field: 0 -> load lives in body
    assert header[10:12] == (0x1800).to_bytes(2, "big")
    assert header[12:14] == (0x1806).to_bytes(2, "big")
    assert raw[0x7C:0x7E] == (0x1800).to_bytes(2, "little")
    assert raw[0x7E:] == image
    h = parse_sid_header(raw)
    assert h.load_address == 0
    assert h.real_load_address == 0x1800
    assert h.data_start == 0x7E
    loaded = SidImage.from_bytes(raw)
    assert loaded.load == 0x1800
    assert loaded.image == image


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
