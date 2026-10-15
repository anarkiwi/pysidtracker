"""The flat PSID/RSID loader against the container's documented fields."""

import pytest

from pysidtracker import (
    SidFormatError,
    is_ntsc,
    is_rsid,
    load_psid,
    psid_image,
    psid_speed,
    psid_songs,
    speed_cia,
    write_psid,
)

BODY = bytes(range(1, 21))


def _container(**kw):
    base = dict(
        load=0x8000,
        init=0x8100,
        play=0x8200,
        image=BODY,
        songs=3,
        start_song=2,
        flags=0,
        speed=0,
    )
    base.update(kw)
    return write_psid(**base)


def test_load_psid_places_the_body_at_the_header_load():
    data = _container()
    mem, load, init, play = load_psid(data)
    assert (load, init, play) == (0x8000, 0x8100, 0x8200)
    assert isinstance(mem, bytearray) and len(mem) == 0x10000
    assert mem[0x8000 : 0x8000 + len(BODY)] == BODY
    assert mem[:0x8000] == bytes(0x8000)
    assert psid_image(data) == (0x8000, 0x8014)


def test_in_body_load_address_wins_when_the_header_field_is_zero():
    real = bytes((0x0E, 0x04))  # 0x040E little-endian
    data = write_psid(load=0, init=0x0600, play=0x0700, image=real + BODY, songs=1)
    mem, load, init, play = load_psid(data)
    assert load == 0x040E
    assert mem[0x40E : 0x40E + len(BODY)] == BODY
    assert (init, play) == (0x0600, 0x0700)
    assert psid_image(data) == (0x40E, 0x40E + len(BODY))


def test_songs_speed_and_clock():
    data = _container(songs=4, start_song=3, flags=0x08, speed=0b101)
    assert psid_songs(data) == (4, 3)
    assert is_ntsc(data)
    assert psid_speed(data) == 0b101
    assert [speed_cia(data, i) for i in range(3)] == [True, False, True]
    assert speed_cia(data) is True


def test_pal_header_is_not_ntsc():
    assert not is_ntsc(_container(flags=0x04))  # bits 2-3 == 0b01 = PAL
    assert not is_ntsc(
        write_psid(load=0, init=1, play=2, image=BODY, version=1, flags=0x08)
    )


def test_rsid_vs_psid():
    rsid = write_psid(load=0x2000, init=0x2100, play=0x2200, image=BODY, kind="RSID")
    assert is_rsid(rsid)
    assert not is_rsid(_container())
    mem, load, init, play = load_psid(rsid)
    assert (load, init, play) == (0x2000, 0x2100, 0x2200)
    assert mem[0x2000 : 0x2000 + len(BODY)] == BODY


def test_rejects_non_sid_bytes():
    for bad in (b"\x01\x02\x03\x04" * 8, b"PSID"):
        with pytest.raises(SidFormatError):
            load_psid(bad)
        with pytest.raises(SidFormatError):
            psid_songs(bad)
