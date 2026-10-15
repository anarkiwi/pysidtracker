"""Flat PSID/RSID loader: the container's facts for a host with its own memory.

``load_psid`` and friends return what a host needs to enter a tune faithfully --
the body placed in 64 KiB of RAM, the header's init/play, song facts, clock and
CIA speed -- over the shared ``header`` / ``SidImage`` parsing.
"""

from __future__ import annotations

from typing import Tuple

from .header import RSID_MAGIC, parse_sid_header
from .image import SidImage


def load_psid(data: bytes) -> Tuple[bytearray, int, int, int]:
    """``(mem64k, load, init, play)`` of a PSID/RSID container's tune.

    ``mem`` holds the body at its load address in 64 KiB of RAM; ``load``
    honours the in-body address when the header field is ``0``.
    """
    image = SidImage.from_sid(data)
    header = image.header
    return image.mem, image.load, header.init_address, header.play_address


def psid_image(data: bytes) -> Tuple[int, int]:
    """``(lo, hi)`` of the cells a PSID/RSID file loads."""
    image = SidImage.from_sid(data)
    return image.load, image.end


def psid_songs(data: bytes) -> Tuple[int, int]:
    """``(songs, startsong)`` from the header (both 1-based)."""
    header = SidImage.from_sid(data).header
    return header.songs, header.start_song


def is_rsid(data: bytes) -> bool:
    """True for an RSID container (real C64 playback, no PSID driver)."""
    return data[:4] == RSID_MAGIC


def is_ntsc(data: bytes) -> bool:
    """True when a v2+ header's clock bits (``flags`` bits 2-3) say NTSC."""
    header = parse_sid_header(data)
    return header.version >= 2 and (header.flags >> 2) & 3 == 2


def psid_speed(data: bytes) -> int:
    """The header's 32-bit ``speed`` word, one bit per song (31 shared from 32 on)."""
    return parse_sid_header(data).speed


def speed_cia(data: bytes, song: int = 0) -> bool:
    """True when song ``song`` (0-based) plays on a CIA timer: its ``speed`` bit."""
    return parse_sid_header(data).cia_speed(song)


__all__ = [
    "load_psid",
    "psid_image",
    "psid_songs",
    "is_rsid",
    "is_ntsc",
    "psid_speed",
    "speed_cia",
]
