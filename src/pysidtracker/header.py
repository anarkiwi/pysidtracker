"""PSID/RSID (``.sid``) container-header parsing.

The outer SID *container* is a big-endian header (documented at
https://www.hvsc.c64.org/download/C64Music/DOCUMENTS/SID_file_format.txt)
wrapping a raw C64 memory image (player code + song data). Every parser in the
family unwraps this identically; :func:`parse_sid_header` is that one
implementation.

The header's advertised load/init/play addresses and subtune count are *not*
trustworthy for packed, relocating, or crunched tunes -- the real layout only
exists after the init routine runs. This module resolves the on-disk fields;
:mod:`pysidtracker.detect` handles the untrustworthy-header problem.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import SidFormatError

PSID_MAGIC = b"PSID"
RSID_MAGIC = b"RSID"

# Field offsets within the (big-endian) SID container header.
_MAGIC_POS = 0x00
_VERSION_POS = 0x04
_DATA_OFFSET_POS = 0x06
_LOAD_ADDRESS_POS = 0x08
_INIT_ADDRESS_POS = 0x0A
_PLAY_ADDRESS_POS = 0x0C
_SONGS_POS = 0x0E
_START_SONG_POS = 0x10
_NAME_POS = 0x16
_AUTHOR_POS = 0x36
_RELEASED_POS = 0x56
_STR_LEN = 0x20
_FLAGS_POS = 0x76
_SECOND_SID_POS = 0x7A  # v3+: address byte of the 2nd SID (0 => single SID)
_THIRD_SID_POS = 0x7C  # v4+: address byte of the 3rd SID
_V2_HEADER_LEN = 0x7C  # PSID/RSID v2 header size (data follows immediately)


def _u16be(data: bytes, pos: int) -> int:
    return (data[pos] << 8) | data[pos + 1]


def decode_cstr(raw: bytes) -> str:
    """Decode a C64 fixed-field string: bytes up to the first NUL, latin-1."""
    return raw.split(b"\0", 1)[0].decode("latin-1")


def encode_cstr(text: str, size: int, pad: bytes = b"\x00") -> bytes:
    """Encode ``text`` as a ``size``-byte latin-1 fixed field, right-padded.

    Raises :class:`SidFormatError` if the encoded text does not fit in ``size``
    bytes (one byte is reserved for the pad terminator, so ``len < size``).
    """
    raw = text.encode("latin-1", "replace")
    if len(raw) >= size:
        raise SidFormatError(f"string {text!r} does not fit in {size} bytes")
    return raw.ljust(size, pad)


# Back-compat alias for the private name earlier callers imported.
_decode_str = decode_cstr


@dataclass
class SidHeader:
    """Decoded SID-container header fields.

    :attr:`load_address` is the raw header field (``0`` for the common
    "load address lives in the first two bytes of the data" case);
    :attr:`real_load_address` is the resolved C64 address the image loads at and
    :attr:`data_start` is the file offset of the first byte of that image.
    """

    magic: bytes
    version: int
    data_offset: int
    load_address: int
    init_address: int
    play_address: int
    songs: int
    start_song: int
    name: str
    author: str
    released: str
    flags: int
    second_sid: int
    third_sid: int
    real_load_address: int
    data_start: int

    @property
    def is_psid(self) -> bool:
        return self.magic == PSID_MAGIC

    @property
    def is_rsid(self) -> bool:
        return self.magic == RSID_MAGIC

    @property
    def is_multi_sid(self) -> bool:
        """True if the header advertises a 2nd/3rd/4th SID chip."""
        return self.version >= 3 and (self.second_sid != 0 or self.third_sid != 0)

    def to_bytes(self) -> bytes:
        """Serialise this header back to its container bytes (inverse of parse).

        Produces a ``data_offset``-length header (at least a v2 header) whose
        fields re-parse to equal values via :func:`parse_sid_header`.
        """
        size = max(_V2_HEADER_LEN, self.data_offset)
        return _pack_header(
            magic=self.magic,
            version=self.version,
            data_offset=size,
            load=self.load_address,
            init=self.init_address,
            play=self.play_address,
            songs=self.songs,
            start_song=self.start_song,
            name=self.name,
            author=self.author,
            released=self.released,
            flags=self.flags,
            second_sid=self.second_sid,
            third_sid=self.third_sid,
        )


def parse_sid_header(data: bytes) -> SidHeader:
    """Decode the PSID/RSID container header at the start of ``data``.

    Raises :class:`SidFormatError` if the magic is neither ``PSID`` nor ``RSID``
    or the header is truncated. When the header ``loadAddress`` field is ``0``
    the real load address is read from the first little-endian word of the data
    area (and :attr:`SidHeader.data_start` skips it).
    """
    if len(data) < _START_SONG_POS + 2:
        raise SidFormatError("input is too short to contain a SID header")
    magic = bytes(data[_MAGIC_POS : _MAGIC_POS + 4])
    if magic not in (PSID_MAGIC, RSID_MAGIC):
        raise SidFormatError(
            "not a SID file (expected 'PSID' or 'RSID' magic at offset 0, "
            f"found {magic!r})"
        )
    version = _u16be(data, _VERSION_POS)
    data_offset = _u16be(data, _DATA_OFFSET_POS)
    load_address = _u16be(data, _LOAD_ADDRESS_POS)
    init_address = _u16be(data, _INIT_ADDRESS_POS)
    play_address = _u16be(data, _PLAY_ADDRESS_POS)
    songs = _u16be(data, _SONGS_POS)
    start_song = _u16be(data, _START_SONG_POS)
    name = _decode_str(data[_NAME_POS : _NAME_POS + _STR_LEN])
    author = _decode_str(data[_AUTHOR_POS : _AUTHOR_POS + _STR_LEN])
    released = _decode_str(data[_RELEASED_POS : _RELEASED_POS + _STR_LEN])
    flags = (
        _u16be(data, _FLAGS_POS) if version >= 2 and len(data) >= _FLAGS_POS + 2 else 0
    )
    second_sid = data[_SECOND_SID_POS] if len(data) > _SECOND_SID_POS else 0
    third_sid = data[_THIRD_SID_POS] if len(data) > _THIRD_SID_POS else 0

    if load_address == 0:
        if data_offset + 2 > len(data):
            raise SidFormatError("SID dataOffset points past the end of the file")
        real_load = data[data_offset] | (data[data_offset + 1] << 8)
        data_start = data_offset + 2
    else:
        if data_offset > len(data):
            raise SidFormatError("SID dataOffset points past the end of the file")
        real_load = load_address
        data_start = data_offset

    return SidHeader(
        magic=magic,
        version=version,
        data_offset=data_offset,
        load_address=load_address,
        init_address=init_address,
        play_address=play_address,
        songs=songs,
        start_song=start_song,
        name=name,
        author=author,
        released=released,
        flags=flags,
        second_sid=second_sid,
        third_sid=third_sid,
        real_load_address=real_load,
        data_start=data_start,
    )


def _pack_u16be(buf: bytearray, pos: int, value: int) -> None:
    buf[pos] = (value >> 8) & 0xFF
    buf[pos + 1] = value & 0xFF


def _pack_header(
    *,
    magic: bytes,
    version: int,
    data_offset: int,
    load: int,
    init: int,
    play: int,
    songs: int,
    start_song: int,
    name: str,
    author: str,
    released: str,
    flags: int,
    second_sid: int = 0,
    third_sid: int = 0,
) -> bytes:
    if magic not in (PSID_MAGIC, RSID_MAGIC):
        raise SidFormatError(f"container magic must be PSID or RSID, got {magic!r}")
    hdr = bytearray(data_offset)
    hdr[_MAGIC_POS : _MAGIC_POS + 4] = magic
    _pack_u16be(hdr, _VERSION_POS, version)
    _pack_u16be(hdr, _DATA_OFFSET_POS, data_offset)
    _pack_u16be(hdr, _LOAD_ADDRESS_POS, load)
    _pack_u16be(hdr, _INIT_ADDRESS_POS, init)
    _pack_u16be(hdr, _PLAY_ADDRESS_POS, play)
    _pack_u16be(hdr, _SONGS_POS, songs)
    _pack_u16be(hdr, _START_SONG_POS, start_song)
    hdr[_NAME_POS : _NAME_POS + _STR_LEN] = encode_cstr(name, _STR_LEN)
    hdr[_AUTHOR_POS : _AUTHOR_POS + _STR_LEN] = encode_cstr(author, _STR_LEN)
    hdr[_RELEASED_POS : _RELEASED_POS + _STR_LEN] = encode_cstr(released, _STR_LEN)
    if version >= 2:
        _pack_u16be(hdr, _FLAGS_POS, flags)
    if version >= 3:
        if len(hdr) > _SECOND_SID_POS:
            hdr[_SECOND_SID_POS] = second_sid & 0xFF
        if len(hdr) > _THIRD_SID_POS:
            hdr[_THIRD_SID_POS] = third_sid & 0xFF
    return bytes(hdr)


def write_psid(
    *,
    load: int,
    init: int,
    play: int,
    image: bytes,
    name: str = "",
    author: str = "",
    released: str = "",
    songs: int = 1,
    start_song: int = 1,
    flags: int = 0,
    version: int = 2,
    kind: str = "PSID",
) -> bytes:
    """Assemble a PSID/RSID v2 container (the inverse of :func:`parse_sid_header`).

    ``kind`` selects the container magic (``"PSID"`` or ``"RSID"``). The load
    address is written to the header ``loadAddress`` field and ``image`` follows
    the header directly, so ``parse_sid_header(write_psid(...))`` reproduces the
    ``load``/``init``/``play``/metadata inputs.
    """
    magic = kind.encode("ascii") if isinstance(kind, str) else bytes(kind)
    header = _pack_header(
        magic=magic,
        version=version,
        data_offset=_V2_HEADER_LEN,
        load=load,
        init=init,
        play=play,
        songs=songs,
        start_song=start_song,
        name=name,
        author=author,
        released=released,
        flags=flags,
    )
    return header + bytes(image)
