"""Synthetic PSID/RSID/PRG builders for the tests (no copyrighted fixtures)."""

import struct

DATA_OFFSET = 0x7C


def build_psid(
    image,
    load,
    init=None,
    play=None,
    version=2,
    songs=1,
    start_song=1,
    name="",
    author="",
    released="",
    flags=0,
    magic=b"PSID",
    second_sid=0,
    data_offset=DATA_OFFSET,
):
    """Assemble a PSID/RSID container around ``image`` bytes."""
    hdr = bytearray(data_offset)
    hdr[0:4] = magic
    struct.pack_into(">H", hdr, 0x04, version)
    struct.pack_into(">H", hdr, 0x06, data_offset)
    struct.pack_into(">H", hdr, 0x08, load)
    struct.pack_into(">H", hdr, 0x0A, load if init is None else init)
    struct.pack_into(">H", hdr, 0x0C, load if play is None else play)
    struct.pack_into(">H", hdr, 0x0E, songs)
    struct.pack_into(">H", hdr, 0x10, start_song)
    hdr[0x16 : 0x16 + len(name)] = name.encode("latin-1")
    hdr[0x36 : 0x36 + len(author)] = author.encode("latin-1")
    hdr[0x56 : 0x56 + len(released)] = released.encode("latin-1")
    struct.pack_into(">H", hdr, 0x76, flags)
    if data_offset > 0x7A:
        hdr[0x7A] = second_sid
    return bytes(hdr) + bytes(image)


def build_psid_embedded_load(image, load, **kwargs):
    """PSID with header loadAddress=0 and the load address embedded in data."""
    body = bytes([load & 0xFF, load >> 8]) + bytes(image)
    return build_psid(body, load=0, **kwargs)


def build_prg(image, load):
    """Bare .prg: 2-byte little-endian load address + image."""
    return bytes([load & 0xFF, load >> 8]) + bytes(image)


# --- tiny hand-assembled 6502 init routines --------------------------------


def store_sig(sig, addr):
    """6502 that stores ``sig`` bytes to ``addr..`` then RTS."""
    code = bytearray()
    for i, b in enumerate(sig):
        target = addr + i
        code += bytes([0xA9, b])  # LDA #b
        code += bytes([0x8D, target & 0xFF, target >> 8])  # STA target
    code += bytes([0x60])  # RTS
    return bytes(code)


RTS_ONLY = bytes([0x60])
