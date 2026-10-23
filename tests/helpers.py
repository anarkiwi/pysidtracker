"""Synthetic PSID/RSID/PRG builders for the tests (no copyrighted fixtures)."""

import struct
from pathlib import Path

import zstandard
from vagrant_moses.media import GEOMETRIES

from pysidtracker.oracle import SidtraceRow
from pysidtracker.registers import PAL_CLOCK_HZ

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
    speed=0,
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
    struct.pack_into(">I", hdr, 0x12, speed)
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


def event(cycle, value):
    """A sidtrace interrupt row (raise, ``*_entry`` or ``rti``) at ``cycle``."""
    return SidtraceRow(cycle, None, None, None, None, None, value)


def handled(source, at, writes=(), entry=7):
    """sidtrace rows of one ``source`` interrupt raised at ``at`` and handled.

    The handler is entered ``entry`` cycles later, makes ``writes`` (``(cycle,
    reg, value)`` on chip 0) and returns one cycle after the last.
    """
    end = max([at + entry] + [w[0] for w in writes]) + 1
    return [
        event(at, source),
        event(at + entry, f"{source}_entry"),
        *(SidtraceRow(c, None, None, None, 0, r, v) for c, r, v in writes),
        event(end, "rti"),
    ]


def trace_csv(rows):
    """``rows`` as sidtrace CSV text (empty delta columns)."""
    lines = ["cycle,cycle_since_nmi,cycle_since_video_irq,cycle_since_cia_irq"]
    lines[0] += ",chip,reg,value"
    for r in rows:
        chip = "" if r.chip is None else r.chip
        reg = "" if r.reg is None else r.reg
        lines.append(f"{r.cycle},,,,{chip},{reg},{r.value}")
    return "\n".join(lines) + "\n"


def build_disk(files, geometry="d64", error_block=False, first_track=1):
    """A synthetic CBM DOS image of ``geometry`` holding ``files``.

    ``files`` is ``[(name, data, type_byte)]`` (at most 8, one directory sector);
    chains take the free sectors from ``first_track`` on, skipping the directory track.
    """
    geom = next(g for g in GEOMETRIES if g.name == geometry)
    img = bytearray(geom.blocks * 256)
    dir_track = geom.dir_ts[0]
    free = (
        (track, sector)
        for track, count in enumerate(geom.sectors, 1)
        if track != dir_track and track >= first_track
        for sector in range(count)
    )
    dir_off = geom.offset(*geom.dir_ts)
    img[dir_off + 1] = 0xFF
    for slot, (name, data, ftype) in enumerate(files):
        chunks = [data[i : i + 254] for i in range(0, len(data), 254)]
        chain = [next(free) for _ in chunks]
        links = chain[1:] + [(0, len(chunks[-1]) + 1)]
        for chunk, (track, sector), link in zip(chunks, chain, links):
            off = geom.offset(track, sector)
            img[off : off + 2 + len(chunk)] = bytes(link) + chunk
        entry = dir_off + 2 + 32 * slot
        img[entry : entry + 3] = bytes((ftype, *chain[0]))
        img[entry + 3 : entry + 19] = name.encode("latin-1").ljust(16, b"\xa0")
    return bytes(img) + (bytes(geom.blocks) if error_block else b"")


def build_t64(files):
    """A synthetic ``.t64`` holding ``files`` ``[(name, prg)]``, end addresses 0."""
    head = bytearray(0x40 + 32 * len(files))
    head[:3] = b"C64"
    head[0x22] = head[0x24] = len(files)
    body = bytearray()
    for i, (name, prg) in enumerate(files):
        entry = 0x40 + 32 * i
        head[entry : entry + 4] = bytes((1, 0x82)) + prg[:2]
        head[entry + 8 : entry + 12] = (len(head) + len(body)).to_bytes(4, "little")
        head[entry + 16 : entry + 32] = name.encode("latin-1").ljust(16, b" ")
        body += prg[2:]
    return bytes(head + body)


def paced_trace(seconds, cycles_per_call, *, init=1000, source="cia_irq"):
    """The sidtrace rows of ``seconds`` of a PAL tune called every
    ``cycles_per_call`` cycles from ``init``; call ``k`` writes ``k & 0xFF`` to reg 0.
    """
    calls = range(init, int(seconds * PAL_CLOCK_HZ), cycles_per_call)
    return [
        row
        for k, at in enumerate(calls)
        for row in handled(source, at, [(at + 40, 0, k & 0xFF)])
    ]


def fake_sidtrace(cycles_per_call, renders, *, cap=None):
    """A ``run_sidtrace`` stand-in writing :func:`paced_trace` of each render's
    ``seconds`` (at most ``cap``), which it appends to ``renders``."""

    def run(_tune_path, out_path, *, seconds=60, **_kw):
        renders.append(seconds)
        rows = paced_trace(min(seconds, cap or seconds), cycles_per_call)
        text = trace_csv(rows).encode()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(zstandard.ZstdCompressor().compress(text))
        return Path(out_path)

    return run
