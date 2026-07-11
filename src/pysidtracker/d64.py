"""Read PRG files out of a Commodore 1541 ``.d64`` disk image.

C64 tracker test fixtures often ship not as bare ``.prg`` files but inside a
``.d64`` disk image (frequently zipped in a csdb.dk release). Every ``py*``
parser that wants those PRGs re-implemented the same 1541 filesystem walk. This
module is that walk, once: :func:`read_d64` yields one :class:`D64File` per PRG
(name + payload bytes), following the directory track/sector chain and each
file's sector chain. Pure stdlib, so it ships in the wheel and pairs with the
:func:`pysidtracker.testing.fetch_disk` downloader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

# 1541 geometry: 35 tracks, 1-based; index 0 unused. Sectors per track vary.
SECTORS_PER_TRACK = [0] + [21] * 17 + [19] * 7 + [18] * 6 + [17] * 5
SECTOR_SIZE = 256
D64_SIZE = sum(SECTORS_PER_TRACK[1:]) * SECTOR_SIZE  # 174848
DIR_TRACK = 18
DIR_SECTOR = 1
FILETYPE_PRG = 0x82


@dataclass(frozen=True)
class D64File:
    """One PRG file recovered from a disk image: PETSCII ``name`` + ``prg`` bytes.

    ``prg`` is the raw file payload, load-address header included (the first two
    bytes are the little-endian C64 load address), exactly as a ``.prg`` on disk.
    """

    name: str
    prg: bytes


def _ts_offset(track: int, sector: int) -> int:
    if not 1 <= track <= 35:
        raise ValueError(f"track out of range: {track}")
    if not 0 <= sector < SECTORS_PER_TRACK[track]:
        raise ValueError(f"sector out of range for track {track}: {sector}")
    return (sum(SECTORS_PER_TRACK[1:track]) + sector) * SECTOR_SIZE


def _chain(data: bytes, track: int, sector: int) -> Iterator[bytes]:
    """Yield each sector's usable bytes along a track/sector chain."""
    visited = set()
    while True:
        if (track, sector) in visited:
            raise ValueError(f"sector chain loops at T{track}S{sector}")
        visited.add((track, sector))
        block = data[_ts_offset(track, sector) :][:SECTOR_SIZE]
        next_track, next_sector = block[0], block[1]
        if next_track == 0:
            # A terminal block: next_sector is the index of its last used byte.
            yield block[2 : next_sector + 1]
            return
        yield block[2:SECTOR_SIZE]
        track, sector = next_track, next_sector


def read_d64(data: bytes) -> List[D64File]:
    """Every PRG file in the ``.d64`` image ``data``, in directory order.

    Walks the directory chain from track 18 and follows each PRG entry's sector
    chain. Raises :class:`ValueError` if ``data`` is too short to be a ``.d64``
    or a sector chain loops.
    """
    if len(data) < D64_SIZE:
        raise ValueError(f"not a d64: {len(data)} bytes (expected >= {D64_SIZE})")
    files: List[D64File] = []
    track, sector = DIR_TRACK, DIR_SECTOR
    seen = set()
    while (track, sector) not in seen:
        seen.add((track, sector))
        block = data[_ts_offset(track, sector) :][:SECTOR_SIZE]
        for slot in range(8):
            entry = block[2 + slot * 32 : 2 + slot * 32 + 30]
            if len(entry) < 19 or entry[0] != FILETYPE_PRG:
                continue
            name = entry[3:19].rstrip(b"\xa0").decode("latin-1", errors="replace")
            prg = b"".join(_chain(data, entry[1], entry[2]))
            files.append(D64File(name, prg))
        if block[0] == 0:
            break
        track, sector = block[0], block[1]
    return files
