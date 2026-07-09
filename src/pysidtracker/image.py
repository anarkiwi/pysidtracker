"""The loaded C64 memory image a parser works against.

:class:`SidImage` unwraps a PSID/RSID container (or a bare ``.prg``) into a full
64 KiB memory image with the tune placed at its load address, plus absolute-
addressed read accessors. This is the shared substitute for the per-parser
``_Image`` / ``load_sid`` / ``_memory_accessors`` helpers.
"""

from __future__ import annotations

from typing import List, Optional

from . import _scan
from .errors import SidParseError
from .header import SidHeader, parse_sid_header

MEM_SIZE = 0x10000


class SidImage:
    """A 64 KiB C64 memory image with the tune loaded at its load address.

    Attributes:
      mem: ``bytearray`` of length ``0x10000`` (the C64 address space).
      load: C64 address the image data starts at.
      end: one past the last loaded byte (``0x10000`` after an init run, which
        may have scattered data across memory).
      header: the parsed :class:`SidHeader`, or ``None`` for a bare ``.prg``.
      image: the raw C64 image bytes (player + data, without the container).
      container: the container bytes preceding ``image`` (PSID/RSID header plus
        any embedded 2-byte load address, or the ``.prg`` load-address prefix),
        kept so a writer can re-emit a byte-identical container.
    """

    __slots__ = ("mem", "load", "end", "header", "image", "container")

    def __init__(
        self,
        mem: bytearray,
        load: int,
        end: int,
        header: Optional[SidHeader],
        image: bytes,
        container: bytes,
    ):
        self.mem = mem
        self.load = load
        self.end = end
        self.header = header
        self.image = image
        self.container = container

    @classmethod
    def from_bytes(cls, data: bytes) -> "SidImage":
        """Load a PSID/RSID ``.sid`` or a bare ``.prg`` image from ``data``."""
        if data[:4] in (b"PSID", b"RSID"):
            return cls.from_sid(data)
        return cls.from_prg(data)

    @classmethod
    def from_sid(cls, data: bytes) -> "SidImage":
        """Load a PSID/RSID container into a 64 KiB image."""
        header = parse_sid_header(data)
        image = bytes(data[header.data_start :])
        container = bytes(data[: header.data_start])
        load = header.real_load_address
        return cls._place(image, load, header, container)

    @classmethod
    def from_prg(cls, data: bytes) -> "SidImage":
        """Load a bare ``.prg`` (2-byte little-endian load address + image)."""
        if len(data) < 2:
            raise SidParseError("PRG too short for a load address")
        load = data[0] | (data[1] << 8)
        return cls._place(bytes(data[2:]), load, None, bytes(data[:2]))

    @classmethod
    def _place(cls, image: bytes, load: int, header, container) -> "SidImage":
        end = load + len(image)
        if end > MEM_SIZE:
            raise SidParseError(
                f"data overruns memory (load {load:#06x}, {len(image)} bytes)"
            )
        mem = bytearray(MEM_SIZE)
        mem[load:end] = image
        return cls(mem, load, end, header, image, container)

    def byte(self, addr: int) -> int:
        """Read one byte at C64 ``addr`` (raises out of ``0..0xFFFF``)."""
        if not 0 <= addr < MEM_SIZE:
            raise SidParseError(f"address {addr:#06x} out of range")
        return self.mem[addr]

    def peek(self, addr: int, default: int = 0) -> int:
        """Read one byte at ``addr``, returning ``default`` if out of range."""
        if 0 <= addr < MEM_SIZE:
            return self.mem[addr]
        return default

    def word(self, addr: int) -> int:
        """Read a little-endian 16-bit word at C64 ``addr``."""
        return self.byte(addr) | (self.byte(addr + 1) << 8)

    def slice(self, addr: int, length: int) -> bytes:
        """Read ``length`` bytes starting at C64 ``addr`` (bounds-checked)."""
        if addr < 0 or addr + length > MEM_SIZE:
            raise SidParseError(f"slice at {addr:#06x}+{length} out of range")
        return bytes(self.mem[addr : addr + length])

    def ptr(self, lo_base: int, hi_base: int, index: int) -> int:
        """Read a split-pointer-table entry (``lo[index]`` + ``hi[index]<<8``)."""
        return self.peek(lo_base + index) | (self.peek(hi_base + index) << 8)

    def find(self, needle: bytes, start: Optional[int] = None) -> int:
        """First address of ``needle`` in the image, or ``-1``.

        Search starts at ``start`` (defaulting to the load address, so the
        container header is skipped).
        """
        return _scan.find_first(self.mem, needle, self.load if start is None else start)

    def find_all(self, needle: bytes, start: Optional[int] = None) -> List[int]:
        """Every address of ``needle`` in the image at or after ``start``."""
        return _scan.find_all(self.mem, needle, self.load if start is None else start)

    def find_split_table(self, lo, hi, *, min_length: int = 8) -> Optional[tuple]:
        """Locate a split lo/hi table anchor (see :func:`_scan.find_split_table`)."""
        return _scan.find_split_table(self.mem, lo, hi, min_length=min_length)
