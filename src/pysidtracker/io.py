"""Forward-only byte reading and small validation helpers.

:class:`ByteCursor` is the shared offset-aware reader every leaf hand-rolled for
parsing a fixed layout; :func:`check` and :func:`byte_range` are the guard
helpers that pair with it (and with the writers in :mod:`pysidtracker.header`).
"""

from __future__ import annotations

from .errors import SidParseError


class ByteCursor:
    """A forward-only reader over ``data`` with offset-aware truncation errors."""

    __slots__ = ("data", "pos")

    def __init__(self, data: bytes):
        self.data = bytes(data)
        self.pos = 0

    def __len__(self) -> int:
        return len(self.data) - self.pos

    def take(self, n: int, what: str) -> bytes:
        """Consume ``n`` bytes; raise :class:`SidParseError` on truncation."""
        if self.pos + n > len(self.data):
            raise SidParseError(
                f"truncated: needed {n} bytes for {what} at offset {self.pos}, "
                f"have {len(self.data) - self.pos}"
            )
        chunk = self.data[self.pos : self.pos + n]
        self.pos += n
        return chunk

    def u8(self, what: str) -> int:
        """Consume one byte."""
        return self.take(1, what)[0]

    def u16le(self, what: str) -> int:
        """Consume a little-endian 16-bit word."""
        chunk = self.take(2, what)
        return chunk[0] | (chunk[1] << 8)


def check(cond: bool, msg: str, error: type = SidParseError) -> None:
    """Raise ``error(msg)`` unless ``cond`` is truthy."""
    if not cond:
        raise error(msg)


def byte_range(value: int, what: str) -> int:
    """Return ``value`` if it is in ``0..0xFF``, else raise :class:`SidParseError`."""
    check(0 <= value <= 0xFF, f"{what} out of byte range: {value}")
    return value
