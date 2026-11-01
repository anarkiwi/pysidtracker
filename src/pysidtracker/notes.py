"""The standard equal-tempered note words of a SID clock.

Note 0 is C0, 24 is C2 and 57 is A4 (440 Hz); the word is the 16-bit frequency
register value that sounds the note on a chip clocked at ``clock_hz``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

A4_NOTE = 57
A4_HZ = 440
PARTS = {"lo": 0, "hi": 1, "word": 2}
MAX_OCTAVES = 64


def frequency(note: int, clock_hz: int) -> int:
    """The SID frequency word of ``note`` at ``clock_hz`` (0 where it rounds to 0
    or does not fit 16 bits)."""
    if note - A4_NOTE > MAX_OCTAVES * 12:
        return 0
    word = round(A4_HZ * 2 ** ((note - A4_NOTE) / 12) * (1 << 24) / clock_hz)
    return word if 0 < word < 1 << 16 else 0


def entry(clock_hz: int, part: int, offset: int, index: int) -> int:
    """Entry ``index`` of :func:`table` (``clock_hz``, ``part``, ``offset``), without the table."""
    word = frequency(index + offset, clock_hz)
    return word if part == PARTS["word"] else word >> 8 * part & 0xFF


@lru_cache(maxsize=None)
def table(clock_hz: int, part: int, offset: int, size: int) -> Tuple[int, ...]:
    """``size`` entries; entry ``i`` is ``part`` (see :data:`PARTS`) of the word of
    note ``i + offset``."""
    words = (frequency(i + offset, clock_hz) for i in range(size))
    if part == PARTS["word"]:
        return tuple(words)
    shift = 8 * part
    return tuple(w >> shift & 0xFF for w in words)


__all__ = ["A4_HZ", "A4_NOTE", "PARTS", "entry", "frequency", "table"]
