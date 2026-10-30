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


def frequency(note: int, clock_hz: int) -> int:
    """The SID frequency word of ``note`` at ``clock_hz`` (0 where it rounds to 0
    or does not fit 16 bits)."""
    word = round(A4_HZ * 2 ** ((note - A4_NOTE) / 12) * (1 << 24) / clock_hz)
    return word if 0 < word < 1 << 16 else 0


@lru_cache(maxsize=None)
def table(clock_hz: int, part: int, offset: int, size: int) -> Tuple[int, ...]:
    """``size`` entries; entry ``i`` is ``part`` (see :data:`PARTS`) of the word of
    note ``i + offset``, 0 for a negative note."""
    words = (
        frequency(i + offset, clock_hz) if i + offset >= 0 else 0 for i in range(size)
    )
    if part == PARTS["word"]:
        return tuple(words)
    shift = 8 * part
    return tuple(w >> shift & 0xFF for w in words)


__all__ = ["A4_HZ", "A4_NOTE", "PARTS", "frequency", "table"]
