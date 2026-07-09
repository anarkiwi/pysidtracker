"""Fast searches over a 64 KiB C64 memory image.

The static playroutine-recognition step scans the loaded image for known byte
signatures and for anchor tables (a low half immediately followed by a matching
high half, e.g. the split note-frequency table many players embed). Both are
numpy-accelerated when numpy is importable and fall back to pure-stdlib
scanning otherwise, so the base install stays dependency-free.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

try:  # numpy is an optional accelerator, never required.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only without numpy
    _np = None


def find_all(mem: Sequence[int], needle: bytes, start: int = 0) -> List[int]:
    """Return every index >= ``start`` where ``needle`` occurs in ``mem``."""
    if not needle:
        return []
    buf = bytes(mem)
    hits = []
    pos = buf.find(needle, start)
    while pos != -1:
        hits.append(pos)
        pos = buf.find(needle, pos + 1)
    return hits


def find_first(mem: Sequence[int], needle: bytes, start: int = 0) -> int:
    """Return the first index >= ``start`` of ``needle`` in ``mem``, or ``-1``."""
    return bytes(mem).find(needle, start)


def find_split_table(
    mem: Sequence[int],
    lo: Sequence[int],
    hi: Sequence[int],
    *,
    min_length: int = 8,
    limit: int = 0x10000,
) -> Optional[tuple]:
    """Locate a split lo/hi table anchor in ``mem``.

    Players commonly embed a two-column table as ``lo[first:first+n]`` directly
    followed by ``hi[first:first+n]`` (a contiguous slice of a longer known
    table, ``lo``/``hi``). Returns ``(addr, first, length)`` for the longest
    such match with ``length >= min_length``, or ``None``. ``addr`` is the start
    of the low column; the high column starts at ``addr + length``.
    """
    lo = bytes(lo)
    hi = bytes(hi)
    n = len(lo)
    if n == 0 or len(hi) != n:
        return None
    limit = min(limit, len(mem))
    if _np is not None:
        return _find_split_table_np(mem, lo, hi, n, min_length, limit)
    return _find_split_table_py(mem, lo, hi, n, min_length, limit)


def _find_split_table_py(mem, lo, hi, n, min_length, limit):
    best = None
    # A table needs 2*min_length bytes, so the last viable start is
    # ``limit - 2 * min_length`` inclusive (a table ending exactly at ``limit``).
    for addr in range(0, limit - 2 * min_length + 1):
        first = mem[addr]
        for fn in range(n):
            if lo[fn] != first:
                continue
            length = 0
            while (
                fn + length < n
                and addr + length < limit
                and mem[addr + length] == lo[fn + length]
            ):
                length += 1
            if length < min_length or addr + 2 * length > limit:
                continue
            if all(mem[addr + length + i] == hi[fn + i] for i in range(length)):
                if best is None or length > best[2]:
                    best = (addr, fn, length)
    return best


def _find_split_table_np(mem, lo, hi, n, min_length, limit):
    arr = _np.frombuffer(bytes(mem)[:limit], dtype=_np.uint8)
    lo_np = _np.frombuffer(lo, dtype=_np.uint8)
    best = None
    # For each possible table offset ``fn`` and run length, a match requires the
    # lo slice at ``addr`` and the hi slice at ``addr+length`` to line up. We
    # scan candidate start addresses cheaply by matching the first lo byte, then
    # verify/extend in the (small) candidate set.
    for fn in range(n):
        starts = _np.nonzero(arr == lo_np[fn])[0]
        for addr in starts.tolist():
            length = 0
            while (
                fn + length < n
                and addr + length < limit
                and arr[addr + length] == lo[fn + length]
            ):
                length += 1
            if length < min_length or addr + 2 * length > limit:
                continue
            if (
                bytes(arr[addr + length : addr + 2 * length].tolist())
                == hi[fn : fn + length]
            ):
                if best is None or length > best[2]:
                    best = (addr, fn, length)
    return best
