"""Program tables (instrument, wave, arpeggio rows) as functional-graph automata.

Each row has a label and at most one ``succ`` row fetched next (``None`` stops):
:func:`minimise` merges indistinguishable rows, :func:`number_chains` numbers
them by first use and :func:`layout_chains` places them in table slots.
"""

from __future__ import annotations

from collections import Counter
from typing import (
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from .errors import SidFormatError

N = TypeVar("N", bound=Hashable)


def _period(seq: Sequence[int]) -> int:
    """The primitive period of the cyclic sequence ``seq`` (KMP failure function)."""
    fail = [0] * len(seq)
    for i in range(1, len(seq)):
        k = fail[i - 1]
        while k and seq[i] != seq[k]:
            k = fail[k - 1]
        fail[i] = k + (seq[i] == seq[k])
    period = len(seq) - fail[-1]
    return period if len(seq) % period == 0 else len(seq)


def _least_rotation(seq: Sequence[int]) -> int:
    """The start of ``seq``'s lexicographically least rotation (Booth's algorithm)."""
    doubled = list(seq) * 2
    fail = [-1] * len(doubled)
    best = 0
    for j in range(1, len(doubled)):
        char, i = doubled[j], fail[j - best - 1]
        while i != -1 and char != doubled[best + i + 1]:
            if char < doubled[best + i + 1]:
                best = j - i - 1
            i = fail[i]
        if char != doubled[best + i + 1]:
            if char < doubled[best]:
                best = j
            fail[j - best] = -1
        else:
            fail[j - best] = i + 1
    return best


def minimise(
    labels: Mapping[N, Hashable], succ: Mapping[N, Optional[N]]
) -> Dict[N, int]:
    """``{node: class}`` of the coarsest partition stable under ``labels`` and ``succ``.

    Nodes share a class iff their label sequences along ``succ`` agree; a node
    missing from ``succ`` stops. Linear time: trees hash-cons ``(label, class of
    succ)``, cycles key on their least primitive rotation.
    """
    lid: Dict[Hashable, int] = {}
    lab = {n: lid.setdefault(label, len(lid)) for n, label in labels.items()}
    ids: Dict[tuple, int] = {}
    cls: Dict[N, int] = {}
    for root in labels:
        path: List[N] = []
        onpath: Dict[N, int] = {}
        node = root
        while node is not None and node not in cls and node not in onpath:
            onpath[node] = len(path)
            path.append(node)
            node = succ.get(node)
        if node is not None and node not in cls:
            cycle = path[onpath[node] :]
            del path[onpath[node] :]
            seq = [lab[c] for c in cycle]
            period = _period(seq)
            start = _least_rotation(seq[:period])
            rot = tuple(seq[start:period] + seq[:start])
            for i, c in enumerate(cycle):
                cls[c] = ids.setdefault((rot, (i - start) % period), len(ids))
            for c in cycle:
                ids.setdefault((lab[c], cls[succ[c]]), cls[c])
        for n in reversed(path):
            nxt = succ.get(n)
            key = (lab[n], None if nxt is None else cls[nxt])
            cls[n] = ids.setdefault(key, len(ids))
    renum: Dict[int, int] = {}
    return {n: renum.setdefault(cls[n], len(renum)) for n in labels}


def number_chains(
    entries: Iterable[Optional[N]], succ: Mapping[N, Optional[N]], start: int = 0
) -> Dict[N, int]:
    """First-use numbers from ``start``, each entry's chain numbered consecutively.

    A chain stops at a numbered node or a stop; ``None`` entries are skipped.
    """
    out: Dict[N, int] = {}
    for node in entries:
        while node is not None and node not in out:
            out[node] = start + len(out)
            node = succ.get(node)
    return out


def layout_chains(
    succ: Mapping[N, Optional[N]],
    size: int,
    first: int = 0,
    error: type = SidFormatError,
) -> Tuple[Dict[N, int], Dict[int, N]]:
    """``(slots, jumps)`` placing every chain in consecutive slots of a ``size`` table.

    Sources go first; a chain running into a placed node ends in a jump slot,
    ``jumps[slot] = node``. Slots start at ``first``, else ``0``; ``error`` is
    raised when neither fits.
    """
    indegree = Counter(t for t in succ.values() if t is not None)
    order = sorted(succ, key=lambda node: indegree[node] > 0)
    for base in dict.fromkeys((first, 0)):
        slots: Dict[N, int] = {}
        jumps: Dict[int, N] = {}
        pos = base
        for node in order:
            while node not in slots:
                slots[node] = pos
                pos += 1
                node = succ[node]
                if node is None:
                    break
                if node in slots:
                    jumps[pos] = node
                    pos += 1
        if pos <= size:
            return slots, jumps
    raise error(f"{len(succ)} rows need {pos} table slots (max {size})")


__all__ = ["layout_chains", "minimise", "number_chains"]
