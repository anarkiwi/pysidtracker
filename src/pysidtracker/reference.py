"""Match patterns derived from an assembled reference player.

Hand-writing a :mod:`~pysidtracker.codescan` pattern per player release is
tedious and drifts from the vendored player source. :func:`reference_pattern`
walks an assembled player (see :mod:`~pysidtracker.asm6502`) and wildcards
everything that legitimately differs between the reference and a tune that
embeds the same code:

* absolute operands pointing back into the player (relocated; validated on
  match, see :func:`match_at`),
* zero-page operands (allocation differs per build),
* operand bytes of instructions the player rewrites with an absolute store
  (self-modified code),
* declared data ranges, and any caller-supplied extra offsets.

Operands that point outside the player (SID/CIA registers, KERNAL) and branch
offsets stay literal. A :class:`Reference` bundles the code, its relocations and
labels, so a match recovers the table bases (``label -> address``) and
code-identical references are told apart by their data bytes
(:func:`find_reference`). ``facets`` carries the ``variant`` facet values a
matched reference name maps to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .codescan import CodeBuffer, CodePattern, find_code_all
from .image import SidImage

# Absolute STY/STA/STX and read-modify-write opcodes: they can rewrite code.
_ABS_STORES = frozenset(
    {0x8C, 0x8D, 0x8E, 0x99, 0x9D}
    | {0x0E, 0x1E, 0x2E, 0x3E, 0x4E, 0x5E, 0x6E, 0x7E}
    | {0xCE, 0xDE, 0xEE, 0xFE}
)


def opcode_length(opcode: int) -> int:
    """Instruction length in bytes (NMOS illegal opcodes included)."""
    low = opcode & 0x0F
    if opcode == 0x20 or low >= 0x0C:
        return 3
    if low in (0x09, 0x0B) and opcode & 0x10:
        return 3
    if low in (0x08, 0x0A) or opcode in (0x00, 0x40, 0x60):
        return 1
    if low == 0x02 and opcode not in (0x82, 0xA2, 0xC2, 0xE2):
        return 1
    return 2


def _is_absolute(opcode: int) -> bool:
    return opcode_length(opcode) == 3


def _is_zero_page(opcode: int) -> bool:
    if opcode_length(opcode) != 2:
        return False
    low = opcode & 0x0F
    # branches (xx10 pattern), immediates (x9, 80/82/89/C2/E2/A0/A2/C0/E0, xB).
    if low == 0x00 and opcode & 0x10:
        return False
    if low == 0x09 or low == 0x0B or opcode in (0x80, 0x82, 0xA0, 0xA2, 0xC0, 0xE0):
        return False
    return opcode not in (0xC2, 0xE2)


@dataclass(frozen=True)
class Reference:
    """An assembled reference player.

    Attributes:
      name: reference identifier (e.g. a release name).
      code: assembled bytes.
      org: address ``code`` was assembled at.
      labels: label name -> absolute address at ``org`` (from ``assemble``);
        recovered per match as table/entry bases.
      data: ``(start, end)`` offsets into ``code`` that are data, not code.
        They are not walked as instructions, are wildcarded in the pattern and
        are compared byte-wise to break ties between code-identical references.
      wild: extra offsets into ``code`` to wildcard (e.g. ``#<label`` immediates).
      facets: facet values (e.g. ``{"variant": "v2.1"}``) a match implies.
    """

    name: str
    code: bytes
    org: int
    labels: Mapping[str, int] = field(default_factory=dict)
    data: Tuple[Tuple[int, int], ...] = ()
    wild: Tuple[int, ...] = ()
    facets: Mapping[str, str] = field(default_factory=dict)

    @property
    def end(self) -> int:
        """First address past the assembled reference."""
        return self.org + len(self.code)

    def _in_data(self, offset: int) -> bool:
        return any(lo <= offset < hi for lo, hi in self.data)

    def instructions(self) -> List[Tuple[int, int]]:
        """``(offset, length)`` of each instruction, skipping data ranges."""
        out: List[Tuple[int, int]] = []
        pos = 0
        while pos < len(self.code):
            if self._in_data(pos):
                pos += 1
                continue
            length = min(opcode_length(self.code[pos]), len(self.code) - pos)
            out.append((pos, length))
            pos += length
        return out

    def relocations(self) -> List[Tuple[int, int]]:
        """``(operand offset, target offset)`` for each relocated word.

        An absolute operand whose target lies inside the reference is
        relocated with it.
        """
        out: List[Tuple[int, int]] = []
        for pos, length in self.instructions():
            if length == 3 and _is_absolute(self.code[pos]):
                target = self.code[pos + 1] | (self.code[pos + 2] << 8)
                if self.org <= target < self.end:
                    out.append((pos + 1, target - self.org))
        return out

    def self_modified(self) -> List[int]:
        """Operand offsets that an absolute store inside the player rewrites."""
        spans = {pos: length for pos, length in self.instructions()}
        out = set()
        for pos, length in self.instructions():
            if length != 3 or self.code[pos] not in _ABS_STORES:
                continue
            target = (self.code[pos + 1] | (self.code[pos + 2] << 8)) - self.org
            for start, size in spans.items():
                if start < target < start + size:
                    out.update(range(start + 1, start + size))
        return sorted(out)

    def wildcards(self) -> List[int]:
        """Every code offset the pattern wildcards, ascending."""
        wild = set(self.wild) | set(self.self_modified())
        for lo, hi in self.data:
            wild.update(range(lo, hi))
        for pos, length in self.instructions():
            opcode = self.code[pos]
            if length == 3 and _is_absolute(opcode):
                target = self.code[pos + 1] | (self.code[pos + 2] << 8)
                if self.org <= target < self.end:
                    wild.update((pos + 1, pos + 2))
            elif _is_zero_page(opcode):
                wild.add(pos + 1)
        return sorted(offset for offset in wild if 0 <= offset < len(self.code))


def reference_pattern(ref: Reference) -> CodePattern:
    """The :class:`CodePattern` for ``ref`` with relocated/variable bytes wild."""
    wild = set(ref.wildcards())
    tokens = [
        "??" if offset in wild else f"{value:02X}"
        for offset, value in enumerate(ref.code)
    ]
    return CodePattern(" ".join(tokens))


@dataclass(frozen=True)
class ReferenceMatch:
    """A reference located in a buffer.

    Attributes:
      reference: the matched :class:`Reference`.
      addr: address of the reference's first byte.
      bases: label -> address in the matched copy (table/entry bases).
      data_score: data bytes equal to the reference's (see :func:`data_score`).
    """

    reference: Reference
    addr: int
    bases: Dict[str, int]
    data_score: int

    @property
    def name(self) -> str:
        return self.reference.name

    @property
    def facets(self) -> Mapping[str, str]:
        return self.reference.facets


def data_score(ref: Reference, buf: bytes, addr: int) -> int:
    """Number of ``ref``'s data bytes that equal ``buf`` at ``addr``."""
    score = 0
    for lo, hi in ref.data:
        for offset in range(lo, hi):
            if buf[addr + offset] == ref.code[offset]:
                score += 1
    return score


def _relocations_valid(ref: Reference, buf: bytes, addr: int) -> bool:
    for operand, target in ref.relocations():
        word = buf[addr + operand] | (buf[addr + operand + 1] << 8)
        if word != addr + target:
            return False
    return True


def match_at(ref: Reference, image: CodeBuffer, addr: int) -> Optional[ReferenceMatch]:
    """``ref`` matched at exactly ``addr``, or ``None``.

    The masked pattern must match and every relocated operand must point at the
    same offset into the matched copy as in the reference.
    """
    buf = bytes(image.mem) if isinstance(image, SidImage) else bytes(image)
    if addr < 0 or addr + len(ref.code) > len(buf):
        return None
    pat = reference_pattern(ref)
    if find_code_all(buf, pat, start=addr, end=addr + len(ref.code)) == []:
        return None
    if not _relocations_valid(ref, buf, addr):
        return None
    bases = {name: addr + (value - ref.org) for name, value in ref.labels.items()}
    return ReferenceMatch(ref, addr, bases, data_score(ref, buf, addr))


def find_reference(
    image: CodeBuffer,
    refs: Sequence[Reference],
    *,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> List[ReferenceMatch]:
    """Every validated reference match, ascending by address.

    At one address, code-identical references are ordered by data score
    (highest first); equal scores keep their order in ``refs``.
    """
    buf = bytes(image.mem) if isinstance(image, SidImage) else bytes(image)
    found: List[Tuple[int, int, int, ReferenceMatch]] = []
    for index, ref in enumerate(refs):
        for hit in find_code_all(buf, reference_pattern(ref), start=start, end=end):
            match = match_at(ref, buf, hit.addr)
            if match is not None:
                found.append((hit.addr, -match.data_score, index, match))
    found.sort(key=lambda item: item[:3])
    return [item[3] for item in found]


def best_reference(
    image: CodeBuffer,
    refs: Iterable[Reference],
    *,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> Optional[ReferenceMatch]:
    """The first match from :func:`find_reference`, or ``None``."""
    matches = find_reference(image, list(refs), start=start, end=end)
    return matches[0] if matches else None
