"""Note-frequency tables and the relocation-tolerant table locator.

The SID plays a note by loading a 16-bit frequency register; a player carries a
per-semitone frequency table (a parallel hi/lo pair) and indexes it by note.
The equal-tempered table is a hardware fact derivable from the SID pitch
formula (``register = f_hz * 2**24 / clock``), so :data:`PAL_FREQ_LO` /
:data:`PAL_FREQ_HI` (and the NTSC variants) are computed here rather than copied
from any tune.

:func:`locate_note_freq` finds a tune's own table pair the way the players read
it: two adjacent ``LDA hi,X`` / ``LDA lo,X`` absolute-indexed operands whose
targets differ by the table length, validated by an octave ramp -- so it pins
the table regardless of relocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .codescan import find_code_all
from .image import SidImage
from .registers import NTSC_CLOCK_HZ, PAL_CLOCK_HZ

# Equal-tempered reference: middle C (261.6255653 Hz) sits at note index 49.
_REF_HZ = 261.6255653
_REF_INDEX = 49
NOTE_COUNT = 96

# The two candidate table lengths tried by the locator (largest valid wins).
NOTE_FREQ_LENGTHS = (96, 95)


def _sid_freq_table(clock: int) -> Tuple[int, ...]:
    """The 96-entry equal-tempered SID frequency-register table for ``clock``."""
    scale = (1 << 24) / clock
    table = tuple(
        min(0xFFFF, round(_REF_HZ * 2 ** ((n - _REF_INDEX) / 12) * scale))
        for n in range(NOTE_COUNT)
    )
    assert len(table) == NOTE_COUNT
    return table


PAL_FREQ = _sid_freq_table(PAL_CLOCK_HZ)
NTSC_FREQ = _sid_freq_table(NTSC_CLOCK_HZ)
PAL_FREQ_LO = bytes(value & 0xFF for value in PAL_FREQ)
PAL_FREQ_HI = bytes(value >> 8 for value in PAL_FREQ)
NTSC_FREQ_LO = bytes(value & 0xFF for value in NTSC_FREQ)
NTSC_FREQ_HI = bytes(value >> 8 for value in NTSC_FREQ)


@dataclass
class NoteFreqTable:
    """The parallel note-frequency hi/lo tables (one entry per semitone)."""

    hi: List[int]
    lo: List[int]
    addr: Optional[int] = None

    def __len__(self) -> int:
        return len(self.hi)

    def freq(self, note: int) -> int:
        """The 16-bit SID frequency register value for semitone ``note``."""
        return (self.hi[note] << 8) | self.lo[note]

    @classmethod
    def pal(cls) -> "NoteFreqTable":
        """The derived equal-tempered PAL table."""
        return cls(hi=list(PAL_FREQ_HI), lo=list(PAL_FREQ_LO))

    @classmethod
    def ntsc(cls) -> "NoteFreqTable":
        """The derived equal-tempered NTSC table."""
        return cls(hi=list(NTSC_FREQ_HI), lo=list(NTSC_FREQ_LO))


def is_octave_ramp(values: Sequence[int], min_steps: int = 6) -> bool:
    """True if ``values`` looks like a note-freq hi octave ramp.

    A valid hi table starts low, ends high, is non-decreasing, and rises at
    least ``min_steps`` times across its span.
    """
    if not values or values[0] == 0 or values[0] > 0x04 or values[-1] < 0x20:
        return False
    steps = 0
    for prev, cur in zip(values, values[1:]):
        if cur < prev:
            return False
        if cur > prev:
            steps += 1
    return steps >= min_steps


_LDA_ABSX = "BD {target:w}"


def locate_note_freq(
    image: SidImage, lengths: Sequence[int] = NOTE_FREQ_LENGTHS
) -> Optional[NoteFreqTable]:
    """Locate and decode a tune's note-frequency tables, or ``None``.

    Collects every ``LDA abs,X`` (opcode ``$BD``) operand target, then looks for
    a hi/lo pair whose targets differ by a candidate table length and whose hi
    table is a valid octave ramp (:func:`is_octave_ramp`). Relocation-tolerant,
    since it keys on the player's own operands.
    """
    targets = {match.captures["target"] for match in find_code_all(image, _LDA_ABSX)}
    for hi_addr in sorted(targets):
        for length in lengths:
            if hi_addr + length not in targets:
                continue
            if hi_addr + 2 * length > len(image.mem):
                continue
            hi = list(image.slice(hi_addr, length))
            if not is_octave_ramp(hi):
                continue
            lo = list(image.slice(hi_addr + length, length))
            return NoteFreqTable(hi=hi, lo=lo, addr=hi_addr)
    return None
