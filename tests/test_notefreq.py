"""Tests for the derived note-frequency tables and the table locator."""

from pysidtracker import (
    NTSC_FREQ_HI,
    NTSC_FREQ_LO,
    PAL_FREQ_HI,
    PAL_FREQ_LO,
    NoteFreqTable,
    SidImage,
    locate_note_freq,
)
from pysidtracker.notefreq import is_octave_ramp

from .helpers import build_prg


def test_table_lengths():
    for table in (PAL_FREQ_LO, PAL_FREQ_HI, NTSC_FREQ_LO, NTSC_FREQ_HI):
        assert len(table) == 96


def test_pal_middle_c():
    # Note index 49 is middle C (~261.6 Hz) on a PAL SID: register ~0x1167.
    table = NoteFreqTable.pal()
    assert abs(table.freq(49) - 0x1167) <= 1


def test_octave_ramp_is_monotonic():
    assert is_octave_ramp(PAL_FREQ_HI)
    assert not is_octave_ramp([0, 0, 0])
    assert not is_octave_ramp([1, 3, 2])  # decreasing


def test_ntsc_higher_than_pal():
    # A faster clock yields a smaller register for the same pitch.
    assert NoteFreqTable.ntsc().freq(49) < NoteFreqTable.pal().freq(49)


def test_locate_note_freq():
    load = 0x1000
    hi_addr = 0x1100
    length = 96
    lo_addr = hi_addr + length
    # Player code: LDA hi,X ; LDA lo,X (adjacent absolute-indexed operands).
    code = bytes(
        [0xBD, hi_addr & 0xFF, hi_addr >> 8, 0xBD, lo_addr & 0xFF, lo_addr >> 8]
    )
    body = bytearray(0x400)
    body[: len(code)] = code
    body[hi_addr - load : hi_addr - load + length] = PAL_FREQ_HI
    body[lo_addr - load : lo_addr - load + length] = PAL_FREQ_LO
    image = SidImage.from_prg(build_prg(bytes(body), load=load))

    table = locate_note_freq(image)
    assert table is not None
    assert table.addr == hi_addr
    assert table.freq(49) == NoteFreqTable.pal().freq(49)


def test_locate_note_freq_absent():
    image = SidImage.from_prg(build_prg(b"\xea" * 32, load=0x1000))
    assert locate_note_freq(image) is None
