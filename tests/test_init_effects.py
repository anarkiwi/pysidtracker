"""Tests for init effects and the song->subtune selection they yield.

The stubs are assembled inits, one per wrapper shape: a direct store to a
cell, a table remap, a per-song module copy, an out-of-range song, and a
selection that aliases.
"""

import pytest

from pysidtracker import (
    CellRead,
    CellWrite,
    InitEffects,
    Jumps,
    SidImage,
    SidParseError,
    init_diff,
    init_effects,
    subtune_selection,
)
from pysidtracker.asm6502 import assemble

from .helpers import build_prg, build_psid

jennings = pytest.importorskip("jennings")

ORG = 0x1000
CELL = 0x0500


def _image(source: str, songs: int = 1) -> tuple:
    code, labels = assemble(source, ORG)
    data = build_psid(code, load=ORG, init=ORG, play=ORG, songs=songs)
    return SidImage.from_bytes(data), labels


def test_direct_sta_cell():
    image, _ = _image(f"\n    STA {CELL:#x}\n    RTS\n")
    effects = init_effects(image, 3)
    assert isinstance(effects, InitEffects)
    assert effects.song == 3
    assert effects.writes == {CELL: CellWrite(3, ORG)}
    assert effects.reads == {}
    assert effects.jumps == Jumps()
    assert init_diff(image, [1]) == set()
    assert image.mem[CELL] == 0  # the caller's image is never mutated


def test_direct_selection_and_diff():
    image, _ = _image(f"\n    STA {CELL:#x}\n    RTS\n", songs=3)
    assert subtune_selection(image, [1, 2, 3], CELL) == {1: 1, 2: 2, 3: 3}
    assert init_diff(image, [1, 2, 3]) == {CELL}


def test_table_remap_indexes_the_song():
    image, labels = _image(
        """
    TAX
    look:
    LDA tbl,X
    pick:
    STA cell
    RTS
tbl:
    .byte 0, 9, 11, 13
cell:
    .byte 0
""",
        songs=4,
    )
    effects = init_effects(image, 2)
    assert effects.writes == {labels["cell"]: CellWrite(11, labels["pick"])}
    assert effects.reads == {labels["tbl"] + 2: [CellRead(11, labels["look"])]}
    assert subtune_selection(image, [1, 2, 3, 4], labels["cell"]) == {
        1: 9,
        2: 11,
        3: 13,
        4: 0,
    }
    assert init_diff(image, [1, 2, 3, 4]) == {labels["cell"]}


def test_module_copy_per_song():
    image, labels = _image(
        """
    CMP #1
    BEQ mod1
    CMP #2
    BEQ mod2
    RTS
mod1:
    LDA #$11
    put1:
    STA d0
    LDA #$12
    put2:
    STA d1
    RTS
mod2:
    LDA #$21
    put3:
    STA d0
    LDA #$22
    put4:
    STA d1
    RTS
d0:
    .byte 0
d1:
    .byte 0
""",
        songs=2,
    )
    one = init_effects(image, 1)
    two = init_effects(image, 2)
    assert one.writes == {
        labels["d0"]: CellWrite(0x11, labels["put1"]),
        labels["d1"]: CellWrite(0x12, labels["put2"]),
    }
    assert two.writes == {
        labels["d0"]: CellWrite(0x21, labels["put3"]),
        labels["d1"]: CellWrite(0x22, labels["put4"]),
    }
    assert init_diff(image, [1, 2]) == {labels["d0"], labels["d1"]}
    assert subtune_selection(image, [1, 2], labels["d0"]) == {1: 0x11, 2: 0x21}


def test_out_of_range_song_leaves_a_value():
    # the table has entries for songs 1 and 2 only; song 3 reads the byte past it
    image, labels = _image(
        """
    TAX
    look:
    LDA tbl,X
    pick:
    STA cell
    RTS
tbl:
    .byte 0, 5, 6
cell:
    .byte 0
""",
        songs=3,
    )
    out = init_effects(image, 3)
    assert out.writes == {labels["cell"]: CellWrite(0, labels["pick"])}
    assert out.reads == {labels["cell"]: [CellRead(0, labels["look"])]}
    assert subtune_selection(image, [1, 2, 3], labels["cell"]) == {1: 5, 2: 6, 3: 0}
    with pytest.raises(SidParseError, match="out of range"):
        subtune_selection(image, [1, 2, 4], labels["cell"])


def test_aliasing_selection_raises():
    image, labels = _image(
        """
    LDA #$07
    pick:
    STA cell
    RTS
cell:
    .byte 0
""",
        songs=2,
    )
    with pytest.raises(SidParseError, match="alias"):
        subtune_selection(image, [1, 2], labels["cell"])


def test_unwritten_cell_keeps_its_preinit_value():
    image, labels = _image("""
    LDA #$09
    STA other
    RTS
other:
    .byte 0
quiet:
    .byte 0
""")
    assert subtune_selection(image, [1], labels["quiet"]) == {1: 0}
    assert subtune_selection(image, [1], labels["other"]) == {1: 0x09}


def test_reads_are_recorded_per_access_in_order():
    image, labels = _image("""
    first:
    LDA quiet
    second:
    LDA quiet
    RTS
quiet:
    .byte 0
""")
    effects = init_effects(image, 1)
    assert effects.reads == {
        labels["quiet"]: [CellRead(0, labels["first"]), CellRead(0, labels["second"])]
    }


def test_watch_limits_recorded_cells():
    image, labels = _image("""
    LDA #$11
    STA c1
    LDA #$22
    STA c2
    RTS
c1:
    .byte 0
c2:
    .byte 0
""")
    assert set(init_effects(image, 1).writes) == {labels["c1"], labels["c2"]}
    watched = init_effects(image, 1, watch=[labels["c1"]])
    full = init_effects(image, 1)
    assert watched.writes == {labels["c1"]: full.writes[labels["c1"]]}
    assert labels["c2"] not in watched.reads


def test_installs_vectors():
    image, _ = _image("""
    LDA #$24
    STA $DC04
    LDA #$06
    STA $DC05
    LDA #$00
    STA $0314
    LDA #$80
    STA $0315
    LDA #$C3
    STA $FFFE
    LDA #$FF
    STA $FFFF
    RTS
""")
    jumps = init_effects(image, 1).jumps
    assert jumps.irq_vector == 0x8000
    assert jumps.hw_irq_vector == 0xFFC3
    assert jumps.cia1_timer_latch == 0x0624
    assert jumps.cia2_timer_latch is None


def test_stack_page_is_never_recorded_even_when_written():
    image, _ = _image("""
    LDA #$5A
    STA $01FF
    STA $01FE
    RTS
""")
    effects = init_effects(image, 1)
    assert 0x01FF not in effects.writes
    assert 0x01FE not in effects.writes


def test_requires_header():
    code, _ = assemble("\n    STA $0500\n    RTS\n", ORG)
    image = SidImage.from_bytes(build_prg(code, load=ORG))
    with pytest.raises(SidParseError, match="no SID header"):
        init_effects(image, 1)
    with pytest.raises(SidParseError, match="no SID header"):
        subtune_selection(image, [1], CELL)
    with pytest.raises(SidParseError, match="no SID header"):
        init_diff(image, [1, 2])
