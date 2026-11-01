"""Tests for the 6502 code walk and operand relocation (pysidtracker.relocate)."""

import pytest

from pysidtracker import SidImage, SidParseError, relocate_operands, walk_code

CODE = {
    0x1000: "ad 00 11",  # LDA $1100
    0x1003: "20 10 10",  # JSR $1010
    0x1006: "18",  # CLC
    0x1007: "90 03",  # BCC $100C (forced: the fall-through is data)
    0x1009: "8d 00 20",  # never reached
    0x100C: "4c 20 10",  # JMP $1020
    0x1010: "8d 00 30",  # STA $3000
    0x1013: "d0 03",  # BNE $1018
    0x1015: "ee 00 10",  # INC $1000 (falls through)
    0x1018: "60",  # RTS
    0x1020: "ad 50 20",  # LDA $2050
    0x1023: "02",  # JAM
    0x1024: "ea",  # never reached
}


def _image() -> SidImage:
    img = SidImage.from_prg(b"\x00\x10" + bytes(0x40))
    for addr, text in CODE.items():
        img.poke_bytes(addr, bytes.fromhex(text))
    return img


def test_relocate_operands_moves_remaps_and_refuses():
    img = _image()
    code = walk_code(img, [0x1000], 0x1000, 0x1040)
    moved = range(0x1000, 0x1040)
    out = relocate_operands(img, code, moved, 0x2000, remap={0x3000: 0x4567})
    assert out == {
        0x1004: 0x3010,
        0x100D: 0x3020,
        0x1011: 0x4567,
        0x1016: 0x3000,
    }
    with pytest.raises(SidParseError, match="operand at 0x1021 reaches 0x2050"):
        relocate_operands(img, code, moved, 0x2000, forbidden=range(0x2000, 0x2100))
