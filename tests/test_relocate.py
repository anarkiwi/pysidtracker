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


def test_walk_code_follows_calls_jumps_and_branches():
    code = walk_code(_image(), [0x1000], 0x1000, 0x1040)
    assert code == {
        0x1000: 3,
        0x1003: 3,
        0x1006: 1,
        0x1007: 2,
        0x100C: 3,
        0x1010: 3,
        0x1013: 2,
        0x1015: 3,
        0x1018: 1,
        0x1020: 3,
        0x1023: 1,
    }


def test_walk_code_stays_inside_bounds():
    code = walk_code(_image(), [0x1010, 0x0800], 0x1010, 0x1016)
    assert code == {0x1010: 3, 0x1013: 2, 0x1015: 3}


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


def test_relocate_operands_error_class_and_wrap():
    img = _image()
    code = {0x1020: 3}

    class Invalid(Exception):
        pass

    with pytest.raises(Invalid):
        relocate_operands(img, code, range(0), 0, forbidden={0x2050}, error=Invalid)
    assert relocate_operands(img, code, range(0x2000, 0x2100), -0x2100) == {
        0x1021: 0xFF50
    }
