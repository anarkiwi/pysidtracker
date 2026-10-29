"""Tests for bitmap-selected fields in byte grammars (BitmapPostLex)."""

import pytest
from lark import Transformer, v_args

from pysidtracker import BitField, ByteGrammar, Parsed, SidParseError
from pysidtracker.bytegrammar import render

LOW = ((6, "T_WGH", 1), (7, "T_WGL", 1), (5, "T_AD", 1), (4, "T_SR", 1))
LOW += ((3, "T_TR", 1), (2, "T_AF", 1), (1, "T_PW", 1))
HIGH = ((7, "T_PS", 1), (6, "T_RE", 1), (5, "T_FV", 1), (4, "T_CP", 1))
HIGH += ((3, "T_ACID", 2),)


@v_args(inline=True)
class ToFields(Transformer):
    """Rows -> ``{tag: value}`` with the bytes they span."""

    # pylint: disable=missing-function-docstring

    def field(self, tag, *values):
        return tag.type, int.from_bytes(bytes(map(ord, values)), "little")

    def low(self, _bitmap, *fields):
        return fields

    high = low

    def row(self, low, high):
        fields = dict(low + high)
        widths = dict((tag, width) for _, tag, width in LOW + HIGH)
        return Parsed(fields, 2 + sum(widths[tag] for tag in fields))


ROWS = ByteGrammar(
    "tests",
    "data/bitmap.lark",
    ("row",),
    ToFields(),
    bitmaps={"LOWMAP": LOW, "HIGHMAP": HIGH},
)


def reference(raw: bytes):
    """The fields and size of ``raw`` decoded bit by bit, as a player does."""
    fields, pos = {}, 0
    for spec in (LOW, HIGH):
        bitmap = raw[pos]
        pos += 1
        for bit, tag, width in spec:
            if bitmap >> bit & 1:
                fields[tag] = int.from_bytes(raw[pos : pos + width], "little")
                pos += width
    return fields, pos


def _row(low: int, high: int) -> bytes:
    values = iter(range(0x10, 0x30))
    out = [low]
    out += [next(values) for bit, _, _ in LOW if low >> bit & 1]
    out.append(high)
    out += [next(values) for bit, _, w in HIGH if high >> bit & 1 for _ in range(w)]
    return bytes(out) + b"\xaa" * 4


@pytest.mark.parametrize("which", ["low", "high"])
def test_every_bitmap_matches_the_reference_decoder(which):
    for bitmap in range(0x100):
        raw = _row(bitmap, 0x5A) if which == "low" else _row(0xA5, bitmap)
        assert ROWS.parse_prefix(raw, "row") == Parsed(*reference(raw)), raw.hex()


def test_unused_bits_select_nothing():
    assert ROWS.parse_prefix(b"\x01\x07", "row") == Parsed({}, 2)


def test_values_are_lexed_in_the_parser_state():
    raw = bytes((0x40, 0x01, 0x08, 0x34, 0x12))
    assert ROWS.parse_prefix(raw, "row") == Parsed({"T_WGH": 1, "T_ACID": 0x1234}, 5)


def test_truncated_values_have_no_terminator():
    with pytest.raises(SidParseError, match="row has no terminator within 3 bytes"):
        ROWS.parse_prefix(b"\xc0\x01\x02", "row")
    with pytest.raises(SidParseError, match="row has no terminator within 3 bytes"):
        ROWS.parse_prefix(b"\x00\x08\x01", "row")


def test_tags_render_without_a_byte():
    lines = list(render(ROWS.tree(b"\x40\x11\x10\x22\x99", "row")))
    assert lines == [
        "row",
        "  low",
        "    LOWMAP=$40",
        "    field T_WGH BYTE=$11",
        "  high",
        "    HIGHMAP=$10",
        "    field T_CP BYTE=$22",
    ]


def test_bitfield_defaults_to_one_byte():
    assert BitField(3, "T_X") == (3, "T_X", 1)
