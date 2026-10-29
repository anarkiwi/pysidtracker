"""Tests for the byte-stream grammar runtime (pysidtracker.bytegrammar)."""

from dataclasses import dataclass
from typing import Optional, Tuple

import pytest
from lark import Transformer, v_args

from pysidtracker import (
    REACH,
    ByteGrammar,
    FacetRegistry,
    Facet,
    Parsed,
    SidImage,
    SidParseError,
    make_package_errors,
    select_rule,
)
from pysidtracker.bytegrammar import render

_Err, ToyParseError, _ = make_package_errors("Toy")


class ToyValidationError(_Err):
    """A toy stream is not writable."""


@dataclass(frozen=True)
class Seq:
    """Items then an $FF end or a $FE n jump."""

    items: Tuple[Tuple[int, int], ...] = ()
    jump: Optional[int] = None

    def to_bytes(self) -> bytes:
        body = bytes(b for item in self.items for b in item)
        return body + (b"\xff" if self.jump is None else bytes((0xFE, self.jump)))


@dataclass(frozen=True)
class Blob:
    """Raw bytes whose parse is an event list, not a Blob."""

    data: bytes

    def to_bytes(self) -> bytes:
        return self.data


@v_args(inline=True)
class ToModel(Transformer):
    """Toy rules -> models."""

    # pylint: disable=missing-function-docstring

    def item(self, ident, value):
        return ord(ident), ord(value)

    def jump(self, _, target):
        return ord(target)

    def seq(self, *items):
        last = items[-1]
        model = Seq(tuple(items[:-1]), last if isinstance(last, int) else None)
        return Parsed(model, len(model.to_bytes()))

    seq_jump = seq

    def events(self, *tokens):
        return Parsed([ord(t) for t in tokens[:-1]], tokens[-1].end_pos)


TOY = ByteGrammar(
    "tests",
    "data/toy.lark",
    ("seq", "seq_jump", "events", "small"),
    ToModel(),
    ToyParseError,
)


def test_parse_prefix_stops_at_terminator():
    assert TOY.parse_prefix(b"\x01\x02\xff\x09\x09", "seq") == Parsed(Seq(((1, 2),)), 3)
    assert TOY.parse_prefix(b"\x03\x04\xfe\x00\xff", "seq_jump") == Parsed(
        Seq(((3, 4),), 0), 4
    )
    assert TOY.parse_prefix(b"\x05\x06\x00\x07", "events") == Parsed([5, 6], 3)


def test_build_variants_read_the_same_bytes_differently():
    assert TOY.parse_prefix(b"\xfe\x00\xff", "seq").model == Seq(((0xFE, 0),))
    assert TOY.parse_prefix(b"\xfe\x00\xff", "seq_jump").model == Seq((), 0)


def test_errors_wrap_into_the_callers_class():
    with pytest.raises(ToyParseError, match="seq has no terminator within 4 bytes"):
        TOY.parse_prefix(b"\x01\x02\x03\x04", "seq")
    with pytest.raises(SidParseError, match=r"small: unexpected byte \$80 at offset 2"):
        TOY.tree(b"\x01\x02\x80\xff", "small")


def test_default_error_is_sid_parse_error():
    plain = ByteGrammar("tests", "data/toy.lark", ("seq",), ToModel())
    with pytest.raises(SidParseError):
        plain.parse_prefix(b"", "seq")


def test_parse_exact_rejects_trailing_bytes():
    assert TOY.parse_exact(b"\x01\x02\xff", "seq") == Seq(((1, 2),))
    with pytest.raises(ToyParseError, match="seq ends after 3 of its 4 bytes"):
        TOY.parse_exact(b"\x01\x02\xff\x00", "seq")


def test_parsers_are_built_once():
    assert TOY.parser() is TOY.parser()
    assert TOY.parser(trees=True) is not TOY.parser()


def test_tree_and_render():
    tree = TOY.tree(b"\x01\x02\xfe\x07\x55", "seq_jump")
    assert list(render(tree)) == [
        "seq_jump",
        "  item ITEM_JUMP=$01 BYTE=$02",
        "  jump JUMP=$fe BYTE=$07",
    ]
    assert list(render(TOY.tree(b"\xff", "seq"))) == ["seq END=$ff"]


def _image(table_and_streams: dict) -> SidImage:
    img = SidImage.from_prg(b"\x00\x10" + bytes(0x200))
    for addr, data in table_and_streams.items():
        img.poke_bytes(addr, data)
    return img


def test_entry_reads_the_stream_a_pointer_table_names():
    img = _image({0x1100: b"\x01\x02\xff", 0x1110: b"\x00\x00\x00"})
    img.put_ptr(0x1000, 0x1004, 1, 0x1100)
    img.put_ptr(0x1000, 0x1004, 2, 0x1110)
    assert TOY.entry(img, 0x1000, 0x1004, 1, "seq") == Parsed(Seq(((1, 2),)), 3)
    with pytest.raises(ToyParseError, match=r"seq 2 at 0x1110: seq has no term"):
        TOY.entry(img, 0x1000, 0x1004, 2, "seq")
    with pytest.raises(ToyParseError, match="voice 2 at 0x1110"):
        TOY.entry(img, 0x1000, 0x1004, 2, "seq", what="voice 2", size=8)


def test_read_window_is_zero_padded_past_ffff():
    img = SidImage.from_prg(b"\xfd\xff\x05\x06\x07")
    assert img.window(0xFFFD, 5) == b"\x05\x06\x07\x00\x00"
    assert TOY.read(img, 0xFFFD, "events").model == [5, 6, 7]
    assert REACH == 0x100


def test_stored_is_none_outside_region_or_unparsable():
    img = _image({0x1100: b"\x01\x02\xff", 0x1180: b"\x01\x02\x03\x04"})
    img.put_ptr(0x1000, 0x1004, 0, 0x1100)
    img.put_ptr(0x1000, 0x1004, 1, 0x1180)
    img.put_ptr(0x1000, 0x1004, 2, 0x3000)
    region = range(0x1100, 0x1184)
    assert TOY.stored(img, 0x1000, 0x1004, 0, "seq", region) == Seq(((1, 2),))
    assert TOY.stored(img, 0x1000, 0x1004, 1, "seq", region) is None
    assert TOY.stored(img, 0x1000, 0x1004, 2, "seq", region) is None
    longer = range(0x1100, 0x1200)
    assert TOY.stored(img, 0x1000, 0x1004, 1, "seq", longer) is None


def test_check_round_trips_models():
    TOY.check(
        [("a", Seq(((1, 2),)), "seq"), ("b", Blob(b"\x05\x00"), "events")],
        ToyValidationError,
        skip_equality=("events",),
    )
    with pytest.raises(ToyValidationError, match="reads back as"):
        TOY.check([("b", Blob(b"\x05\x00"), "events")], ToyValidationError)
    with pytest.raises(ToyValidationError, match="a is not one seq stream"):
        TOY.check([("a", Seq((), 3), "seq")], ToyValidationError)


def test_check_catches_a_model_that_reparses_differently():
    with pytest.raises(ToyValidationError, match="ends after 2 of its 3 bytes"):
        TOY.check([("x", Seq(((0xFE, 0),)), "seq_jump")], ToyValidationError)


def test_listing():
    lines = list(TOY.listing([("voice 0", Seq(((1, 2),)), "seq")]))
    assert lines == ["voice 0:", "  seq", "    item ITEM=$01 BYTE=$02", "    END=$ff"]


@dataclass(frozen=True)
class _Resolved:
    jumps: bool
    tempo: bool = False


def test_select_rule_on_variants_mappings_and_attributes():
    rules = (("tempo", "seq_tempo"), ("jumps", "seq_jump"))
    reg = FacetRegistry([Facet("jumps", "build", default=True)])
    variant = reg.detect(None)
    assert select_rule(variant, rules, "seq") == "seq_jump"
    assert select_rule(variant.replace("jumps", False), rules, "seq") == "seq"
    assert select_rule({"tempo": 1, "jumps": 1}, rules, "seq") == "seq_tempo"
    assert select_rule(_Resolved(jumps=True), rules, "seq") == "seq_jump"
    assert select_rule(object(), rules, "seq") == "seq"
