"""Tests for the text-grammar runtime over a toy typed model."""

from dataclasses import dataclass
from itertools import product

import pytest

from pysidtracker import make_package_errors
from pysidtracker.textgrammar import (
    ModelTransformer,
    TextGrammar,
    classes,
    hexint,
    rule_name,
    seq,
    term,
)

from . import toytext
from .toytext import Jump, Named, Step, Stop, WaveNote

_, ToyError, _ = make_package_errors("ToyText")


@dataclass
class SetAD:
    """A dataclass outside the toy module."""

    value: int


class Build(ModelTransformer):
    """The toy program: entries numbered from 1."""

    # pylint: disable=missing-function-docstring
    classes = classes(toytext)

    def entry(self, children):
        return tuple(children)

    def start(self, children):
        return seq(children, 1, "entry", ToyError)


GRAMMAR = TextGrammar("tests", "data/toytext.lark", Build, ToyError)

INTS = (0, 1, 0x7F, 0xFF, 0x1234, -1, -0x80)
JUMPS = tuple(map(Jump, INTS))
NOTES = tuple(WaveNote(n, r) for n, r in product(INTS, (False, True)))
NAMES = ("", "a b", 'q"uote', "café", "back\\slash", "#not a comment")
OPS = (
    Stop(),
    *JUMPS,
    *NOTES,
    *(Step(j, n) for j, n in product((None, *JUMPS), (None, *NOTES))),
    *map(Named, NAMES),
)


def dumps(program) -> str:
    """The toy program as text, one numbered entry a line."""
    return "".join(f"{hexint(n)}: {term(op)}\n" for n, op in enumerate(program, 1))


def test_rule_names():
    assert rule_name(SetAD) == "set_ad" and rule_name(WaveNote) == "wave_note"
    assert set(classes(toytext)) == {"stop", "jump", "wave_note", "step", "named"}
    assert classes(toytext, SetAD)["set_ad"] is SetAD


@pytest.mark.parametrize("op", OPS, ids=term)
def test_each_constructor_round_trips(op):
    assert GRAMMAR.loads(dumps([op])) == (op,)


def test_program_round_trips():
    assert GRAMMAR.loads(dumps(OPS)) == OPS


@pytest.mark.parametrize(
    "op,text",
    [
        (Stop(), "Stop"),
        (Jump(-0x10), "Jump(-$10)"),
        (WaveNote(3, True), "WaveNote($03, relative)"),
        (WaveNote(3), "WaveNote($03)"),
        (Step(None, WaveNote(1)), "Step(none, WaveNote($01))"),
        (Named('a"'), 'Named("a\\"")'),
    ],
)
def test_term(op, text):
    assert term(op) == text


def test_loads_reads_decimal_and_comments():
    text = "# a program\n1: Jump(16)  # decimal\n2: Jump(-$10)\n$03: Stop\n"
    assert GRAMMAR.loads(text) == (Jump(16), Jump(-16), Stop())


@pytest.mark.parametrize(
    "text,message",
    [
        ("1: Stop\n3: Stop\n", "entry 3 where 2 is due"),
        ("1: Jump($10000)\n", "jump target 65536 out of range"),
        ("1: Halt\n", "No terminal matches 'H'"),
    ],
)
def test_loads_errors(text, message):
    with pytest.raises(ToyError, match=message):
        GRAMMAR.loads(text)


def test_too_many_values():
    class Narrow(Build):
        """``jump`` rules build a no-field dataclass."""

        classes = {**Build.classes, "jump": Stop}

    with pytest.raises(ToyError, match="Stop takes 0 values"):
        TextGrammar("tests", "data/toytext.lark", Narrow, ToyError).loads("1: Jump(1)")


def test_rule_without_a_class_stays_a_tree():
    class Bare(ModelTransformer):
        """No model classes: every rule is a tree."""

    tree = TextGrammar("tests", "data/toytext.lark", Bare).loads("1: Stop")
    assert tree.data == "start" and tree.children[0].children[0] == 1
