"""``notes(part, K)`` on a ``ro`` state array: its own items, or the standard table."""

import pytest

from pysidtracker.notes import frequency
from pysidtracker.playspec import SpecError, compile_program, elaborate, parse
from pysidtracker.playspec.pyexec import controls

from .conftest import BACKENDS

PAL = 985248
SPEC = """spec t;
layout { out: addr; words, bytes: list; }
cell o: u16[3] @ L.out;
state pitch: u16[] ro notes(word, -24) = L.words;
state lows: u8[] ro notes(lo, -24) = L.bytes;
machine m {
  state i: u8 = 0;
  step {
    o[0] := pitch[i + 24];
    o[1] := u16(lows[i + 24]);
    o[2] := pitch[i + 1] - pitch[i];
    i += 1;
  }
}
schedule { m; }
"""
WORDS = [(37 * i + 11) & 0xFFFF for i in range(64)]
BINDING = {"out": 0x2000, "words": WORDS, "bytes": [w & 0xFF for w in WORDS]}


def outputs(standard, calls=3):
    memory = bytearray(0x10000)
    program = elaborate(parse(SPEC), BINDING, standard=standard)
    frame, state = compile_program(program), controls(program)
    seen = []
    for _ in range(calls):
        frame(memory.__getitem__, memory.__setitem__, state, None, {})
        seen.append(
            tuple(
                memory[0x2000 + 2 * k] | memory[0x2001 + 2 * k] << 8 for k in range(3)
            )
        )
    return seen


P = "state pitch: u16[] ro notes(word, -24) = L.words;"
B = "state lows: u8[] ro notes(lo, -24) = L.bytes;"
MALFORMED = (
    (P, "state pitch: u16[] notes(word, -24) = L.words;"),
    (P, "state pitch: u16[] ro notes(lo, -24) = L.words;"),
    (B, "state lows: u8[] ro notes(word, -24) = L.bytes;"),
    (P, "state pitch: u16 notes(word, -24) = 0;"),
)


@pytest.fixture(autouse=True)
def one_engine(backend):
    """The items are bound before either engine runs: test on one."""
    if backend != BACKENDS[0]:
        pytest.skip("bound before the engine runs")


def test_items_are_the_native_table_or_the_standard_one():
    assert outputs(None) == [
        (WORDS[i + 24], WORDS[i + 24] & 0xFF, (WORDS[i + 1] - WORDS[i]) & 0xFFFF)
        for i in range(3)
    ]
    std = [frequency(i - 24, PAL) for i in range(64)]
    assert outputs(PAL) == [
        (frequency(i, PAL), frequency(i, PAL) & 0xFF, (std[i + 1] - std[i]) & 0xFFFF)
        for i in range(3)
    ]
    for old, new in MALFORMED:
        with pytest.raises(SpecError, match="notes needs a ro array|needs an array"):
            elaborate(parse(SPEC.replace(old, new)), BINDING)
