"""A cursor sweeps one byte lane of a word, held in a cell or in a state."""

import re
from dataclasses import replace

import pytest

from pysidtracker.playspec import SpecError, elaborate, native, parse
from pysidtracker.playspec.emit import emit
from pysidtracker.playspec.outorder import inline
from pysidtracker.playspec.pyexec import compile_program, controls

from .test_playspec_native import run_both

SPEC = """spec lane;
layout { tab: addr; }
cell o: u8[2] @ 0x20;
cell pos: u8 @ 0x22;
cell cnt: u8 @ 0x23;
WORD
cursor t {
  table @ L.tab width 3;
  pos pos; count cnt; values word;
  on frame step;
  match c0 {
    0..0x7f => sweep(into=word.LANE, count=c0, step=c1);
    0xff => stop;
  }
}
machine m { step { fire frame; o[0] := lo(word); o[1] := hi(word); } }
schedule { m; }
"""
WORDS = {
    "cell": "cell word: u16 @ (0x24, 0x25);",
    "state": "state word: u16 = 0x10F0;",
}


def play(text, frames=6):
    """The words after each of ``frames`` calls, the executors checked to agree."""
    program = elaborate(parse(text), {"tab": 0x2000})
    memory = bytearray(0x10000)
    memory[0x2000:0x2006] = bytes([2, 0x20, 0, 0xFF, 0, 0])
    if "cell word" in text:
        memory[0x24:0x26] = bytes([0xF0, 0x10])
    if native.available():
        assert run_both(program, memory, frames) is None
    call, state, seen = compile_program(program), controls(program), []
    for _ in range(frames):
        call(memory.__getitem__, memory.__setitem__, state)
        seen.append(memory[0x20] | memory[0x21] << 8)
    return seen


@pytest.mark.strain_exempt(reason="asserts equality of two executors")
@pytest.mark.parametrize("where", WORDS)
@pytest.mark.parametrize(
    "lane,words",
    [
        ("lo", [0x1010, 0x1030, 0x1030, 0x1030, 0x1030, 0x1030]),
        ("hi", [0x30F0, 0x50F0, 0x50F0, 0x50F0, 0x50F0, 0x50F0]),
    ],
)
def test_a_sweep_moves_one_byte_of_a_word_and_carries_nowhere(where, lane, words):
    text = SPEC.replace("WORD", WORDS[where]).replace("LANE", lane)
    assert play(text) == words


@pytest.mark.strain_exempt(reason="the same coverage under each backend fixture")
def test_a_swept_lane_is_emitted_and_a_missing_high_byte_refused():
    text = SPEC.replace("WORD", WORDS["cell"]).replace("LANE", "hi")
    program = elaborate(parse(text), {"tab": 0x2000})
    flat = replace(program, body=inline(program.body, program.procs), procs={})
    source = emit(flat, "x")
    assert re.search(r"sweep\(into=\w+\.hi,", source)
    assert elaborate(parse(source), {}).objects
    with pytest.raises(SpecError, match="no high byte"):
        elaborate(
            parse(text.replace("u16 @ (0x24, 0x25)", "u8 @ 0x24")), {"tab": 0x2000}
        )
