"""Nibble fields ``x.hi`` / ``x.lo``: reads and writes of names, cells and view fields."""

from dataclasses import replace

import pytest

from pysidtracker.playspec import SpecError, elaborate, parse
from pysidtracker.playspec import outorder as O
from pysidtracker.playspec.emit import emit, isomorphic
from pysidtracker.playspec.pyexec import run

HEAD = (
    "spec t;\nlayout { a, b, p: addr; }\ncell v: u8 @ L.a;\ncell o: u8[4] @ L.b;\n"
    "cell w: u16 @ L.p;\nview rec @ w { x @ 1; }\n"
)
TAIL = "machine m {{ step {{ {} }} }}\nschedule {{ m; }}\n"
A, B, P = 0x1000, 0x1100, 0x1200


def program(stmts):
    return elaborate(parse(HEAD + TAIL.format(stmts)), {"a": A, "b": B, "p": P})


def outputs(prog, value, rec=0):
    mem = bytearray(0x2000)
    mem[A] = value
    mem[P], mem[P + 1] = 0x00, 0x13
    mem[0x1301] = rec
    run(prog, mem)
    return bytes(mem[B : B + 4]), mem[A]


def test_reads_are_the_shift_and_the_mask():
    new = program("o[0] := v.hi; o[1] := v.lo; o[2] := rec.x.hi; o[3] := rec.x.lo;")
    old = program(
        "o[0] := v >> 4; o[1] := v & 0x0F; o[2] := mem[0x1301] >> 4; o[3] := mem[0x1301] & 0x0F;"
    )
    assert isomorphic(new.body, old.body) is None
    assert outputs(new, 0xA7, 0x5C)[0] == bytes([0xA, 7, 5, 0xC])


@pytest.mark.parametrize("value", range(0, 256, 5))
def test_writes_replace_one_nibble_and_keep_the_other(value):
    prog = program("v.lo := v.hi; o[0] := v; v.hi := 3; o[1] := v;")
    out, _ = outputs(prog, value)
    assert out[:2] == bytes([(value & 0xF0) | (value >> 4), 0x30 | (value >> 4)])


def test_a_local_is_a_target_too():
    prog = program("let t = v; t.hi := 3; t.lo := t.hi; o[0] := t;")
    assert outputs(prog, 0xAB)[0][0] == 0x33


def test_the_emitted_spec_elaborates_to_the_same_program():
    prog = program("v.hi := v.lo; o[0] := rec.x.hi;")
    flat = replace(prog, body=O.inline(prog.body, prog.procs), procs={})
    back = elaborate(parse(emit(flat)), {"a": A, "b": B, "p": P})
    assert isomorphic(flat.body, O.inline(back.body, back.procs)) is None


def test_a_wide_operand_has_no_nibble():
    with pytest.raises(SpecError, match="nibble of a u8"):
        program("o[0] := w.hi;")


def test_compound_assignment_reads_then_writes_the_nibble():
    out, cell = outputs(program("v.hi += 1; v.lo |= 8;"), 0x2A)
    assert cell == 0x3A
