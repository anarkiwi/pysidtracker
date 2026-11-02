"""The ``match`` statement: arms by value, list and range, static guards, default."""

from dataclasses import replace

import pytest

from pysidtracker.playspec import SpecError, elaborate, parse
from pysidtracker.playspec import outorder as O
from pysidtracker.playspec.emit import declared, emit, isomorphic
from pysidtracker.playspec.pyexec import run

HEAD = "spec t;\nlayout { a, b: addr; f, g: flag; }\ncell v: u8 @ L.a;\ncell o: u8[4] @ L.b;\n"
TAIL = "machine m {{ step {{ {} }} }}\nschedule {{ m; }}\n"
A, B = 0x1000, 0x1100

MATCH = """match v {
  1 => o[0] := 10;
  2, 3 => { o[0] := 20; o[1] += 1; }
  5..9 => o[0] := 30;
  4 if L.f => o[0] := 40;
  0x20 if L.g => o[0] := 50;
  7 => o[0] := 60;
  _ => { o[0] := 70; }
}"""
CHAIN = """if v == 1 {
  o[0] := 10;
} elif v == 2 || v == 3 {
  o[0] := 20; o[1] += 1;
} elif v >= 5 && v <= 9 {
  o[0] := 30;
} elif v == 4 && L.f {
  o[0] := 40;
} elif v == 0x20 && L.g {
  o[0] := 50;
} elif v == 7 {
  o[0] := 60;
} else {
  o[0] := 70;
}"""


def program(stmts, f=0, g=0):
    spec = parse(HEAD + TAIL.format(stmts))
    return elaborate(spec, {"a": A, "b": B, "f": f, "g": g})


def outputs(prog, value):
    mem = bytearray(0x2000)
    mem[A] = value
    run(prog, mem)
    return bytes(mem[B : B + 4])


@pytest.mark.parametrize("flags", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_match_is_the_hand_written_if_chain(flags):
    new, old = program(MATCH, *flags), program(CHAIN, *flags)
    assert isomorphic(new.body, old.body) is None
    assert all(outputs(new, v) == outputs(old, v) for v in range(256))


def test_arms_values_lists_ranges_and_first_match_wins():
    prog = program(MATCH, 1, 1)
    first = {v: outputs(prog, v)[0] for v in (1, 2, 3, 4, 5, 7, 9, 10, 0x20)}
    assert first == {1: 10, 2: 20, 3: 20, 4: 40, 5: 30, 7: 30, 9: 30, 10: 70, 0x20: 50}
    assert outputs(prog, 3)[1] == 1


def test_a_guard_off_drops_its_arm_at_bind_time():
    prog = program(MATCH)
    assert outputs(prog, 4)[0] == 70 and outputs(prog, 0x20)[0] == 70


def test_without_default_no_arm_leaves_the_state():
    prog = program("match v { 1 => o[0] := 9; }")
    assert outputs(prog, 1)[0] == 9 and outputs(prog, 2)[0] == 0


def test_only_a_default_runs_always():
    assert outputs(program("match v { _ => o[0] := 5; }"), 77)[0] == 5


def test_a_computed_value_is_evaluated_once():
    stmts = "match o[1] + 1 { 1 => o[0] := 1; 2 => o[0] := 2; _ => o[0] := 3; }"
    prog = program(stmts)
    mem = bytearray(0x2000)
    run(prog, mem)
    assert mem[B] == 1


def test_match_in_a_proc_with_return_arms():
    text = (
        HEAD
        + "proc f(n: u8) -> u8 { match n { 0 => return 4; 1..2 => return 5; _ => {} }"
        " return 6; }\n" + TAIL.format("o[0] := f(v);")
    )
    prog = elaborate(parse(text), {"a": A, "b": B, "f": 0, "g": 0})
    assert [outputs(prog, v)[0] for v in (0, 1, 2, 3)] == [4, 5, 5, 6]


def test_roundtrip_is_isomorphic():
    prog = program(MATCH, 1, 1)
    flat = replace(prog, body=declared(O.inline(prog.body, prog.procs)), procs={})
    source = emit(flat, "m")
    assert "match" not in source and "elif" in source
    back = elaborate(parse(source), {"a": A, "b": B, "f": 1, "g": 1})
    assert isomorphic(flat.body, O.inline(back.body, back.procs)) is None


@pytest.mark.parametrize(
    "stmts",
    [
        "match v { _ => o[0] := 1; 1 => o[0] := 2; }",
        "match v { 1 => }",
        "match v { 1..2..3 => o[0] := 1; }",
        "match { 1 => o[0] := 1; }",
    ],
)
def test_malformed_match_is_refused(stmts):
    with pytest.raises(SpecError):
        parse(HEAD + TAIL.format(stmts))
