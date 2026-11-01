"""Emitting reordered programs as playspec source, and proving them call by call."""

import pytest

from pysidtracker.playspec import elaborate, parse
from pysidtracker.playspec import outorder as O
from pysidtracker.playspec import outproof
from pysidtracker.playspec.emit import EmitError, declared, emit, isomorphic
from pysidtracker.playspec.ir import K, V

from tests.test_playspec_outorder import BIND, HEAD, ORDER, PLAYER, STALE, program

VOLATILE = HEAD + """cell osc: u8 @ L.sid + 0x1B volatile;
cell tab: u8[16] @ L.w + 0x40 ro notes(lo, 0);
machine m {
  step {
    let a = osc;
    let b: u16 = u16(a) + 0x300;
    while tick != 0 bound 4 {
      tick -= 1;
    }
    chip[0] := tab[a & 0x0F];
    chip[1] := ite(a < 3, 7, 9);
    chip[2] := lo(b) ^ hi(b);
    if slt(a, 0x10) || !(a == b - b) {
      chip[3] := -a;
      assert a != 0x80;
    } elif a > 2 {
      chip[4] := ~a;
    }
  }
}
schedule { m; }
"""


@pytest.mark.parametrize("text", [PLAYER, STALE, VOLATILE])
def test_emitted_spec_elaborates_to_the_reordered_program(text):
    result = O.reorder(program(text), ORDER)
    source = emit(result.program, "ordered")
    back = elaborate(parse(source), BIND)
    assert isomorphic(result.program.body, O.inline(back.body, back.procs)) is None
    for standard in (None, 985248):
        again = elaborate(parse(source), {}, standard=standard)
        assert again.body


def test_emitted_names_are_readable():
    source = emit(O.reorder(program(STALE), ORDER).program, "ordered")
    assert "let out_v0_pw_lo: u8 = pw;" in source
    assert "sid[V0_AD] := pw;" in source and "const V0_AD = 5;" in source


def test_isomorphic_reports_the_first_difference():
    a = [("set", "x", K(1, 8)), ("set", "y", V("x", 8))]
    assert isomorphic(a, [("set", "p", K(1, 8)), ("set", "q", V("p", 8))]) is None
    assert "assigns" in isomorphic(a, [("set", "p", K(1, 8)), ("set", "p", V("p", 8))])
    assert "statements" in isomorphic(a, a[:1])
    assert "wr" in isomorphic(a, [("wr", K(1, 16), K(1, 8)), a[1]])
    loop = [("loop", K(1, 1), [], 3, 1)]
    assert "bound" in isomorphic(loop, [("loop", K(1, 1), [], 4, 9)])


def test_declared_initialises_a_variable_read_before_it_is_assigned():
    body = [("if", V("c", 1), [("set", "x", K(1, 8))], []), ("set", "y", V("x", 8))]
    out = declared(body, ["c"])
    assert out[0] == ("set", "x", K(0, 8)) and out[1:] == body


def test_emit_refuses_what_it_cannot_state():
    prog = program(STALE)
    with pytest.raises(EmitError):
        emit(prog)
    flat = O.replace(prog, body=[("native", K(1, 16), (), K(2, 16))], procs={})
    with pytest.raises(EmitError):
        emit(flat)


@pytest.mark.parametrize("text", [PLAYER, STALE, VOLATILE])
def test_prove_discharges_every_obligation(text):
    proof = outproof.prove(program(text), ORDER, timeout_ms=10_000)
    assert proof.holds and proof.open == []
    assert proof.passes["forward"].total > 0


def test_a_store_after_its_flush_is_found():
    slot = (5, 1)
    placed = [O.flush(slot), ("set", O.out_var(slot).n, K(1, 8))]
    assert outproof.flushed_last(placed, [slot]) == O.flag_var(slot).n
    assert outproof.flushed_last(placed[:1], [slot]) is None


def test_census_lists_what_is_left():
    result = O.reorder(program(PLAYER), ORDER)
    result.text = emit(result.program, "ordered")
    found = O.census(result)
    assert found["voices"] == [0, 1, 2] and found["song_registers"] == [21, 22, 23, 24]
    assert found["finish"] and found["promoted_cells"] == 3 and found["counters"] == 4


def test_a_spent_budget_leaves_obligations_open():
    proof = outproof.prove(program(PLAYER), ORDER, budget_s=0)
    assert not proof.holds and any("unchecked" in item for item in proof.open)


def test_obligations_discharge_the_same_over_processes():
    stages = {}
    result = O.reorder(program(PLAYER), ORDER, proof=stages)
    sizes = outproof.var_widths(stages["nested"], stages["placed"], stages["forwarded"])
    cond = next(item for item in stages["forward"] if item[0] == "cond")
    wrong = ("cond", cond[1], cond[2], O.fold("lnot", 1, (cond[3],)))
    items = stages["forward"] + [wrong]
    found = [
        outproof.discharge(
            items, sizes, result.ranges, {}, timeout_ms=10_000, jobs=jobs
        )
        for jobs in (1, 2)
    ]
    assert found[0] == found[1] and found[1].failed.startswith(
        f"obligations {len(items) - 1}: cond"
    )
