"""Procs and shared subexpressions through :func:`~pysidtracker.playspec.outorder.reorder`:
a proc called from several sites stays a call, and the result still plays, proves and
emits as the original."""

import pytest

from pysidtracker.playspec import elaborate, parse
from pysidtracker.playspec import outorder as O
from pysidtracker.playspec import outproof
from pysidtracker.playspec.emit import emit, isomorphic, pruned
from pysidtracker.playspec.ir import K, Op, Rd, V

from .test_playspec_native import run_both
from .test_playspec_outorder import BIND, HEAD, ORDER, play

pytestmark = pytest.mark.strain_exempt(
    reason="asserts the shape and the writes of reordered procs, which coverage cannot see"
)

PROCS = HEAD + """cell tmp: u8 @ L.w + 12;
cell tab: u8[8] @ L.w + 32;
machine voice[x in 0..3] {
  cell n: u8 @ L.w + 16 + x;
  proc bump() {
    n += 3;
    if n & 0x80 != 0 { n := n ^ 0x55; }
    n := n + (n >> 3);
    n := n ^ (n << 1);
    n += tmp;
    if n == 0 { n := 1; }
  }
  proc show() { chip[7 * x + 3] := tmp ^ n; bump(); chip[7 * x + 2] := n; }
  step {
    tmp := tick;
    let before = n + 1;
    bump();
    show();
    chip[7 * x + 1] := before;
    tmp := n;
    bump();
    show();
  }
}
machine clock { step { tick += 1; } }
schedule { clock; for v in 0..3 { voice[v]; } }
"""

SHARED = HEAD + """cell tab: u8[8] @ L.w + 32;
machine m {
  step {
    let i = tick & 7;
    chip[0] := tab[i] + tab[i + 1] + 1;
    chip[1] := tab[i] + tab[i + 1] + 1;
    tab[i] := 5;
    chip[2] := tab[i] + tab[i + 1] + 1;
    chip[3] := tab[i] + tab[i + 1] + 1;
    tick += 1;
  }
}
schedule { m; }
"""


def build(text):
    """``text`` elaborated against the test binding."""
    return elaborate(parse(text), BIND)


def memory_of(seed=7):
    """Memory with a table and varied cells."""
    mem = bytearray(0x10000)
    for a in range(0x1000, 0x1040):
        mem[a] = (a * seed + 3) & 0xFF
    return bytes(mem)


def calls(body):
    """The ``call`` statements of ``body``."""
    return [s for s in O.statements(body) if s[0] == "call"]


@pytest.fixture(name="procs")
def fixture_procs():
    """The original program and its reordering."""
    original = build(PROCS)
    return original, O.reorder(original, ORDER)


def test_a_proc_with_several_call_sites_stays_a_call(procs, monkeypatch):
    original, result = procs
    assert len(result.program.procs) == 6 and len(calls(result.program.body)) == 12
    monkeypatch.setattr(O, "outlined", lambda body, procs: frozenset())
    assert result.after.statements < O.reorder(original, ORDER).after.statements
    assert not calls(O.inline(result.program.body, result.program.procs))
    assert O.inline(original.body, original.procs) != result.program.body


def test_outlined_is_what_a_call_is_smaller_than():
    one = [("set", "a", K(1, 8)), ("set", "b", K(1, 8))]
    procs = {"p": one, "q": [("call", "p")] + one, "r": one[:1]}
    body = [("call", "p"), ("call", "q"), ("call", "q"), ("call", "r"), ("call", "r")]
    assert O.outlined(body, procs) == {"q"}
    assert O.outlined(body[:1], procs) == frozenset()
    assert O.reached(body, procs) == ["r", "q", "p"]
    stop = {"p": one + [("finish",)], "q": [("call", "p")] * 3}
    assert O.outlined([("call", "q")] * 3, stop) == frozenset()


def test_the_result_plays_the_original_writes_in_order(procs):
    original, result = procs
    mem = memory_of()
    old = play(original, mem, 6)[1]
    new = play(result.program, mem, 6)[1]
    slots = result.slots
    assert new == [O.kept(w, slots, result.native) for w in old]
    assert run_both(result.program, bytearray(mem), 6) is None


def test_a_value_read_before_a_call_is_not_forwarded_past_what_it_changes(procs):
    original, result = procs
    mem = memory_of(11)
    old, new = play(original, mem, 3)[0], bytearray(play(result.program, mem, 3)[0])
    assert all(old[a] == new[a] for a in range(0x10000) if a not in result.promoted)
    effect = O.Effects(result.program.procs)
    assert any(effect(s).stores for s in calls(result.program.body))


def test_the_result_is_proved_and_checked(procs):
    original, result = procs
    assert O.check(original, result) is None
    proof = outproof.prove(original, ORDER, timeout_ms=20_000)
    assert proof.holds and proof.open == []


def test_a_wrong_proc_is_found(procs):
    original, result = procs
    pid = next(iter(result.program.procs))
    bad = [("wr", K(0x1010, 16), K(1, 8))] + result.program.procs[pid]
    result.program = O.replace(result.program, procs={**result.program.procs, pid: bad})
    assert O.check(original, result) is not None


def test_emitted_procs_elaborate_to_the_same_program(procs):
    _original, result = procs
    text = emit(result.program, "ordered")
    back = build(text)
    assert len(back.procs) == len(pruned(result.program).procs) + 1
    flat = pruned(result.program)
    assert (
        isomorphic(
            O.inline(flat.body, flat.procs),
            O.inline(back.body, back.procs),
            (result.program.objects, back.objects),
        )
        is None
    )
    assert text.count("proc ") == len(back.procs) - 1


def test_variables_shared_with_a_proc_are_declared_by_the_body():
    body = [("call", "p"), ("set", "y", V("x", 8)), ("set", "z", V("w", 8))]
    procs = {"p": [("set", "x", K(1, 8)), ("set", "v", V("w", 8))]}
    summary = {"p": O.exposure(procs["p"], frozenset(), {})}
    assert summary["p"] == ({"w"}, {"x", "v"})
    assert O.exposure(body, frozenset(), summary) == ({"w"}, {"x", "v", "y", "z"})
    assert O.shared(body, procs) == {"x", "w"}


def test_dead_work_in_a_proc_is_removed_and_an_empty_proc_is_not_called():
    procs = {
        "p": [("set", "a", K(1, 8)), ("set", "b", K(2, 8))],
        "q": [("set", "c", K(3, 8))],
    }
    body = [("call", "p"), ("call", "q"), ("set", "out", V("a", 8))]
    live = frozenset({"out"})
    got, kept = O.eliminate_all(body, procs, live, live)
    assert kept == {"p": [("set", "a", K(1, 8))]} and ("call", "q") not in got
    assert O.proc_reads({"r": [("set", "a", V("t", 8)), ("call", "p")], **procs})[
        "r"
    ] == {"t"}


def test_shared_subexpressions_are_computed_once_until_a_store_changes_them():
    original = build(SHARED)
    result = O.reorder(original, ORDER)
    temps = [
        s for s in result.program.body if s[0] == "set" and s[1].startswith("cse_")
    ]
    assert 2 <= len(temps) <= 3
    assert all(not isinstance(s[2], V) for s in temps)
    reads = [
        n
        for s in result.program.body
        for e in O.stmt_exprs(s)
        for n in O.walk(e)
        if isinstance(n, Rd) and not isinstance(n.a, K)
    ]
    assert len(reads) <= 2 * len(temps)
    mem = memory_of(5)
    assert play(result.program, mem, 8)[0] == play(original, mem, 8)[0]
    assert O.check(original, result) is None
    assert outproof.prove(original, ORDER, timeout_ms=20_000).holds


def test_shared_expressions_pick_what_repays_a_variable():
    word = Op("add", 8, (Rd(K(0x10, 16)), K(1, 8)))
    big = Op("add", 8, (Op("add", 8, (word, Rd(Op("zext", 16, (V("i", 8),))))), word))
    body = [("set", "a", big), ("set", "b", big), ("set", "c", word)]
    assert {big, word} <= O.shared_exprs(body, {})
    assert O.shared_exprs(body[:1], {}) == {word}
    assert O.shared_exprs(body[:1], {"p": body[1:2]}) >= {word, big}
