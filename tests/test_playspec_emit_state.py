"""Emitting programs with hidden controls: each is a ``state``, and the emitted spec plays
the same calls as the program."""

import re
from dataclasses import replace

import pytest

from pysidtracker.playspec import elaborate, native, parse
from pysidtracker.playspec import outorder as O
from pysidtracker.playspec.emit import EmitError, emit, isomorphic, pruned
from pysidtracker.playspec.ir import K, V
from pysidtracker.playspec.pyexec import compile_program, controls

from .test_playspec_events import BINDING, FULL, STATED_FULL, STATED_WALKER, TABLE
from .test_playspec_native import run_both, sid_writes_of, table_memory

SCALARS = """spec s;
layout { starts: list; }
cell chip: u8[25] @ 0xD400 out;
cell o: u8[8] @ 0x30;
state n: u8 = 3;
state w: u16 = 0x1234;
state f: bool = true;
state t: u12 = 0x123;
machine m[x in 0..2] {
  state k: u8 = L.starts[x];
  step {
    chip[7 * x] := n + k;
    k += 2;
    chip[7 * x + 1] := lo(w);
    chip[7 * x + 2] := hi(w) ^ k;
    if f { o[x] := n; }
    o[2 + x] := lo(t);
  }
}
machine tick {
  step { n += 1; w += 0x101; f := !f; t := t + 5; }
}
schedule { tick; for v in 0..2 { m[v]; } }
"""
HIDDEN = """spec h;
layout { fr: addr; }
cell chip: u8[25] @ 0xD400 out;
cell v: u8 @ L.fr;
machine m {
  control phase { low = 0, high = 1 }
  step {
    in phase { low => { v += 1; goto high; } high => { v += 2; goto low; } }
    chip[0] := v;
  }
}
schedule { m; }
"""
WALK = STATED_WALKER.replace(
    "o: u8[5] @ 0x20;", "o: u8[5] @ 0x20;\ncell chip: u8[25] @ 0xD400 out;"
)


def build(text, **binding):
    """``text`` elaborated against ``binding``."""
    return elaborate(parse(text), binding)


def play(prog, memory, frames=12):
    """Per call: the memory, the SID writes and the control values."""
    mem, state, seen = bytearray(memory), controls(prog), []
    call = compile_program(prog)
    for _ in range(frames):
        writes: list = []
        call(mem.__getitem__, sid_writes_of(mem, writes), state, None, {})
        seen.append((bytes(mem), writes, tuple(state)))
    return seen


def again(prog, name="again"):
    """``prog`` emitted, parsed and elaborated again, with its source."""
    text = emit(prog, name)
    return build(text), text


def same(prog, memory):
    """The emitted ``prog`` plays as ``prog``, and is ``prog`` up to names."""
    back, text = again(prog)
    assert (
        isomorphic(
            pruned(prog).body,
            O.inline(back.body, back.procs),
            (prog.objects, back.objects),
        )
        is None
    )
    kept = [i for i, c in enumerate(prog.controls) if c in pruned(prog).controls]
    old = [(m, w, tuple(c[i] for i in kept)) for m, w, c in play(prog, memory)]
    assert old == play(back, memory)
    if native.available():
        assert run_both(prog, memory) is None and run_both(back, memory) is None
    return back, text


def memory_of(seed=(5, 0x81, 0x40, 0xFE, 0x20)):
    """Scratch memory with a table, a stream and the walker table."""
    mem = table_memory(seed)
    mem[0x2000 : 0x2000 + len(TABLE)] = bytes(TABLE)
    return mem


@pytest.fixture(name="scalars")
def fixture_scalars():
    """The reordered program with states of every width."""
    return O.reorder(
        build(SCALARS, starts=[5, 7]), O.ORDER, observed=[(0x30, 0x37)]
    ).program


def test_stated_variables_are_top_level_states_of_their_width(scalars):
    back, text = same(scalars, bytes(0x10000))
    for decl in (
        "state n: u8 = 3;",
        "state w: u16 = 0x1234;",
        "state f: bool = true;",
        "state t: u12 = 0x123;",
        "state k: u8 = 5;",
        "state k_2: u8 = 7;",
    ):
        assert decl in text
    assert [c[1:] for c in back.controls] == [c[1:] for c in scalars.controls]


def test_an_emitted_state_is_read_by_name(scalars):
    text = emit(scalars, "named")
    assert re.search(r"\bn \+= 1;|\bn := \(n \+ 1\);", text)
    assert not re.search(r"state_|let n\b", text)


def test_a_control_declaration_is_a_state_of_the_first_value():
    prog = O.reorder(build(HIDDEN, fr=0x40), O.ORDER).program
    assert len(prog.controls) == 1
    back, text = same(prog, bytes(0x10000))
    assert "state phase: u8 = 0;" in text and back.controls[0][1:] == (8, 0)


@pytest.mark.parametrize("source", [FULL, STATED_FULL], ids=["cells", "state"])
def test_objects_over_state_bind_by_name(source):
    prog = O.reorder(build(source, **BINDING), O.ORDER).program
    back, text = same(prog, memory_of())
    owned = {n for n, _w, _d in prog.controls}
    assert len(back.controls) == len(owned)
    assert (len(owned) == 10) == (source == STATED_FULL)
    if owned:
        for kind in ("counter spd", "cursor pwt", "cursor ptn", "owner flt"):
            assert kind in text.replace("_ordered", "")
        assert "cell spd_cell" not in text


def test_a_cursor_over_state_with_wide_values():
    prog = build(WALK, tab=0x2000)
    flat = O.inline(prog.body, prog.procs)
    prog = replace(prog, procs={}, body=O.declared(flat, [c[0] for c in prog.controls]))
    back, text = same(prog, memory_of())
    assert "state pw: u16 = 0;" in text
    assert [c[1] for c in back.controls] == [8, 8, 16, 8, 16]


def test_reorder_leaves_unobserved_states_unread_and_emit_drops_them():
    prog = O.reorder(build(SCALARS, starts=[5, 7]), O.ORDER).program
    assert len(prog.controls) == 6 and len(pruned(prog).controls) == 4
    back, text = same(prog, bytes(0x10000))
    assert "state f" not in text and len(back.controls) == 4


def test_a_control_nothing_reads_is_dropped_with_what_feeds_it(scalars):
    dead = [("zz", 8, 9)]
    body = scalars.body + [("set", "tmp", V("zz", 8)), ("set", "zz", V("tmp", 8))]
    prog = replace(scalars, controls=scalars.controls + dead, body=body)
    assert pruned(scalars) is scalars
    cut = pruned(prog)
    assert cut.controls == scalars.controls and cut.body == scalars.body
    assert "zz" not in emit(prog, "cut")
    kept = replace(prog, body=scalars.body + [("set", "zz", K(1, 8))])
    assert pruned(kept).controls == scalars.controls
    assert pruned(replace(scalars, body=[("set", "n", V("n", 8))] + scalars.body))


def test_procs_are_still_refused():
    prog = build(HIDDEN, fr=0x40)
    with pytest.raises(EmitError, match="procs"):
        emit(replace(prog, procs={"p": []}))
