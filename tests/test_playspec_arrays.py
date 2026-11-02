"""Array states: song data indexed at run time, on every backend."""

import random
import re
from pathlib import Path

import pytest

from pysidtracker.asm6502 import assemble
from pysidtracker.playspec import (
    SpecError,
    SpecPlayer,
    SpecRuntimeError,
    elaborate,
    load,
    native,
    parse,
)
from pysidtracker.playspec import outorder as O
from pysidtracker.playspec import outproof, smt
from pysidtracker.playspec.emit import emit, isomorphic
from pysidtracker.playspec.ir import Op
from pysidtracker.playspec.pyexec import compile_program, controls
from pysidtracker.testing import whole_song

from .helpers import build_psid
from .test_playspec_events import frames
from .test_playspec_native import run_both

pytestmark = pytest.mark.strain_exempt(
    reason="asserts values, diagnostics and agreement between executors, which coverage cannot see"
)
fast = pytest.mark.skipif(not native.available(), reason="the native core is not built")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "arrays"
ROWS = [7, 11, 13, 17, 19]

SONG = """
spec song;
layout { rows: list; }
cell out: u8 @ 0x2000;
cell idx: u8 @ 0x2001;
cell pair: u16 @ 0x2002;
cell echo: u8 @ 0x2004;
state data: u8[] ro = L.rows;
state words: u16[2] ro = [0x1234, 0xabc];
state hist: u8[4] = 0;
state k: u8 = 0;
machine m {
  step {
    out := data[idx];
    pair := words[idx & 1];
    hist[k & 3] := out + k;
    echo := hist[idx & 3];
    k += 1;
  }
}
schedule { m; }
"""


@pytest.fixture(autouse=True)
def backend():
    """These tests do not use the 6502 engines the suite is otherwise run on."""
    return None


def build(text, **binding):
    """``text`` elaborated against ``binding``."""
    return elaborate(parse(text), binding)


def song():
    """The :data:`SONG` spec bound to :data:`ROWS`."""
    return build(SONG, rows=ROWS)


def play(prog, indexes):
    """``(out, pair, echo)`` after each call, the index cell set before it."""
    mem, call, state, seen = bytearray(0x10000), compile_program(prog), None, []
    state = controls(prog)
    for index in indexes:
        mem[0x2001] = index
        call(mem.__getitem__, mem.__setitem__, state)
        seen.append((mem[0x2000], mem[0x2002] | mem[0x2003] << 8, mem[0x2004]))
    return seen


def run_cell(prog, calls=1):
    """The byte at ``$10`` after ``calls`` calls from zeroed memory."""
    mem, call, state = bytearray(0x10000), compile_program(prog), controls(prog)
    for _ in range(calls):
        call(mem.__getitem__, mem.__setitem__, state)
    return mem[0x10]


def model(indexes):
    """What :data:`SONG` does, by hand."""
    hist, seen = [0] * 4, []
    for k, index in enumerate(indexes):
        out = ROWS[index]
        hist[k & 3] = (out + k) & 0xFF
        seen.append((out, (0x1234, 0xABC)[index & 1], hist[index & 3]))
    return seen


def test_read_only_and_mutable_arrays_are_indexed_at_run_time():
    indexes = [random.Random(3).randrange(5) for _ in range(40)]
    indexes += [4, 0, 2, 1, 3, 3, 3, 0]
    assert play(song(), indexes) == model(indexes)


def test_an_index_past_the_items_is_a_runtime_error():
    with pytest.raises(SpecRuntimeError, match="past the 5 items"):
        play(song(), [4, 5])


def test_a_mutable_store_past_the_items_is_a_runtime_error():
    text = SONG.replace("hist[k & 3]", "hist[k]")
    with pytest.raises(SpecRuntimeError, match=r"hist\[...\] index past its 4 items"):
        play(build(text, rows=ROWS), [0] * 5)


def ops_of(prog):
    """The ``lut`` and ``aget`` operators of ``prog``'s statements."""
    return [
        n
        for s in O.statements(O.inline(prog.body, prog.procs))
        for e in O.stmt_exprs(s)
        for n in O.walk(e)
        if isinstance(n, Op) and n.op in ("lut", "aget")
    ]


def test_a_static_index_is_the_item_itself():
    tables = lambda prog: {o.a[0].v for o in ops_of(prog) if o.op == "lut"}
    static = build(SONG.replace("data[idx]", "data[2] + hist[1]"), rows=ROWS)
    assert tables(song()) == {tuple(ROWS), (0x1234, 0xABC)}
    assert tables(static) == {(0x1234, 0xABC)}
    assert [o.op for o in ops_of(static)] == ["lut", "aget"]
    assert play(static, [0])[0][0] == ROWS[2] + 0


def test_the_program_names_the_elements_of_a_mutable_array():
    prog = song()
    assert list(prog.arrays) == ["hist"] and len(prog.arrays["hist"]) == 4
    assert [prog.keys[n] for n in prog.arrays["hist"]] == [
        f"hist[{i}]" for i in range(4)
    ]


BAD = {
    "a static index past the items": ("out := data[5];", "data[5] is past its 5 items"),
    "a static store past the items": ("hist[4] := 1;", "hist[4] is past its 4 items"),
    "a store to a ro array": ("data[0] := 1;", "write to read-only array data"),
    "a bare array": ("out := data;", "data is an array; index it"),
}


@pytest.mark.parametrize("what", BAD)
def test_misuse_is_an_elaboration_error(what):
    stmt, message = BAD[what]
    with pytest.raises(
        SpecError, match=message.replace("[", r"\[").replace("]", r"\]")
    ):
        build(SONG.replace("k += 1;", stmt), rows=ROWS)


HEAD = "spec bad;\nlayout { rows: list; }\ncell o: u8 @ 0x10;\n"
DECLARATIONS = {
    "length mismatch": (
        "state a: u8[4] ro = L.rows;\nmachine m { step { o := a[1]; } }",
        "5 initial items for 4 elements",
    ),
    "literal length mismatch": (
        "state a: u8[2] = [1, 2, 3];\nmachine m { step { o := a[1]; } }",
        "3 initial items for 2 elements",
    ),
    "unread": (
        "state a: u8[] ro = L.rows;\nmachine m { step { o := 1; } }",
        "state a is never read",
    ),
    "written only": (
        "state a: u8[2] = 0;\nmachine m { step { a[o & 1] := 1; } }",
        "state a is never read",
    ),
    "item too wide": (
        "state a: u8[] ro = [1, 300];\nmachine m { step { o := a[1]; } }",
        "an item does not fit u8",
    ),
    "ro from a scalar": (
        "state a: u8[2] ro = 1;\nmachine m { step { o := a[1]; } }",
        "a ro array is initialised from a list",
    ),
    "no length": (
        "state a: u8[] = 1;\nmachine m { step { o := a[1]; } }",
        r"the length is needed",
    ),
    "list on a scalar": (
        "state a: u8 = [1];\nmachine m { step { o := a; } }",
        "a list or `ro` needs an array",
    ),
    "ro on a scalar": (
        "state a: u8 ro = 1;\nmachine m { step { o := a; } }",
        "a list or `ro` needs an array",
    ),
    "other qualifier": (
        "state a: u8[2] out = 1;\nmachine m { step { o := a[1]; } }",
        "optionally `ro`",
    ),
}


@pytest.mark.parametrize("what", DECLARATIONS)
def test_bad_declarations_are_elaboration_errors(what):
    text, message = DECLARATIONS[what]
    with pytest.raises(SpecError, match=message):
        build(HEAD + text + "\nschedule { m; }", rows=ROWS)


def test_a_broadcast_item_fills_a_mutable_array_and_len_is_static():
    prog = build(
        HEAD
        + "state a: u8[3] = 9;\nmachine m { step { o := a[o & 3 % 3] + len(a); } }\nschedule { m; }"
    )
    assert [d for _n, _w, d in prog.controls] == [9, 9, 9]
    assert run_cell(prog) == 12


def test_checking_without_a_binding_sizes_what_it_can():
    prog = elaborate(parse(SONG))
    assert len(prog.arrays["hist"]) == 4
    assert "rows" not in prog.arrays


@fast
@pytest.mark.parametrize("seed", range(6))
def test_native_agrees_with_python_on_lookups_stores_and_errors(seed):
    rng = random.Random(seed)
    memory = bytearray(0x10000)
    memory[0x2001] = rng.randrange(5)
    assert run_both(song(), memory, 9) is None
    memory[0x2001] = 5 + rng.randrange(200)
    error = run_both(song(), memory, 3)
    assert isinstance(error, SpecRuntimeError)


@fast
def test_native_stores_past_a_mutable_array_fail_as_python_does():
    prog = build(SONG.replace("hist[k & 3]", "hist[k]"), rows=ROWS)
    assert isinstance(run_both(prog, bytearray(0x10000), 6), SpecRuntimeError)


@fast
def test_the_array_program_runs_natively_without_falling_back():
    prog = song()
    assert native.encode(prog)
    ops = native.OPS
    code = native.encode(prog)[0]
    used = {
        int.from_bytes(code[i : i + 8], "little", signed=True)
        for i in range(0, len(code), 16)
    }
    assert ops["VarX"] in used and ops["Note"] in used


TUNE = "init:\n  rts\npa:\n  rts\nwrap:\n  jsr pa\n  rts\n"


def tune_player(text, use_native, **binding):
    """A player of a tune whose play routine calls the one modelled entry."""
    code, labels = assemble(TUNE, 0x1000)
    image = build_psid(code, 0x1000, init=labels["init"], play=labels["wrap"])[0x7C:]
    return SpecPlayer(
        parse(text),
        image,
        0x1000,
        labels["init"],
        labels["wrap"],
        bind=lambda view, entry: binding,
        entries=lambda view: {labels["pa"]},
        native=use_native,
    )


@pytest.mark.parametrize("use_native", [False, pytest.param(True, marks=fast)])
def test_an_array_state_is_read_and_set_by_name(use_native):
    play_ = tune_player(SONG, use_native, rows=ROWS)
    play_.render_grid(whole_song(6))
    if use_native:
        assert (
            play_._played >= 5 and play_._bails == 0
        )  # pylint: disable=protected-access
    assert play_.state("k") == 6
    assert play_.state("hist") == [(ROWS[0] + i) & 0xFF for i in (4, 5, 2, 3)]
    assert play_.state("hist[1]") == ROWS[0] + 5
    play_.set_state("hist", [1, 2, 3, 4])
    assert play_.state("hist") == [1, 2, 3, 4]
    play_.set_state("hist[2]", 9)
    assert play_.state("hist") == [1, 2, 9, 4]
    with pytest.raises(ValueError, match="4 elements, not 2"):
        play_.set_state("hist", [1, 2])
    play_.render_grid(whole_song(1))
    assert play_.state("hist")[2] == (ROWS[0] + 6) & 0xFF
    assert play_.state("hist")[0] == 1


def test_a_machine_instance_owns_its_array():
    text = HEAD + """
machine v[x in 0..2] {
  state seen: u8[2] = 0;
  step { seen[o & 1] += x + 1; o := seen[0] + seen[1]; }
}
schedule { v[0]; v[1]; }
"""
    prog = build(text)
    assert sorted(prog.arrays) == ["v[0].seen", "v[1].seen"]
    assert run_cell(prog, 3) == 6


ROW_TABLE = [2, 0x34, 0x12, 1, 0x56, 0x03, 9, 0xAB, 0xCD]
TABLES = """
spec tables;
layout { rows: list; }
cell o: u8 @ 0x20;
cell w: u16 @ 0x22;
cell at: u8 @ 0x24;
cell p: u8 @ 0x25;
cell q: u8 @ 0x26;
state rows: u8[] ro = L.rows;
table t in rows stride 3 { kind: u8; val: u16 @ 1; }
view v in rows @ p { first @ 0; third @ 2; }
machine m {
  step {
    o := t[at].kind;
    w := t[at].val;
    q := v.third + v.first;
  }
}
schedule { m; }
"""


@pytest.mark.parametrize("at", [0, 1, 2])
def test_tables_and_views_read_an_array_in_place_of_memory(at):
    prog = build(TABLES, rows=ROW_TABLE)
    mem = bytearray(0x10000)
    mem[0x24] = at
    mem[0x25] = at
    compile_program(prog)(mem.__getitem__, mem.__setitem__, [])
    row = ROW_TABLE[3 * at : 3 * at + 3]
    assert mem[0x20] == row[0]
    assert mem[0x22] | mem[0x23] << 8 == row[1] | row[2] << 8
    assert mem[0x26] == (ROW_TABLE[at + 2] + ROW_TABLE[at]) & 0xFF


@fast
def test_native_agrees_on_tables_and_views():
    mem = bytearray(0x10000)
    mem[0x24] = 2
    mem[0x25] = 3
    assert run_both(build(TABLES, rows=ROW_TABLE), mem, 3) is None
    mem[0x24] = 3
    assert isinstance(run_both(build(TABLES, rows=ROW_TABLE), mem, 3), SpecRuntimeError)


def test_a_table_over_an_array_is_read_only_and_needs_a_byte_array():
    with pytest.raises(SpecError, match="write to table t"):
        build(TABLES.replace("o := t[at].kind;", "t[at].kind := 1;"), rows=ROW_TABLE)
    with pytest.raises(SpecError, match="must be a `u8\\[\\] ro` state array"):
        build(TABLES.replace("state rows: u8[] ro", "state rows: u8[]"), rows=ROW_TABLE)


PULSE = [2, 0x10, 0, 0x81, 0x20, 0, 0xFF, 0, 0, 3, 8, 0, 0x90, 5, 0, 0xFF, 0, 0]


def rows_binding():
    """The instrument-rows example bound to two instruments laid end to end."""
    return {"w": 0x40, "pulse_rows": PULSE, "pulse_start": [0, 9]}


def memory_form():
    """The example with its rows in memory, as a player lays them out today."""
    text = (EXAMPLES / "instrument-rows.play").read_text()
    text = text.replace(
        "pulse_rows, pulse_start: list", "pulse_rows: addr; pulse_start: addr"
    )
    text = text.replace("state rows: u8[] ro = L.pulse_rows;\n", "")
    text = text.replace(
        "state first: u8[] ro = L.pulse_start;\n",
        "cell first: u8[2] @ L.pulse_start;\n",
    )
    text = text.replace("table rows width 3;", "table @ L.pulse_rows width 3;")
    return elaborate(
        parse(text), {"w": 0x40, "pulse_rows": 0x2000, "pulse_start": 0x2100}
    )


@pytest.mark.parametrize("instrument", [0, 1])
def test_a_cursor_over_instrument_rows_is_the_cursor_over_memory(instrument):
    watch = (0x40, 0x41, 0x42, 0x43, 0xD402, 0xD403)
    given = {0x44: [instrument], 0x45: [1]}
    arrays = frames(
        elaborate(load(EXAMPLES / "instrument-rows.play"), rows_binding()),
        given,
        8,
        watch,
    )
    memory = frames(memory_form(), {**given, 0x2000: PULSE, 0x2100: [0, 9]}, 8, watch)
    assert arrays == memory
    assert len({row[0] for row in arrays}) > 2


def test_each_instrument_starts_at_its_own_row():
    prog = elaborate(load(EXAMPLES / "instrument-rows.play"), rows_binding())
    seen = {}
    for instrument in (0, 1):
        mem = bytearray(0x10000)
        mem[0x44] = instrument
        call, state = compile_program(prog), controls(prog)
        for n in range(3):
            mem[0x45] = int(n == 0)
            call(mem.__getitem__, mem.__setitem__, state)
        seen[instrument] = (mem[0x40], mem[0x42])
    assert seen == {0: (0x30, 0), 1: (0x20, 9)}


@fast
@pytest.mark.parametrize("instrument", [0, 1])
def test_native_cursor_over_rows_agrees_and_reports_a_read_past_them(instrument):
    prog = elaborate(load(EXAMPLES / "instrument-rows.play"), rows_binding())
    mem = bytearray(0x10000)
    mem[0x44], mem[0x45] = instrument, 1
    assert run_both(prog, mem, 10) is None
    short = elaborate(
        load(EXAMPLES / "instrument-rows.play"),
        {**rows_binding(), "pulse_rows": PULSE[:10]},
    )
    mem[0x44] = 1
    assert isinstance(run_both(short, mem, 10), SpecRuntimeError)


def test_a_cursor_must_read_byte_arrays_it_declares():
    text = (
        (EXAMPLES / "instrument-rows.play")
        .read_text()
        .replace("table rows", "table nope")
    )
    with pytest.raises(SpecError, match="nope is not a state array"):
        elaborate(parse(text), rows_binding())
    text = (EXAMPLES / "instrument-rows.play").read_text()
    with pytest.raises(SpecError, match="exactly one table clause"):
        elaborate(parse(text.replace("table rows width 3;", "")), rows_binding())


REORDER = """
spec reorder;
layout { rows: list; }
cell chip: u8[25] @ 0xD400 out;
cell sel: u8 @ 0x2000;
state notes: u8[] ro = L.rows;
state last: u8[3] = 0;
machine voice[x in 0..3] {
  step {
    chip[7 * x + 1] := last[x];
    last[x] := notes[sel + x];
    chip[7 * x] := notes[x];
  }
}
schedule { for v in 0..3 { voice[v]; } }
"""


def test_reordering_and_proving_a_spec_with_arrays():
    prog = build(REORDER, rows=[3, 5, 7, 9, 11])
    result = O.reorder(prog, O.ORDER)
    source = emit(result.program, "ordered")
    assert "u8[] ro = [3, 5, 7, 9, 11]" in source and "u8[3] = [0, 0, 0]" in source
    back = elaborate(parse(source), {})
    assert isomorphic(result.program.body, O.inline(back.body, back.procs)) is None
    proof = outproof.prove(prog, O.ORDER, timeout_ms=30000)
    assert proof.holds, proof.open


def test_the_bounded_check_finds_an_unconstrained_index_and_clears_a_masked_one():
    prog = build(REORDER, rows=[3, 5, 7, 9, 11])
    found = smt.bmc(prog, 1, timeout_ms=30000)
    assert not found.holds and "past the 5 items" in found.what
    masked = build(
        REORDER.replace("notes[sel + x]", "notes[(sel & 1) + x]"), rows=[3, 5, 7, 9, 11]
    )
    assert smt.bmc(masked, 1, timeout_ms=30000).holds


def test_emitted_cursor_over_rows_round_trips():
    prog = elaborate(load(EXAMPLES / "instrument-rows.play"), rows_binding())
    result = O.reorder(prog, O.ORDER)
    source = emit(result.program, "ordered")
    assert re.search(r"table (\w+), \1, \1 width 3;", source)
    back = elaborate(parse(source), {})
    objects = (result.program.objects, back.objects)
    body = O.inline(back.body, back.procs)
    assert isomorphic(result.program.body, body, objects) is None
    assert sorted(o.tabs for o in back.objects.values()) == sorted(
        o.tabs for o in result.program.objects.values()
    )
    watch = (0x40, 0x41, 0x42, 0x43, 0xD402)
    given = {0x44: [1], 0x45: [1]}
    assert frames(back, given, 6, watch) == frames(result.program, given, 6, watch)
