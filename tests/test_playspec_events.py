"""Events and event-driven objects: counters, owners, walkers, cursors and records."""

import pytest

from pysidtracker.playspec import (
    SpecError,
    StreamValidator,
    compile_program,
    elaborate,
    parse,
)
from pysidtracker.playspec import outorder as O
from pysidtracker.playspec import outproof, smt
from pysidtracker.playspec.emit import emit, isomorphic
from pysidtracker.playspec.ir import K, Rd
from pysidtracker.playspec.lowerops import mk
from pysidtracker.playspec.objects import SpecRuntimeError
from pysidtracker.playspec.pyexec import controls
from pysidtracker.notes import frequency


@pytest.fixture(autouse=True)
def backend():
    """These tests do not use the 6502 engines the suite is otherwise run on."""
    return None


pytestmark = pytest.mark.strain_exempt(
    reason="asserts the values, diagnostics and equalities of the object semantics, which coverage cannot see"
)


def build(text, **binding):
    """``text`` elaborated against ``binding``."""
    return elaborate(parse(text), binding)


def frames(prog, memory, count, watch):
    """``watch`` (addresses) after each of ``count`` calls of ``prog`` on ``memory``."""
    mem = bytearray(0x10000)
    for addr, data in memory.items():
        mem[addr : addr + len(data)] = bytes(data)
    call, state, seen = compile_program(prog), controls(prog), []
    for _ in range(count):
        call(mem.__getitem__, mem.__setitem__, state)
        seen.append(tuple(mem[a] for a in watch))
    return seen


COUNTERS = """
spec c;
cell down: u8 @ 0x10;
cell up: u8 @ 0x11;
cell beat: u8 @ 0x12;
cell rows: u8 @ 0x13;
cell wide: u16 @ 0x14;
cell tempo: u8 @ 0x16;
counter spd: down { on frame -= 1; reload tempo; wrap -> tick; }
counter step: up { on tick += 1; wrap at 1 -> row; on note reset; }
counter total: rows { on row += 1; }
counter big: wide { on frame += 1; wrap at 0x0101; }
machine m { step { fire frame; if mem[0x20] != 0 { fire note; } } }
schedule { m; }
"""


def test_a_tempo_countdown_ticks_every_reload_plus_one_frames_and_cascades():
    prog = build(COUNTERS)
    watch = (0x10, 0x11, 0x13)
    seen = frames(prog, {0x10: [2], 0x16: [2]}, 9, watch)
    assert seen == [
        (1, 0, 0),
        (0, 0, 0),
        (2, 1, 0),
        (1, 1, 0),
        (0, 1, 0),
        (2, 0, 1),
        (1, 0, 1),
        (0, 0, 1),
        (2, 1, 1),
    ]


def test_a_counter_reloads_to_its_width_and_a_note_resets_it():
    prog = build(COUNTERS)
    seen = frames(prog, {0x14: [0xFF, 0x00], 0x16: [9]}, 3, (0x14, 0x15))
    assert seen == [(0, 1), (1, 1), (0, 0)]
    mem = {0x11: [1], 0x20: [1], 0x16: [9]}
    assert frames(prog, mem, 2, (0x11,)) == [(0,), (0,)]


@pytest.mark.parametrize(
    "decl, message",
    [
        ("counter a: down { on frame -= 1; wrap -> frame; }", "not below frame"),
        ("counter a: down { on row -= 1; wrap -> tick; }", "not below row"),
        ("counter a: down { on frame -= 1; wrap -> bogus; }", "undefined event"),
        ("counter a: down { on bogus -= 1; }", "undefined event 'bogus'"),
        ("counter a: down { on frame -= 0; }", "positive constant"),
        ("counter a: down { on frame -= 1; on frame reset; }", "twice"),
        ("counter a: up { on frame |= 2; }", "positive constant"),
        ("counter a: tbl { on frame -= 1; }", "scalar cell"),
        ("counter a: down { on frame -= 1; }\ncounter a: up { }", "already declared"),
        ("event row: tick;", "already declared"),
        ("event extra: nowhere;", "no event 'nowhere'"),
    ],
)
def test_counter_and_event_declarations_are_checked(decl, message):
    head = (
        "spec c;\ncell down: u8 @ 0x10;\ncell up: u8 @ 0x11;\ncell tbl: u8[4] @ 0x30;\n"
    )
    text = head + decl + "\nmachine m { step { fire frame; } }\nschedule { m; }\n"
    with pytest.raises(SpecError, match=message):
        build(text)


EXPLICIT = """
spec e;
cell c: u8 @ 0x10;
cell hits: u8 @ 0x11;
event vib: note;
event pulse: vib;
counter n: c { on vib += 3; wrap at 2 -> pulse; }
counter h: hits { on pulse += 1; }
machine m { step {
  let flag = false;
  react n on vib(2, 0) -> flag;
  if flag { hits += 10; }
  fire vib;
} }
schedule { m; }
"""


def test_user_events_join_the_hierarchy_and_react_states_a_reaction_by_hand():
    prog = build(EXPLICIT)
    assert prog.events["pulse"] == "vib"
    assert frames(prog, {0x10: [1]}, 3, (0x10, 0x11)) == [(0, 11), (0, 22), (0, 33)]


@pytest.mark.parametrize(
    "stmt, message",
    [
        ("react nobody on frame();", "no object 'nobody'"),
        ("react n on frame(1, 2);", "reacting to 'frame'"),
        ("react n on vib(1);", "takes 2 operands"),
        ("let q = 1; react n on vib(1, 2) -> q;", "needs a bool variable"),
        ("fire bogus;", "undefined event"),
    ],
)
def test_react_and_fire_statements_are_checked(stmt, message):
    text = EXPLICIT.replace("fire vib;", stmt)
    with pytest.raises(SpecError, match=message):
        build(text)


WALKER = """
spec w;
layout { tab: addr; }
cell pos: u8 @ 0x20;
cell cnt: u8 @ 0x21;
cell pw: u16 @ (0x22, 0x23);
cell band: u8 @ 0x24;
cell fine: u16 @ (0x25, 0x26);
walker t {
  table @ L.tab width 3;
  pos pos; count cnt; values pw, band, fine; start 0;
  on frame step; on note reset;
  row 0..0x3f: sweep(into=pw, count=c0, step=c1) then band = c2;
  row 0x40..0x4f: delay(count=c0 & 0x0f) then band = c1;
  row 0x50..0x5f: sweep(into=fine, count=c0 & 3, step=c1, frac=3);
  row 0x80..0xfd: set(pw.hi = c0 & 0x7f, pw.lo = c1, band = c0 >> 4, fine.lo = c2);
  row 0xfe: jump(c1);
  row 0xff: stop;
}
machine m { step { fire frame; } }
schedule { m; }
"""
TABLE = [
    0x02, 0x10, 0x07,
    0x42, 0x99, 0x00,
    0x81, 0x20, 0x03,
    0x01, 0xF0, 0x09,
    0xFE, 0x00, 0x00,
]  # fmt: skip


def test_a_walker_sweeps_delays_sets_and_jumps_on_each_frame():
    prog = build(WALKER, tab=0x2000)
    memory = {0x2000: TABLE}
    seen = frames(prog, memory, 14, (0x20, 0x21, 0x22, 0x23, 0x24))
    assert seen == [
        (0, 1, 0x10, 0, 0),
        (0, 2, 0x20, 0, 0),
        (3, 0, 0x20, 0, 7),
        (3, 1, 0x20, 0, 7),
        (3, 2, 0x20, 0, 7),
        (6, 0, 0x20, 0, 0x99),
        (9, 0, 0x20, 1, 8),
        (9, 1, 0x10, 1, 8),
        (12, 0, 0x10, 1, 9),
        (0, 1, 0x20, 1, 9),
        (0, 2, 0x30, 1, 9),
        (3, 0, 0x30, 1, 7),
        (3, 1, 0x30, 1, 7),
        (3, 2, 0x30, 1, 7),
    ]


def test_a_walker_sweeps_a_fixed_point_word_and_stops():
    prog = build(WALKER, tab=0x2000)
    rows = [0x51, 0xFE, 0, 0xFF, 0, 0]
    seen = frames(prog, {0x2000: rows}, 5, (0x20, 0x21, 0x25, 0x26))
    assert seen == [(0, 1, 6, 0xFF)] + [(3, 0, 6, 0xFF)] * 4


def test_a_note_resets_a_walker_to_its_start():
    text = WALKER.replace("fire frame;", "fire frame; if mem[0x30] != 0 { fire note; }")
    prog = build(text, tab=0x2000)
    seen = frames(
        prog, {0x2000: TABLE, 0x20: [6], 0x21: [2], 0x30: [1]}, 1, (0x20, 0x21)
    )
    assert seen == [(0, 0)]


def test_a_walker_that_jumps_forever_fails():
    prog = build(WALKER, tab=0x2000)
    with pytest.raises(SpecRuntimeError, match="jumps forever"):
        frames(prog, {0x2000: [0xFE, 3, 0, 0xFE, 0, 0]}, 1, ())


def test_a_walker_over_column_tables_advances_one_row_at_a_time():
    text = """
    spec w;
    cell pos: u8 @ 0x20; cell cnt: u8 @ 0x21; cell v: u8 @ 0x22;
    walker t { table @ 0x2000, 0x2100; pos pos; count cnt; values v;
      on frame step;
      row 0..0x7f: set(v = c1);
      row 0xff: stop; }
    machine m { step { fire frame; } }
    schedule { m; }
    """
    memory = {0x2000: [1, 2, 0xFF], 0x2100: [11, 22, 33]}
    assert frames(build(text), memory, 4, (0x20, 0x22)) == [
        (1, 11),
        (2, 22),
        (2, 22),
        (2, 22),
    ]


WALKER_HEAD = """spec w;
cell pos: u8 @ 0x20; cell cnt: u8 @ 0x21; cell pw: u16 @ 0x22; cell band: u8 @ 0x24;
walker t { table @ 0x2000 width 2; %s }
machine m { step { fire frame; } }
schedule { m; }
"""
GOOD = "pos pos; count cnt; values pw, band; "


@pytest.mark.parametrize(
    "body, message",
    [
        ("pos pos; count cnt;", "exactly one values"),
        ("count cnt; values pw, band;", "exactly one pos"),
        (GOOD + "row 0: sweep(into=pw, count=c0);", "sweep"),
        (GOOD + "row 1: set(band.hi = c0);", "no high byte"),
        (
            GOOD + "row 0: sweep(into=band, count=c0, step=c1, frac=3);",
            "frac needs a word",
        ),
        (GOOD + "row 0: set(pw = c9);", "undefined name 'c9'"),
        (GOOD + "row 0: set(pw = c0 + c1);", "an operand is"),
        (GOOD + "row 0: set(nope = c0);", "a value of"),
        (GOOD + "row 0: set(pw = (c0 >> 1) & 3);", "an operand is"),
        (GOOD + "row 0: set(pw.mid = c0);", "a value of"),
        (GOOD + "table @ 1;", "exactly one table"),
    ],
)
def test_walker_declarations_are_checked(body, message):
    with pytest.raises(SpecError, match=message):
        build(WALKER_HEAD % body)


CURSOR = """
spec c;
cell pos: u8 @ 0x40; cell run: u8 @ 0x41;
cell note: u8 @ 0x42; cell inst: u8 @ 0x43; cell fx: u8 @ 0x44;
cursor ptn @ 0x2000 pos pos {
  on row next; on instrument reset;
  run 0xF0..0xF7 base 0xF0 -> run;
  field n -> note = n & 0x7f;
  field i when neg(n) -> inst = i;
  field f when i & 0x80 != 0 -> fx = f >> 4;
}
machine m { step { fire row; if mem[0x50] != 0 { fire instrument; } } }
schedule { m; }
"""
PATTERN = [0x85, 0x01, 0x05, 0x83, 0x90, 0x70, 0xF2, 0x07]


def test_a_cursor_reads_variable_rows_and_run_length_gaps():
    prog = build(CURSOR)
    seen = frames(prog, {0x2000: PATTERN}, 7, (0x40, 0x41, 0x42, 0x43, 0x44))
    assert seen == [
        (2, 0, 5, 1, 0),
        (3, 0, 5, 1, 0),
        (6, 0, 3, 0x90, 7),
        (7, 2, 0, 0x90, 7),
        (7, 1, 0, 0x90, 7),
        (7, 0, 0, 0x90, 7),
        (8, 0, 7, 0x90, 7),
    ]


def test_a_cursor_reset_clears_its_position_and_run():
    prog = build(CURSOR)
    memory = {0x2000: PATTERN, 0x40: [6], 0x41: [3], 0x50: [1]}
    assert frames(prog, memory, 1, (0x40, 0x41, 0x42)) == [(0, 0, 0)]


@pytest.mark.parametrize(
    "old, new, message",
    [
        ("field n ->", "field n when neg(n) ->", "first field"),
        ("when i & 0x80 != 0", "when i + 1", "condition tests"),
        ("base 0xF0", "base 0xF8", "base is above"),
        ("on row next;", "on row next; on row reset;", "twice"),
    ],
)
def test_cursor_declarations_are_checked(old, new, message):
    with pytest.raises(SpecError, match=message):
        build(CURSOR.replace(old, new))


def test_a_cursor_without_a_run_marker_reads_every_byte_as_a_row():
    text = CURSOR.replace("run 0xF0..0xF7 base 0xF0 -> run;", "")
    seen = frames(build(text), {0x2000: [0xF2, 0x07]}, 2, (0x40, 0x42))
    assert seen == [(2, 0x72), (3, 0)]


OWNER = """
spec o;
cell filt: u8 @ 0x60;
owner f: filt { id x * 7; none 0xFF; on note claim when mem[0x70 + x] != 0; on legato release; }
machine voice[x in 0..3] {
  step {
    if mem[0x71] != 0 && x == 1 { fire legato; }
    fire note;
    if f.mine { mem[0x80 + x] := 1; }
    if f.free { mem[0x90 + x] := 1; }
    if owns(filt, x * 7) { mem[0xA0 + x] := 1; }
  }
}
schedule { voice[0]; voice[1]; voice[2]; }
"""


def test_an_owner_is_claimed_released_and_tested_by_the_owning_voice():
    prog = build(OWNER)
    watch = [0x60, 0x80, 0x81, 0x82, 0x90, 0x91, 0x92, 0xA0, 0xA1, 0xA2]
    first = frames(prog, {0x60: [0xFF], 0x71: [1]}, 1, watch)[0]
    assert first == (7, 0, 1, 0, 1, 0, 0, 0, 1, 0)
    second = frames(build(OWNER), {0x60: [0xFF], 0x70: [1], 0x72: [1]}, 1, watch)[0]
    assert second == (14, 1, 0, 1, 0, 0, 0, 1, 0, 1)
    free = frames(build(OWNER), {0x60: [0xFF]}, 1, watch)[0]
    assert free == (0xFF, 0, 0, 0, 1, 1, 1, 0, 0, 0)


RECORD = """
spec r;
layout { base: addr; }
machine voice[x in 0..3] {
  record v @ L.base stride 6 { dur: u8; freq: u16; flags: u8 @ 5; }
  step { dur += 1; freq += 0xFF; flags := x; }
}
schedule { voice[0]; voice[1]; voice[2]; }
"""


def test_a_record_places_each_voices_fields_at_a_stride():
    prog = build(RECORD, base=0x3000)
    cells = {(c.owner, c.name): c for c in prog.cells}
    assert cells[("voice[2]", "freq")].place == (0x300D,)
    assert cells[("voice[1]", "flags")].record == "v"
    seen = frames(prog, {}, 2, [0x3000 + 6 * v + o for v in range(3) for o in range(6)])
    assert seen[1] == (2, 0xFE, 0x01, 0, 0, 0) + (2, 0xFE, 0x01, 0, 0, 1) + (
        2,
        0xFE,
        0x01,
        0,
        0,
        2,
    )


@pytest.mark.parametrize(
    "fields, message",
    [
        ("a: u8; b: u16 @ 5;", "ends past the stride"),
        ("a: u16; b: u8 @ 1;", "overlaps a"),
        ("a: bool;", "u8 or u16"),
        ("dur: u8;", "already a cell"),
    ],
)
def test_record_declarations_are_checked(fields, message):
    text = f"""spec r;
    cell dur: u8 @ 0x10;
    record r @ 0x3000 stride 6 {{ {fields} }}
    machine m {{ step {{ }} }}
    schedule {{ m; }}
    """
    with pytest.raises(SpecError, match=message):
        build(text)


def test_a_u16_cell_carries_into_its_high_byte_without_help():
    text = """spec u;
    cell w: u16 @ (0x10, 0x20);
    cell s: u8 @ 0x30;
    machine m { step { w += sx(s); } }
    schedule { m; }
    """
    prog = build(text)
    assert frames(
        prog, {0x10: [0xF0], 0x20: [0x01], 0x30: [0x20]}, 1, (0x10, 0x20)
    ) == [(0x10, 0x02)]
    assert frames(
        prog, {0x10: [0x10], 0x20: [0x01], 0x30: [0xF0]}, 1, (0x10, 0x20)
    ) == [(0x00, 0x01)]


def test_objects_check_without_a_binding_and_derive_with_a_spec():
    text = WALKER.replace("fire frame;", "fire frame; fire note;")
    prog = elaborate(parse(text), None)
    assert prog.objects["t"].locs["pos"] == (0x20,)
    base = parse(COUNTERS)
    child = parse(
        "spec d extends c;\ncounter extra: up { on frame += 1; }\n", {"c": base}
    )
    assert [o[2] for o in child.objects] == ["spd", "step", "total", "big", "extra"]
    with pytest.raises(SpecError, match="counter spd is already in base"):
        parse("spec d extends c;\ncounter spd: up { on frame += 1; }\n", {"c": base})


def test_constant_note_reads_fold_to_the_table_entry_and_others_stay():
    clock = K(985248, 0)
    word = frequency(5 - 24, 985248)
    assert mk("note", 8, clock, K(0, 0), K(-24, 0), K(5, 8)).v == word & 0xFF
    assert mk("note", 16, clock, K(2, 0), K(-24, 0), K(5, 16)).v == word
    assert mk("note", 8, clock, K(1, 0), K(0, 0), K(60000, 16)).v == 0
    assert mk("note", 8, clock, K(0, 0), K(0, 0), K(300, 8)).op == "note"
    assert mk("note", 8, clock, K(0, 0), K(0, 0), Rd(K(0x10, 16))).op == "note"


NOTES = """spec n;
layout { sid, lo: addr; }
cell chip: u8[25] @ L.sid out;
cell flo: u8[96] @ L.lo ro notes(lo, -24);
machine m { step { chip[0] := flo[30]; } }
schedule { m; }
"""


def test_reordering_a_program_that_reads_a_note_table_at_a_constant_index():
    prog = elaborate(parse(NOTES), {"sid": 0xD400, "lo": 0x2000}, standard=985248)
    result = O.reorder(prog, O.ORDER)
    stores = [s for s in O.statements(result.program.body) if s[0] == "wr"]
    assert [s[2].v for s in stores] == [frequency(6, 985248) & 0xFF]


FULL = """spec f;
layout { sid, tab: addr; }
cell chip: u8[25] @ L.sid out;
cell tick: u8 @ 0x10;
cell spdc: u8 @ 0x11;
cell tempo: u8 @ 0x12;
cell wpos: u8 @ 0x13;
cell wcnt: u8 @ 0x14;
cell pw: u16 @ (0x15, 0x16);
cell band: u8 @ 0x17;
cell cpos: u8 @ 0x18;
cell crun: u8 @ 0x19;
cell cnote: u8 @ 0x1A;
cell owned: u8 @ 0x1B;
counter spd: spdc { on frame += 1; wrap at tempo -> tick; }
walker pwt {
  table @ L.tab width 2; pos wpos; count wcnt; values pw, band;
  on frame step; on note reset;
  row 0..0x7f: sweep(into=pw, count=c0, step=c1);
  row 0x80..0xfd: set(pw.hi = c0 & 0x7f, pw.lo = c1);
  row 0xfe: jump(c1);
  row 0xff: stop;
}
cursor ptn @ L.tab + 0x40 pos cpos {
  on tick next;
  run 0xF0..0xF7 base 0xF0 -> crun;
  field n -> cnote = n & 0x7f;
}
owner flt: owned { on note claim; none 0xFF; }
machine voice[x in 0..2] {
  step {
    chip[7 * x + 2] := lo(pw);
    chip[7 * x + 3] := hi(pw);
    if flt.mine { chip[21] := cnote; }
    chip[7 * x + 4] := 0x41;
    chip[7 * x + 4] := 0x40;
  }
}
machine clock { step { fire frame; if tick == 3 { fire note; } } }
schedule { clock; for v in 0..2 { voice[v]; } }
"""
BINDING = {"sid": 0xD400, "tab": 0x2000}


def test_reorder_keeps_the_objects_and_proves_the_simplified_code_around_them():
    prog = elaborate(parse(FULL), BINDING)
    result = O.reorder(prog, O.ORDER)
    before = [s for s in O.statements(O.inline(prog.body, prog.procs)) if s[0] == "obj"]
    after = [s for s in O.statements(result.program.body) if s[0] == "obj"]
    assert [(s[1], s[2]) for s in after] == [(s[1], s[2]) for s in before]
    assert not any(s[0] == "fire" for s in O.statements(result.program.body))
    assert set(result.program.objects) == {"spd", "pwt", "ptn", "flt"}
    proof = outproof.prove(prog, O.ORDER, timeout_ms=20000)
    assert proof.holds, proof.open


def test_an_emitted_spec_elaborates_to_the_same_program():
    prog = elaborate(parse(FULL), BINDING)
    result = O.reorder(prog, O.ORDER)
    text = emit(result.program, "full_ordered")
    back = elaborate(parse(text), {})
    assert (
        isomorphic(
            result.program.body,
            O.inline(back.body, back.procs),
            (result.program.objects, back.objects),
        )
        is None
    )
    assert {o.kind for o in back.objects.values()} == {
        "counter",
        "walker",
        "cursor",
        "owner",
    }
    tweaked = text.replace("on frame step;", "on frame step;\n  on tick step;")
    other = elaborate(parse(tweaked), {})
    assert isomorphic(
        result.program.body,
        O.inline(other.body, other.procs),
        (result.program.objects, other.objects),
    )


def test_an_object_that_writes_the_sid_is_not_rescheduled():
    sid = FULL.replace(
        "cell pw: u16 @ (0x15, 0x16);", "cell pw: u16 @ (0xD400, 0xD401);"
    )
    with pytest.raises(O.OrderError, match="writes the SID"):
        O.reorder(elaborate(parse(sid), BINDING), O.ORDER)


def test_a_table_at_a_computed_address_leaves_every_cell_in_memory_and_is_proved():
    wild = FULL.replace("table @ L.tab width 2;", "table @ mem16[0x1C] width 2;")
    prog = elaborate(parse(wild), BINDING)
    body = O.inline(prog.body, prog.procs)
    walker = next(s for s in O.statements(body) if s[0] == "obj" and s[1] == "pwt")
    assert [d[0] for d in walker[7]] == [0, 1]
    result = O.reorder(prog, O.ORDER)
    assert result.promoted == []
    proof = outproof.prove(prog, O.ORDER, timeout_ms=20000)
    assert proof.holds, proof.open


def test_the_solver_treats_a_reaction_as_one_deterministic_function_of_its_inputs():
    import z3  # pylint: disable=import-outside-toplevel

    prog = elaborate(parse(FULL), BINDING)
    body = [s for s in O.statements(O.inline(prog.body, prog.procs)) if s[0] == "obj"]
    walker = next(s for s in body if s[1] == "pwt")
    twice = smt.step(prog, smt.symbolic_memory("m"))
    again = smt.Executor().run(
        [walker, walker], smt.State(smt.symbolic_memory("m"), {})
    )
    one = smt.Executor().run([walker], smt.State(smt.symbolic_memory("m"), {}))
    same = smt.Executor().run([walker], smt.State(smt.symbolic_memory("m"), {}))
    solver = z3.Solver()
    solver.add(one.mem != same.mem)
    assert solver.check() == z3.unsat
    solver = z3.Solver()
    solver.add(one.mem != again.mem)
    assert solver.check() == z3.sat
    assert twice.writes


def test_words_the_object_syntax_uses_stay_ordinary_names():
    words = (
        "pos count start width values row field run base then table id none when wrap"
    )
    cells = "\n".join(
        f"cell {w}: u8 @ {0x40 + i};" for i, w in enumerate(words.split())
    )
    text = f"""spec names;
{cells}
cell next_pos: u8 @ 0x60;
cell reset_all: u8 @ 0x61;
cell stepper: u8 @ 0x62;
cell claimed: u8 @ 0x63;
cell reload: u8 @ 0x64;
cell at: u8 @ 0x65;
machine m {{
  step {{
    let record = 1;
    next_pos += record;
    reset_all := next_pos + pos;
    stepper := count + start + width + values + row + field + run + base + then + table;
    claimed := id + none + when + wrap + reload + at;
  }}
}}
schedule {{ m; }}
"""
    memory = {0x40 + i: [i + 1] for i in range(len(words.split()))}
    seen = frames(build(text), memory, 1, (0x60, 0x61, 0x62, 0x63))
    assert seen == [(1, 2, 65, 54)]


HANDLERS = """
spec h;
cell first: u8 @ 0x10;
cell second: u8 @ 0x11;
cell seen: u8 @ 0x12;
cell wraps: u8 @ 0x13;
counter a: first { on frame += 1; wrap at 1 -> tick; }
handler glue on frame, tick { seen := first * 16 + second; wraps += 1; }
counter b: second { on frame += 1; }
machine voice[x in 0..2] {
  cell mine: u8 @ 0x20 + x;
  handler local on tick { mine += 1; }
  step { fire frame; }
}
schedule { voice[0]; voice[1]; }
"""


def test_handlers_run_with_the_objects_in_declaration_order():
    prog = build(HANDLERS)
    seen = frames(prog, {0x10: [0]}, 3, (0x10, 0x11, 0x12, 0x13, 0x20, 0x21))
    assert seen == [(0, 2, 1, 3, 0, 1), (0, 4, 3, 6, 0, 2), (0, 6, 5, 9, 0, 3)]


def test_handlers_name_known_events_and_are_unique():
    with pytest.raises(SpecError, match="undefined event 'nope'"):
        build(HANDLERS.replace("on tick { mine", "on nope { mine"))
    twice = HANDLERS.replace("handler local", "handler glue")
    with pytest.raises(SpecError, match="glue is already declared"):
        build(twice)


def roundtrip(text, **binding):
    """The reordered program of ``text`` and the program its emitted spec elaborates to."""
    result = O.reorder(elaborate(parse(text), binding), O.ORDER)
    back = elaborate(parse(emit(result.program, "again")), {})
    return result, back


@pytest.mark.parametrize(
    "text, binding",
    [
        (WALKER, {"tab": 0x2000}),
        (CURSOR, {}),
        (OWNER, {}),
        (COUNTERS, {}),
    ],
)
def test_every_row_field_and_reaction_form_is_emitted_and_elaborated_back(
    text, binding
):
    result, back = roundtrip(text, **binding)
    assert result.program.objects
    assert (
        isomorphic(
            result.program.body,
            O.inline(back.body, back.procs),
            (result.program.objects, back.objects),
        )
        is None
    )


def test_a_cursors_stream_is_validated_against_its_grammar_each_time_it_reads():
    text = """spec g;
    cell pos: u8 @ 0x40; cell note: u8 @ 0x41;
    cursor ptn @ 0x2000 pos pos grammar "tests.test_bytegrammar:TOY:seq" {
      on row next;
      field n -> note = n;
    }
    machine m { step { fire row; } }
    schedule { m; }
    """
    prog = build(text)
    assert prog.grammars == [("tests.test_bytegrammar:TOY:seq", 3)]
    mem = bytearray(0x10000)
    mem[0x2000:0x2003] = bytes([1, 2, 0xFF])
    validator = StreamValidator()
    call = compile_program(prog)
    call(mem.__getitem__, mem.__setitem__, [], validator.hook(prog, mem.__getitem__))
    assert list(validator.failures()) == []
    assert validator.results == {("tests.test_bytegrammar:TOY:seq", 0x2000): None}
    mem[0x2000:0x2003] = bytes([1, 2, 3])
    validator = StreamValidator()
    call(mem.__getitem__, mem.__setitem__, [], validator.hook(prog, mem.__getitem__))
    assert list(validator.failures()) == [("tests.test_bytegrammar:TOY:seq", 0x2000)]


def test_the_census_counts_the_object_reactions_a_reordered_program_keeps():
    result = O.reorder(elaborate(parse(FULL), BINDING), O.ORDER)
    census = O.census(result)
    assert census["objects"] == {"counter": 1, "cursor": 1, "owner": 1, "walker": 2}
    assert census["statements"]["obj"] == 5
