"""Streams and ``via`` tables over array states: the same reads as over memory, on every backend."""

import random
from pathlib import Path

import pytest

from pysidtracker.playspec import (
    complexity,
    SpecError,
    SpecRuntimeError,
    elaborate,
    load,
    native,
)
from pysidtracker.playspec import outorder as O
from pysidtracker.playspec import outproof, parse, smt
from pysidtracker.playspec.emit import emit, isomorphic
from pysidtracker.playspec.pyexec import compile_program, controls
from pysidtracker.playspec.validate import StreamValidator

from .test_playspec_event_examples import pattern
from .test_playspec_events import frames
from .test_playspec_native import run_both

pytestmark = pytest.mark.strain_exempt(
    reason="asserts values, diagnostics and agreement between executors, which coverage cannot see"
)
fast = pytest.mark.skipif(not native.available(), reason="the native core is not built")

EXAMPLE = (
    Path(__file__).resolve().parents[1] / "examples" / "arrays" / "pattern-stream.play"
)


@pytest.fixture(autouse=True)
def backend():
    """These tests do not use the 6502 engines the suite is otherwise run on."""
    return None


WATCH = tuple(range(0x40, 0x47))
ORDER_AT, PATTERNS_AT = 0x2800, 0x3000


def song(seed):
    """Three random patterns and an orderlist: ``(patterns, firsts, order)``."""
    rng = random.Random(seed)
    patterns = [pattern(rng) for _ in range(3)]
    firsts = [sum(map(len, patterns[:k])) for k in range(3)]
    order = [rng.randrange(3) for _ in range(20)] + [0xFF]
    return patterns, firsts, order


def array_form(seed, tail=2, lead=4):
    """The example bound to the song of ``seed`` as lists; the orderlist starts at ``lead``."""
    patterns, firsts, order = song(seed)
    flat = [b for p in patterns for b in p] + [0] * tail
    binding = {
        "w": 0x40,
        "order_start": lead,
        "pattern_bytes": flat,
        "pattern_first": firsts,
        "order_bytes": [0xEE] * lead + order,
    }
    return elaborate(load(EXAMPLE), binding)


def memory_text():
    """The example with its lists in memory."""
    text = EXAMPLE.read_text()
    for old, new in (
        ("pattern_bytes, pattern_first, order_bytes: list", "pattern_first: addr"),
        ("state pattern_bytes: u8[] ro = L.pattern_bytes;\n", ""),
        (
            "state pattern_first: u16[] ro = L.pattern_first;",
            "cell pattern_first: u16[3] @ L.pattern_first ro;",
        ),
        ("state order_bytes: u8[] ro = L.order_bytes;\n", ""),
        ("table orderlist in order_bytes via seqt", "table orderlist via seqt"),
        ("stream in pattern_bytes @", "stream @"),
    ):
        assert old in text
        text = text.replace(old, new)
    return text


def memory_form(seed):
    """The memory form of the example bound to the song of ``seed``, and that memory."""
    text = memory_text()
    patterns, firsts, order = song(seed)
    memory = {ORDER_AT: order}
    for k, ptn in enumerate(patterns):
        memory[PATTERNS_AT + firsts[k]] = ptn
    memory[0x2900] = [
        b for f in firsts for b in ((PATTERNS_AT + f) & 255, (PATTERNS_AT + f) >> 8)
    ]
    binding = {"w": 0x40, "order_start": ORDER_AT, "pattern_first": 0x2900}
    return elaborate(parse(text), binding), memory


@pytest.mark.parametrize("seed", range(6))
def test_a_pattern_stream_over_an_array_plays_as_over_memory(seed):
    prog, memory = memory_form(seed)
    over_memory = frames(prog, memory, 150, WATCH)
    over_array = frames(array_form(seed), {}, 150, WATCH)
    assert over_array == over_memory
    assert len({row[1] for row in over_array}) == 3
    assert len({row[4] for row in over_array}) > 3


def test_the_example_stream_cursor_reads_the_array_in_place_of_memory():
    prog = array_form(0)
    cursor = prog.objects["pattern"] if "pattern" in prog.objects else None
    cursor = cursor or next(
        o for o in prog.objects.values() if o.kind == "cursor" and o.tabs
    )
    assert len(cursor.tabs) == cursor.spec[2] == 3 and len(set(cursor.tabs)) == 1
    assert cursor.tabs[0][-2:] == (0, 0)


@fast
@pytest.mark.parametrize("seed", range(3))
def test_native_agrees_on_the_array_example_and_reports_a_read_past_its_end(seed):
    assert run_both(array_form(seed), bytearray(0x10000), 150) is None
    error = run_both(array_form(seed, tail=1), bytearray(0x10000), 150)
    assert isinstance(error, SpecRuntimeError)
    assert "past its" in str(error)


def test_a_cursor_reads_a_whole_row_ahead_so_the_binder_pads_by_the_fields_less_one():
    for tail in (2, 3):
        frames(array_form(1, tail), {}, 150, WATCH)
    with pytest.raises(SpecRuntimeError, match="past its"):
        frames(array_form(1, 1), {}, 150, WATCH)


def test_the_example_round_trips_through_emit():
    prog = array_form(2)
    back = elaborate(parse(emit(prog, "example")), {})
    flat = O.inline(prog.body, prog.procs)
    objects = (prog.objects, back.objects)
    assert isomorphic(flat, O.inline(back.body, back.procs), objects) is None
    assert frames(back, {}, 60, WATCH) == frames(prog, {}, 60, WATCH)


PROC = """
spec streams;
layout { rows: list; }
cell out: u8 @ 0x20;
cell echo: u8 @ 0x21;
cell at: u8 @ 0x22;
cell pos: u8 @ 0x23;
cell chip: u8[25] @ 0xD400 out;
state rows: u8[] ro = L.rows;
proc take(s: stream) -> u8 {
  let a = next s;
  let b = next s;
  return a + b + s[0] + s[1] + (a ^ b) + (a & 0x0F) * 3;
}
machine m {
  step {
    stream s in rows = at @ pos;
    stream t in rows = at + 2 @ 0;
    chip[0] := take(s);
    chip[7] := take(t);
    out := take(s);
    echo := t[0] + t.pos;
    pos := s.pos;
  }
}
schedule { m; }
"""
ROWS = [(37 * i + 11) % 251 for i in range(24)]


def model(at, pos):
    """What :data:`PROC` stores to ``out``, ``echo`` and ``pos``, by hand."""

    def take(first):
        a, b = ROWS[first], ROWS[first + 1]
        return (
            a + b + ROWS[first + 2] + ROWS[first + 3] + (a ^ b) + (a & 15) * 3
        ) & 255

    return take(at + pos + 2), (ROWS[at + 4] + 2) & 255, pos + 4


@pytest.mark.parametrize("at,pos", [(0, 0), (3, 1), (9, 2), (0, 12)])
def test_a_proc_stream_over_an_array_reads_from_its_start_index(at, pos):
    prog = elaborate(parse(PROC), {"rows": ROWS})
    mem = bytearray(0x10000)
    mem[0x22], mem[0x23] = at, pos
    compile_program(prog)(mem.__getitem__, mem.__setitem__, controls(prog))
    assert (mem[0x20], mem[0x21], mem[0x23]) == model(at, pos)
    assert (mem[0x20], mem[0x21], mem[0x23]) != model(at + 1, pos)


def test_the_array_stream_is_the_memory_stream():
    text = PROC.replace("state rows: u8[] ro = L.rows;\n", "").replace(
        "layout { rows: list; }", "layout { rows: addr; }"
    )
    text = text.replace("s in rows = at @", "s = L.rows + word(at, 0) @").replace(
        "t in rows = at + 2 @", "t = L.rows + word(at + 2, 0) @"
    )
    over_memory = elaborate(parse(text), {"rows": 0x2000})
    over_array = elaborate(parse(PROC), {"rows": ROWS})
    rng = random.Random(5)
    for _ in range(10):
        given = {0x22: [rng.randrange(8)], 0x23: [rng.randrange(8)], 0x2000: ROWS}
        watch = (0x20, 0x21, 0x23, 0xD400, 0xD407)
        assert frames(over_array, given, 1, watch) == frames(
            over_memory, given, 1, watch
        )


def test_a_stream_past_the_end_is_an_error_and_never_wraps():
    prog = elaborate(parse(PROC), {"rows": ROWS})
    mem = bytearray(0x10000)
    mem[0x22], mem[0x23] = 20, 0
    call, state = compile_program(prog), controls(prog)
    with pytest.raises(SpecRuntimeError, match="past the 24 items"):
        call(mem.__getitem__, mem.__setitem__, state)
    mem[0x22] = 250
    with pytest.raises(SpecRuntimeError, match="items"):
        call(mem.__getitem__, mem.__setitem__, state)


def test_a_stream_that_starts_past_its_array_is_an_elaboration_error():
    text = PROC.replace("stream t in rows = at + 2 @ 0;", "stream t in rows = 25 @ 0;")
    with pytest.raises(SpecError, match="stream starts at 25, past the 24 items"):
        elaborate(parse(text), {"rows": ROWS})


@pytest.mark.parametrize(
    "old,new,message",
    [
        ("s in rows =", "s in nope =", "nope is not a state array"),
        ("state rows: u8[] ro", "state rows: u8[]", r"must be a `u8\[\] ro`"),
        (
            "state rows: u8[] ro = L.rows;",
            "state rows: u16[] ro = L.rows;",
            "must be a",
        ),
    ],
)
def test_a_stream_needs_a_ro_byte_array(old, new, message):
    text = PROC.replace(old, new)
    if (
        "= L.rows;" in text
        and "u8[] =" not in text
        and old.startswith("state rows: u8[] ro =")
    ):
        pass
    with pytest.raises(SpecError, match=message):
        elaborate(parse(text), {"rows": ROWS})


@fast
@pytest.mark.parametrize("at,pos", [(0, 0), (5, 3), (18, 0)])
def test_native_agrees_on_proc_streams_and_their_end(at, pos):
    prog = elaborate(parse(PROC), {"rows": ROWS})
    mem = bytearray(0x10000)
    mem[0x22], mem[0x23] = at, pos
    assert run_both(prog, mem, 1) is None
    mem[0x22] = 21
    assert isinstance(run_both(prog, mem, 1), SpecRuntimeError)


def test_reordering_with_the_proc_kept_proves_and_round_trips():
    prog = elaborate(parse(PROC), {"rows": ROWS})
    assert O.outlined(prog.body, prog.procs)
    result = O.reorder(prog, O.ORDER)
    assert result.program.procs
    back = elaborate(parse(emit(result.program, "ordered")), {})
    flat = O.inline(result.program.body, result.program.procs)
    assert isomorphic(flat, O.inline(back.body, back.procs)) is None
    assert any("take" in pid for pid in back.procs)
    assert outproof.prove(prog, O.ORDER, timeout_ms=60000).holds


def test_the_bounded_check_finds_a_stream_that_can_leave_its_array():
    unconstrained = elaborate(parse(PROC), {"rows": ROWS})
    found = smt.bmc(unconstrained, 1, timeout_ms=60000)
    assert not found.holds and "past the 24 items" in found.what
    masked = PROC.replace(
        "stream s in rows = at @ pos;", "stream s in rows = at & 7 @ pos & 7;"
    )
    masked = masked.replace(
        "stream t in rows = at + 2 @ 0;", "stream t in rows = 2 @ 0;"
    )
    assert smt.bmc(elaborate(parse(masked), {"rows": ROWS}), 1, timeout_ms=60000).holds


GRAMMAR = """
spec g;
layout { rows: list; }
cell q: u8 @ 0x20;
cell cpos: u8 @ 0x21;
cell note: u8 @ 0x22;
state rows: u8[] ro = L.rows;
cursor ptn { stream in rows @ 4 grammar "tests.test_bytegrammar:TOY:seq"; pos cpos;
  on row next;
  field n -> note = n;
}
machine m { step {
  stream s in rows = 0 @ 0 grammar "tests.test_bytegrammar:TOY:seq";
  q := next s;
  fire row;
} }
schedule { m; }
"""
SEQ = "tests.test_bytegrammar:TOY:seq"


def validated(rows, strict=False):
    """The validator after one call of :data:`GRAMMAR` bound to ``rows``."""
    prog = elaborate(parse(GRAMMAR), {"rows": rows})
    validator = StreamValidator(strict=strict)
    mem = bytearray(0x10000)
    hook = validator.hook(prog, mem.__getitem__)
    compile_program(prog)(mem.__getitem__, mem.__setitem__, controls(prog), hook)
    return prog, validator


def test_grammars_parse_the_array_a_stream_reads():
    prog, validator = validated([1, 2, 0xFF, 0, 1, 2, 0xFF, 0, 0])
    assert sorted(prog.grammar_arrays) == [0, 1]
    assert validator.results == {
        (f"{SEQ} in rows", 0): None,
        (f"{SEQ} in rows", 4): None,
    }
    _prog, validator = validated([1, 2, 3, 0, 1, 2, 3, 0, 0])
    assert sorted(validator.failures()) == [
        (f"{SEQ} in rows", 0),
        (f"{SEQ} in rows", 4),
    ]
    with pytest.raises(Exception, match=r"stream at rows\[0\] is not"):
        validated([1, 2, 3, 0, 1, 2, 3, 0, 0], strict=True)


def test_a_stream_cut_short_by_the_array_does_not_parse():
    _prog, validator = validated([1, 2, 0xFF, 0, 1, 2])
    assert list(validator.failures()) == [(f"{SEQ} in rows", 4)]


SEQT = """
spec vias;
layout { rows: list; starts: list; }
cell o: u8 @ 0x20;
cell w: u16 @ 0x22;
cell sel: u8 @ 0x24;
cell r: u8 @ 0x25;
cell o2: u8 @ 0x26;
cell o3: u8 @ 0x27;
cell o4: u8 @ 0x28;
cell ptr: u16 @ 0x30;
cell idx: u8 @ 0x32;
state rows: u8[] ro = L.rows;
state starts: u16[] ro = L.starts;
state seqt: u16 = 0;
state small: u8 = 0;
table ord in rows via seqt stride 3 { kind: u8; val: u16 @ 1; }
table byptr in rows via ptr stride 1 { b: u8; }
table byidx in rows via idx stride 1 { b: u8; }
table bysmall in rows via small stride 2 { b: u8 @ 1; }
table many in rows via starts stride 1 { b: u8; }
machine m {
  step {
    seqt := sel;
    small := sel;
    o := ord[r].kind;
    w := ord[r].val;
    o2 := byptr[r].b + byidx[r].b;
    o3 := many[sel][r].b;
    o4 := bysmall[r].b + small;
  }
}
schedule { m; }
"""
STARTS = [0, 5, 11]


def via_run(sel, r, ptr, idx, rows=ROWS, text=SEQT):
    """The cells :data:`SEQT` stores for one call."""
    prog = elaborate(parse(text), {"rows": rows, "starts": STARTS})
    mem = bytearray(0x10000)
    mem[0x24], mem[0x25], mem[0x30], mem[0x31], mem[0x32] = (
        sel,
        r,
        ptr & 255,
        ptr >> 8,
        idx,
    )
    compile_program(prog)(mem.__getitem__, mem.__setitem__, controls(prog))
    return prog, mem


@pytest.mark.parametrize("seed", range(12))
def test_a_table_reads_the_array_from_the_start_a_state_or_cell_holds(seed):
    rng = random.Random(seed)
    sel, r, ptr, idx = (
        rng.randrange(3),
        rng.randrange(3),
        rng.randrange(8),
        rng.randrange(8),
    )
    _prog, mem = via_run(sel, r, ptr, idx)
    at = sel + 3 * r
    assert mem[0x20] == ROWS[at]
    assert mem[0x22] | mem[0x23] << 8 == ROWS[at + 1] | ROWS[at + 2] << 8
    assert mem[0x26] == (ROWS[ptr + r] + ROWS[idx + r]) & 255
    assert mem[0x27] == ROWS[STARTS[sel] + r]
    assert mem[0x28] == (ROWS[sel + 2 * r + 1] + sel) & 255


@fast
def test_native_agrees_on_via_tables_and_their_end():
    prog = elaborate(parse(SEQT), {"rows": ROWS, "starts": STARTS})
    mem = bytearray(0x10000)
    mem[0x24], mem[0x25] = 2, 2
    assert run_both(prog, mem, 3) is None
    mem[0x25] = 40
    assert isinstance(run_both(prog, mem, 1), SpecRuntimeError)
    mem[0x24], mem[0x25] = 5, 0
    assert isinstance(run_both(prog, mem, 1), SpecRuntimeError)


def test_a_via_table_over_an_array_past_its_end_is_a_runtime_error():
    with pytest.raises(SpecRuntimeError, match="past the 24 items"):
        via_run(0, 8, 0, 0)
    with pytest.raises(SpecRuntimeError, match="past the 3 items"):
        via_run(3, 0, 0, 0)


@pytest.mark.parametrize(
    "old,new,message",
    [
        ("via seqt stride 3", "via nope stride 3", "table ord: nope is not a u16 cell"),
        ("many[sel][r].b", "many[r].b", "table many takes 2 index"),
        ("o := ord[r].kind;", "ord[r].kind := 1;", "write to table ord"),
    ],
)
def test_misused_via_tables_are_diagnosed(old, new, message):
    text = SEQT.replace(old, new)
    with pytest.raises(SpecError, match=message):
        elaborate(parse(text), {"rows": ROWS, "starts": STARTS})


def test_a_memory_table_via_a_pointer_still_takes_only_a_u16():
    text = SEQT.replace("table byidx in rows via idx", "table byidx via idx")
    with pytest.raises(SpecError, match="table byidx: idx is not a u16 cell"):
        elaborate(parse(text), {"rows": ROWS, "starts": STARTS})


def test_a_via_table_over_an_array_proves_and_reorders():
    text = """
spec ord;
layout { rows: list; }
cell chip: u8[25] @ 0xD400 out;
cell sel: u8 @ 0x2000;
state rows: u8[] ro = L.rows;
table orderlist in rows via seqt stride 1 { b: u8; }
machine voice[x in 0..3] {
  state seqt: u16 = x * 3;
  step {
    chip[7 * x] := orderlist[sel & 1].b;
    chip[7 * x + 1] := orderlist[1].b;
  }
}
schedule { for v in 0..3 { voice[v]; } }
"""
    prog = elaborate(parse(text), {"rows": list(range(1, 12))})
    result = O.reorder(prog, O.ORDER)
    back = elaborate(parse(emit(result.program, "ordered")), {})
    flat = O.inline(result.program.body, result.program.procs)
    assert isomorphic(flat, O.inline(back.body, back.procs)) is None
    assert outproof.prove(prog, O.ORDER, timeout_ms=60000).holds
    mem = bytearray(0x10000)
    mem[0x2000] = 2
    sids = []
    compile_program(prog)(
        mem.__getitem__, lambda a, v: sids.append((a, v)), controls(prog)
    )
    written = dict(sids)
    assert [written[0xD400 + 7 * x] for x in range(3)] == [1, 4, 7]
    assert [written[0xD401 + 7 * x] for x in range(3)] == [2, 5, 8]


def test_the_complexity_tool_reads_the_array_forms():
    analysis = complexity.analyse(parse(EXAMPLE.read_text()))
    kinds = {n: analysis.layout[n]["kind"] for n in ("pattern_first", "order_bytes")}
    assert kinds == {"pattern_first": "list", "order_bytes": "list"}
    assert analysis.addressing == []
    assert analysis.layout["pattern_first"]["decl"] == 1


HEADER = """
spec aw;
layout { x: addr; tab, first: list; }
cell pos: u8 @ 0x20;
cell cnt: u8 @ 0x21;
cell pw: u16 @ (0x22, 0x23);
cell band: u8 @ 0x24;
cell fine: u16 @ (0x25, 0x26);
cell tail: u8 @ 0x27;
cell sel: u8 @ 0x28;
"""
SLICES = """
cursor t {
  TABLE
  pos pos; count cnt; values pw, band, fine, tail; start 0;
  on frame step; on note reset;
  match c0 {
    0..0x3f => sweep(into=pw, count=c0, step=c1) then band = c2;
    0x40..0x4f => hold(c0 & 0x0f) then band = c1;
    0x50..0x5f => sweep(into=FINE, count=c0 & 3, step=c1, frac=3);
    0x80..0xfd => value(pw.hi = c0 & 0x7f, pw.lo = c1, band = c0 >> 4, fine.lo = c2);
    0xfe => jump(c1);
    0xff => stop;
  } SHARED
}
machine m { step { fire frame; } }
schedule { m; }
"""
BASED = (
    HEADER
    + "state tab: u8[] ro = L.tab;\nstate first: u8[] ro = L.first;\n"
    + SLICES.replace("TABLE", "table tab @ first[sel] width 3;")
    .replace("FINE", "fine.hi")
    .replace("SHARED", "then tail = c2;")
)
EXPLICIT_THEN = (
    SLICES.replace("TABLE", "table @ 0x2000 + word(first[sel], 0) width 3;")
    .replace("FINE", "fine")
    .replace("SHARED", "")
    .replace("then band = c2;", "then band = c2, tail = c2;")
    .replace("then band = c1;", "then band = c1, tail = c2;")
    .replace("frac=3);", "frac=3) then tail = c2;")
    .replace("fine.lo = c2);", "fine.lo = c2) then tail = c2;")
)
SLICE0 = [2, 0x10, 7, 0x42, 0x99, 0, 0x81, 0x20, 3, 0x51, 5, 1, 0xFE, 0, 0]
SLICE1 = [1, 0x30, 2, 0x51, 4, 9, 0x80, 0x11, 5, 0xFE, 3, 0]
WATCH_AW = tuple(range(0x20, 0x29))


def based(first=(0, 15)):
    """The array form of :data:`BASED` over two instrument slices."""
    return elaborate(parse(BASED), {"x": 0, "tab": SLICE0 + SLICE1, "first": first})


def twin():
    """The memory form: the slices where the table of :data:`BASED` is, shared ``then`` spelt out."""
    text = HEADER.replace("; tab, first: list", "").replace(
        "cell sel", "cell first: u8[2] @ 0x30 ro;\ncell sel"
    )
    return elaborate(parse(text + EXPLICIT_THEN), {"x": 0})


@pytest.mark.parametrize("sel", [0, 1])
def test_a_cursors_array_table_has_a_base_and_the_row_tail_is_shared(sel):
    given = {0x28: [sel], 0x30: [0, 15], 0x2000: SLICE0 + SLICE1}
    over_array = frames(based(), given, 24, WATCH_AW)
    assert over_array == frames(twin(), given, 24, WATCH_AW)
    assert len({row[0] for row in over_array}) > 3
    assert {row[7] for row in over_array} - {0}


def test_the_base_moves_the_slice_a_jump_is_relative_to():
    seen = [frames(based(), {0x28: [s]}, 24, WATCH_AW) for s in (0, 1)]
    assert seen[0] != seen[1]
    shifted = frames(based((0, 12)), {0x28: [1]}, 6, WATCH_AW)
    assert shifted != seen[1][:6]


@fast
@pytest.mark.parametrize("sel", [0, 1])
def test_native_agrees_on_a_based_cursor_and_its_end(sel):
    mem = bytearray(0x10000)
    mem[0x28] = sel
    assert run_both(based(), mem, 24) is None
    assert isinstance(run_both(based((0, 40)), mem, 24), SpecRuntimeError) or not sel


def test_a_based_cursor_round_trips_through_emit():
    prog = based()
    back = elaborate(parse(emit(prog, "based")), {})
    flat = O.inline(prog.body, prog.procs)
    assert (
        isomorphic(flat, O.inline(back.body, back.procs), (prog.objects, back.objects))
        is None
    )
    given = {0x28: [1]}
    assert frames(back, given, 24, WATCH_AW) == frames(prog, given, 24, WATCH_AW)


def test_a_fraction_sweep_into_the_high_byte_is_the_sweep_of_the_word():
    word = BASED.replace("fine.hi", "fine")
    given = {0x28: [0]}
    assert frames(
        elaborate(parse(word), {"x": 0, "tab": SLICE0 + SLICE1, "first": [0, 15]}),
        given,
        20,
        WATCH_AW,
    ) == frames(based(), given, 20, WATCH_AW)
    byte = BASED.replace("frac=3", "frac=0")
    assert frames(
        elaborate(parse(byte), {"x": 0, "tab": SLICE0 + SLICE1, "first": [0, 15]}),
        given,
        20,
        WATCH_AW,
    ) != frames(based(), given, 20, WATCH_AW)
    bad = BASED.replace("sweep(into=fine.hi,", "sweep(into=band.hi,")
    with pytest.raises(SpecError, match="frac needs a word"):
        elaborate(parse(bad), {"x": 0, "tab": SLICE0, "first": [0, 0]})


def test_a_shared_then_does_not_follow_a_stop_or_a_jump():
    stop = [0xFF, 0, 0, 0xFF, 0, 0]
    binding = {"x": 0, "tab": stop, "first": [0, 3]}
    prog = elaborate(parse(BASED), binding)
    assert {row[7] for row in frames(prog, {}, 4, WATCH_AW)} == {0}
    jump = elaborate(parse(BASED), {**binding, "tab": [0xFE, 3, 7, 0xFF, 0, 0]})
    assert {row[7] for row in frames(jump, {}, 4, WATCH_AW)} == {0}
    value = elaborate(parse(BASED), {**binding, "tab": [0x81, 1, 7, 0xFF, 0, 0]})
    assert {row[7] for row in frames(value, {}, 4, WATCH_AW)} == {7}


LANES = """
spec lanes;
cell chip: u8[25] @ 0xD400 out;
cell o: u8 @ 0x20;
cell w: u16 @ 0x22;
cell echo: u8 @ 0x24;
state fm: u16 = 0x1234;
machine m {
  step {
    fm.hi := o;
    fm.lo += 1;
    w := fm;
    echo := fm.hi + fm.lo;
    chip[0] := fm.lo;
    chip[1] := fm.hi;
  }
}
schedule { m; }
"""


def test_a_u16_states_bytes_are_written_and_read_as_lanes():
    prog = elaborate(parse(LANES))
    mem = bytearray(0x10000)
    mem[0x20] = 0x7F
    state = controls(prog)
    for n in range(3):
        compile_program(prog)(mem.__getitem__, mem.__setitem__, state)
        assert mem[0x22] | mem[0x23] << 8 == 0x7F00 | (0x35 + n)
        assert mem[0x24] == (0x7F + 0x35 + n) & 255
    mem[0x20] = 0xFF
    state[0] = 0x12FF
    compile_program(prog)(mem.__getitem__, mem.__setitem__, state)
    assert mem[0x22] | mem[0x23] << 8 == 0xFF00


def test_lane_writes_reorder_prove_and_agree_natively():
    prog = elaborate(parse(LANES))
    result = O.reorder(prog, O.ORDER)
    back = elaborate(parse(emit(result.program, "ordered")), {})
    assert (
        isomorphic(
            O.inline(result.program.body, result.program.procs),
            O.inline(back.body, back.procs),
        )
        is None
    )
    assert outproof.prove(prog, O.ORDER, timeout_ms=60000).holds
    if native.available():
        mem = bytearray(0x10000)
        mem[0x20] = 9
        assert run_both(prog, mem, 4) is None
