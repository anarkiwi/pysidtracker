"""Bit fields, tables, latches, accumulators, the unified cursor, instances and output
binding, each against the statements it abbreviates, and native against Python."""

import random

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pysidtracker.playspec import SpecError, compile_program, elaborate, native, parse
from pysidtracker.playspec.pyexec import controls
from pysidtracker.playspec import outorder as O
from pysidtracker.playspec.emit import emit, isomorphic

from .test_playspec_events import build, frames
from .test_playspec_native import run_both, table_memory

pytestmark = pytest.mark.strain_exempt(
    reason="asserts the values, diagnostics and equalities of the construct semantics"
)


@pytest.fixture(autouse=True)
def backend():
    """These tests do not use the 6502 engines the suite is otherwise run on."""
    return None


def both(text, memory, count=6, **binding):
    """The program of ``text`` after ``count`` calls on Python and, when built, native."""
    prog = build(text, **binding)
    if native.available():
        assert run_both(prog, memory, count) is None
    return prog


def memory_of(seed, size=0x300):
    rng = random.Random(seed)
    mem = bytearray(0x10000)
    mem[:size] = bytes(rng.randrange(256) for _ in range(size))
    return mem


def same(a, b, seed, watch, count=8, **binding):
    """Whether specs ``a`` and ``b`` leave ``watch`` equal after ``count`` calls."""
    mem = memory_of(seed)
    pa = build(a, **binding)
    pb = build(b, **binding)
    mem_a = {0: bytes(mem[:0x300])}
    return frames(pa, mem_a, count, watch) == frames(pb, mem_a, count, watch)


WATCH = tuple(range(0x10, 0x60))


def check_roundtrip(text, **binding):
    """``text`` reordered, emitted and elaborated again is the same program."""
    result = O.reorder(build(text, **binding), O.ORDER)
    back = elaborate(parse(emit(result.program, "again")), {})
    objects = (result.program.objects, back.objects)
    assert (
        isomorphic(result.program.body, O.inline(back.body, back.procs), objects)
        is None
    )
    return result.program


# -- bit fields -------------------------------------------------------------------------

BITS = """
spec b;
cell ad: u8 @ 0x10 { hi: 4..7, lo: 0..3 };
cell sel: u8 @ 0x11;
cell arr: u8[4] @ 0x20 { flag: 7, low: 0..6 };
cell wide: u16 @ 0x30 { top: 8..15, mid: 4..11 };
cell plain: u8 @ 0x40;
machine m { step {
  ad.hi := sel;
  ad.lo += 3;
  arr[sel & 3].flag := sel;
  arr[1].low := ad.hi;
  wide.top := ad.lo;
  wide.mid ^= 0x5A;
  plain.hi := ad.hi;
  mem[0x50] := arr[2].flag;
}}
schedule { m; }
"""

BITS_BY_HAND = """
spec b;
cell ad: u8 @ 0x10;
cell sel: u8 @ 0x11;
cell arr: u8[4] @ 0x20;
cell wide: u16 @ 0x30;
cell plain: u8 @ 0x40;
machine m { step {
  ad := (ad & 0x0F) | ((sel & 0x0F) << 4);
  ad := (ad & 0xF0) | (((ad & 0x0F) + 3) & 0x0F);
  arr[sel & 3] := (arr[sel & 3] & 0x7F) | ((sel & 1) << 7);
  arr[1] := (arr[1] & 0x80) | ((ad >> 4) & 0x7F);
  wide := (wide & 0x00FF) | (u16(ad & 0x0F) << 8);
  wide := (wide & 0xF00F) | (u16((((wide >> 4) & 0xFF) ^ 0x5A) & 0xFF) << 4);
  plain.hi := ad.hi;
  mem[0x50] := (arr[2] >> 7) & 1;
}}
schedule { m; }
"""


@pytest.mark.parametrize("seed", range(8))
def test_named_bit_fields_are_the_masks_and_shifts_they_abbreviate(seed):
    mem = memory_of(seed)
    given_mem = {0: bytes(mem[:0x100])}
    assert frames(both(BITS, mem), given_mem, 5, WATCH) == frames(
        build(BITS_BY_HAND), given_mem, 5, WATCH
    )


def test_a_bit_field_that_is_not_declared_falls_back_to_the_nibble():
    text = "spec b; cell c: u8 @ 0x10 { top: 6..7 }; machine m { step { c.hi := 3; mem[0x11] := c.top; } } schedule { m; }"
    seen = frames(build(text), {0x10: [0x0F]}, 1, (0x10, 0x11))
    assert seen == [(0x3F, 0)]


@pytest.mark.parametrize(
    "decl, use, message",
    [
        ("cell c: u8 @ 0x10 { f: 4..9 };", "c.f := 1;", "not in u8"),
        ("cell c: u8 @ 0x10 { f: 5..4 };", "c.f := 1;", "not in u8"),
        ("cell c: u8[2] @ 0x10 { f: 0..3 };", "c[0].g := 1;", "cannot assign"),
        ("cell c: u8 @ 0x10;", "c[0].f := 1;", "cannot assign"),
    ],
)
def test_bit_field_declarations_are_checked(decl, use, message):
    with pytest.raises(SpecError, match=message):
        build(f"spec b; {decl} machine m {{ step {{ {use} }} }} schedule {{ m; }}")


# -- tables and split cells -------------------------------------------------------------

TABLES = """
spec t;
layout { voices, ptr_lo, ptr_hi, reloc: addr; }
table voices @ L.voices stride 7 { freq: u16; ctrl: u8; ad: u8 @ 5; sr: u8; }
cell ptrs: u16[3] @ (L.ptr_lo, L.ptr_hi);
cell cur: u16 @ 0x60;
table pattern via ptrs stride 3 offset L.reloc { note: u8; inst: u8; fx: u8; }
table one via cur stride 2 { a: u8; b: u8; }
cell sel: u8 @ 0x62;
machine m { step {
  voices[sel & 1].freq := voices[1].freq + 0x101;
  voices[2].ad := voices[1].ctrl;
  mem[0x80] := pattern[sel & 1][2].inst;
  pattern[1][1].fx := sel;
  mem[0x81] := one[3].b;
  one[sel & 7].a := hi(ptrs[2]);
  mem[0x82] := lo(ptrs[2]);
  mem[0x83] := hi(ptrs[sel & 1]);
}}
schedule { m; }
"""

TABLES_BY_HAND = """
spec t;
layout { voices, ptr_lo, ptr_hi, reloc: addr; }
cell ptrs: u16[3] @ (L.ptr_lo, L.ptr_hi);
cell cur: u16 @ 0x60;
cell sel: u8 @ 0x62;
machine m { step {
  let f = word(mem[L.voices + 7 + 0], mem[L.voices + 7 + 1]) + 0x101;
  let at = L.voices + 7 * u16(sel & 1);
  mem[at] := lo(f);
  mem[at + 1] := hi(f);
  mem[L.voices + 14 + 5] := mem[L.voices + 7 + 2];
  mem[0x80] := mem[ptrs[sel & 1] + L.reloc + 2 * 3 + 1];
  mem[ptrs[1] + L.reloc + 3 + 2] := sel;
  mem[0x81] := mem[cur + 3 * 2 + 1];
  mem[cur + 2 * u16(sel & 7)] := hi(ptrs[2]);
  mem[0x82] := lo(ptrs[2]);
  mem[0x83] := hi(ptrs[sel & 1]);
}}
schedule { m; }
"""
TABLE_BINDING = {"voices": 0x100, "ptr_lo": 0x200, "ptr_hi": 0x210, "reloc": 0x10}


def table_memory_for(seed):
    rng = random.Random(seed)
    mem = memory_of(seed, 0x300)
    for i in range(3):
        mem[0x200 + i] = rng.randrange(256)
        mem[0x210 + i] = rng.choice((0x20, 0x30, 0x31))
    mem[0x60:0x62] = (0x4000 + rng.randrange(0x100)).to_bytes(2, "little")
    return mem


@pytest.mark.parametrize("seed", range(10))
def test_tables_address_rows_fields_and_sub_tables_as_hand_written_arithmetic(seed):
    mem = table_memory_for(seed)
    watch = tuple(range(0x80, 0x84)) + tuple(range(0x100, 0x120)) + (0x62,)
    image = {
        0: bytes(mem[:0x300]),
        0x2000: bytes(mem[0x2000:0x2100]),
        0x4000: bytes(mem[0x4000:0x4100]),
    }
    one = build(TABLES, **TABLE_BINDING)
    two = build(TABLES_BY_HAND, **TABLE_BINDING)
    if native.available():
        assert run_both(one, mem, 4) is None
    assert frames(one, image, 4, watch) == frames(two, image, 4, watch)


@pytest.mark.parametrize(
    "decl, use, message",
    [
        ("table t @ 0x10 stride 2 { a: u8; b: u8 @ 0; }", "", "overlaps"),
        ("table t @ 0x10 stride 2 { a: u8; b: u16; }", "", "past the stride"),
        (
            "cell q: u8 @ 0x10; table t via q stride 2 { a: u8; }",
            "mem[0] := t[0].a;",
            "not a u16 cell",
        ),
        ("table t @ 0x10 stride 2 { a: u8; }", "mem[0] := t[0].zz;", "no field"),
        ("table t @ 0x10 stride 2 { a: u8; }", "mem[0] := t[0][1].a;", "takes 1 index"),
        ("table t @ 0x10 via q stride 2 { a: u8; }", "", "syntax"),
    ],
)
def test_table_declarations_and_uses_are_checked(decl, use, message):
    with pytest.raises(SpecError, match=message):
        build(f"spec t; {decl} machine m {{ step {{ {use} }} }} schedule {{ m; }}")


# -- latch and event chains -------------------------------------------------------------

CHAIN = """
spec chain;
cell gate: u8 @ 0x10;
cell pending: u8 @ 0x11;
cell wave: u8 @ 0x12;
cell ticks: u8 @ 0x13;
cell log: u8[16] @ 0x20;
cell at: u8 @ 0x14;
cell restart: u8 @ 0x15;
cell tempo: u8 @ 0x16;
cell cur: u16 @ 0x18;
cell next_ptr: u16 @ 0x1A;
event gate_off: row;
event gate_on: row;
counter beat: tempo { on frame -= 1; reload 2; wrap -> tick; }
counter gap: ticks { on tick += 1; wrap at 1 -> gate_on; on gate_off reset; }
latch wave := pending on gate_on;
latch cur := next_ptr on gate_on when pending != 0;
handler off on note { gate := 0; fire gate_off; }
handler on_ on gate_on { gate := 1; log[at & 15] := wave; at += 1; }
machine m { step { fire frame; if restart != 0 { restart := 0; fire note; } } }
schedule { m; }
"""


def test_a_note_ends_the_gate_and_ticks_later_start_the_next_one():
    prog = both(CHAIN, bytearray(0x10000), 1)
    mem = {0x11: [7], 0x1A: [0x34, 0x12]}
    run = frames(prog, mem, 9, (0x10, 0x12, 0x13, 0x14))
    assert run[0] == (1, 7, 0, 0) or run[0][3] == 0
    state = bytearray(0x10000)
    state[0x11] = 7
    state[0x1A:0x1C] = bytes([0x34, 0x12])
    state[0x15] = 1
    seen = frames(
        prog,
        {0x11: [7], 0x15: [1], 0x1A: [0x34, 0x12]},
        8,
        (0x10, 0x13, 0x14, 0x12, 0x18),
    )
    assert seen[0][0] == 0 and seen[0][1] == 0
    assert any(row[0] == 1 and row[2] == 1 for row in seen)
    assert seen[-1][3:] == (7, 0x34) or seen[-1][3] == 7


def test_a_latch_is_a_handler_that_copies_on_its_event():
    latch = "spec l; cell a: u8 @ 0x10; cell b: u8 @ 0x11; latch a := b on row when b != 0; machine m { step { fire row; } } schedule { m; }"
    hand = "spec l; cell a: u8 @ 0x10; cell b: u8 @ 0x11; handler h on row { if b != 0 { a := b; } } machine m { step { fire row; } } schedule { m; }"
    for b in (0, 5):
        mem = {0x10: [1], 0x11: [b]}
        assert frames(build(latch), mem, 2, (0x10,)) == frames(
            build(hand), mem, 2, (0x10,)
        )


# -- accumulators -----------------------------------------------------------------------

ACCUM = """
spec a;
cell pw: u16 @ 0x10;
cell dir: u8 @ 0x12;
cell beat: u8 @ 0x13;
cell goal: u16 @ 0x14;
cell stp: u16 @ 0x16;
cell ramp: u16 @ 0x20;
cell hit: u8 @ 0x30;
event turn: tick;
accum tri: pw { step stp; dir dir; on frame triangle; on turn flip; }
counter half: beat { on frame += 1; wrap at 3 -> turn; }
accum seek: ramp { step 5; on frame ramp to goal then stop -> row; }
counter hits: hit { on row += 1; }
machine m { step { fire frame; } }
schedule { m; }
"""

ACCUM_BY_HAND = """
spec a;
cell pw: u16 @ 0x10;
cell dir: u8 @ 0x12;
cell beat: u8 @ 0x13;
cell goal: u16 @ 0x14;
cell stp: u16 @ 0x16;
cell ramp: u16 @ 0x20;
cell hit: u8 @ 0x30;
machine m { step {
  if dir & 1 != 0 { pw := pw - stp; } else { pw := pw + stp; }
  if beat + 1 > 3 { beat := 0; dir ^= 1; } else { beat += 1; }
  if ramp != goal {
    if ramp < goal {
      if goal - ramp < 5 { ramp := goal; } else { ramp += 5; }
    } else {
      if ramp - goal < 5 { ramp := goal; } else { ramp -= 5; }
    }
    if ramp == goal { hit += 1; }
  }
}}
schedule { m; }
"""


@pytest.mark.parametrize("seed", range(10))
def test_accumulators_ramp_clamp_and_turn_as_the_hand_written_statements(seed):
    rng = random.Random(seed)
    mem = {
        0x10: [rng.randrange(256), rng.randrange(256)],
        0x14: [rng.randrange(256), rng.randrange(2)],
        0x16: [rng.randrange(1, 90), 0],
        0x20: [rng.randrange(256), 0],
    }
    watch = (0x10, 0x11, 0x12, 0x13, 0x20, 0x21, 0x30)
    got = frames(both(ACCUM, bytearray(0x10000), 1), mem, 40, watch)
    assert got == frames(build(ACCUM_BY_HAND), mem, 40, watch)


def test_a_ramp_to_a_target_stops_there_and_flags_the_arrival_once_per_step():
    text = "spec a; cell v: u16 @ 0x10; cell t: u16 @ 0x12; cell n: u8 @ 0x14; accum a: v { step 3; on frame ramp to t then stop -> row; } counter c: n { on row += 1; } machine m { step { fire frame; } } schedule { m; }"
    seen = frames(both(text, bytearray(0x10000), 1), {0x12: [10]}, 6, (0x10, 0x14))
    assert seen == [(3, 0), (6, 0), (9, 0), (10, 1), (10, 1), (10, 1)]


@pytest.mark.parametrize(
    "body, message",
    [
        ("on frame triangle;", "needs a dir"),
        ("on frame ramp -> frame;", "not below"),
        ("on frame ramp; on frame flip;", "twice"),
    ],
)
def test_accumulator_declarations_are_checked(body, message):
    text = f"spec a; cell v: u16 @ 0x10; accum a: v {{ {body} }} machine m {{ step {{ fire frame; }} }} schedule {{ m; }}"
    with pytest.raises(SpecError, match=message):
        build(text)


# -- counters ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "clause, start, expect",
    [
        ("on frame += 1; wrap at 2; reload 7;", 2, 7),
        ("on frame += 1; wrap at 2; reload 7;", 1, 2),
        ("on frame -= 1; reload 9;", 0, 9),
        ("on frame -= 2; reload 9;", 1, 9),
        ("on frame -= 2; reload 9;", 5, 3),
        ("on frame += 1; wrap when c == 3; reload 1;", 3, 1),
        ("on frame += 1; wrap when c == 3; reload 1;", 4, 5),
        ("on frame += 1; on note reset; reload 4;", 0, 1),
    ],
)
def test_counters_test_before_or_after_the_step_and_reload_on_a_wrap(
    clause, start, expect
):
    text = f"spec c; cell c: u8 @ 0x10; counter k: c {{ {clause} }} machine m {{ step {{ fire frame; }} }} schedule {{ m; }}"
    assert frames(both(text, bytearray(0x10000), 1), {0x10: [start]}, 1, (0x10,)) == [
        (expect,)
    ]


def test_a_counter_wrap_fires_its_event_into_the_counters_below():
    text = "spec c; cell a: u8 @ 0x10; cell b: u8 @ 0x11; cell d: u8 @ 0x12; counter x: a { on frame += 1; wrap at 1 -> tick; } counter y: b { on tick += 1; wrap at 1 -> row; } counter z: d { on row += 1; } machine m { step { fire frame; } } schedule { m; }"
    seen = frames(both(text, bytearray(0x10000), 1), {}, 12, (0x12,))
    assert seen[-1] == (3,) and seen[2] == (0,) and seen[3] == (1,)


# -- instances and owners ---------------------------------------------------------------

VOICES = """
spec v;
layout { w: addr; }
cell chip: u8[25] @ 0xD400 out;
cell gate: u8[3] @ 0x10;
cell owner_id: u8 @ 0x20;
owner filt: owner_id { id x + 1; none 0; on note claim when gate[x] != 0; on legato release; }
machine voice[x in 0..3] {
  record slot @ L.w stride 7 { freq: u16; pw: u16; ctrl: u8; }
  step {
    freq += 0x100 * (x + 1);
    chip[7 * x] := lo(freq);
    chip[7 * x + 1] := hi(freq);
    ctrl := ctrl + 1;
    if gate[x] != 0 { fire note; } else { fire legato; }
    if filt.mine { chip[23] := ctrl; }
  }
}
schedule { for v in 0..3 { voice[v]; } }
"""


def test_one_machine_instanced_per_voice_has_its_own_record_cells_and_claims_an_owner():
    prog = build(VOICES, w=0x100)
    assert {c.name for c in prog.cells if c.record} == {"freq", "pw", "ctrl"}
    mem = {0x10: [0, 1, 1]}
    seen = frames(prog, mem, 3, (0x20, 0x100 + 4, 0x107 + 4, 0x10E + 4))
    assert seen[0][0] == 3 and seen[0][1:] == (1, 1, 1) and seen[2][1:] == (3, 3, 3)
    free = frames(prog, {0x10: [0, 0, 0], 0x20: [2]}, 1, (0x20,))
    assert free == [(0,)]


# -- output binding ---------------------------------------------------------------------

BODY = """
layout { g: addr; }
machine m { step {
  chip[0] := 1; chip[4] := 0x11; chip[5] := 0x22; chip[6] := 0x33; chip[0] := 7;
}}
schedule { m; }
"""


def sid_writes(program):
    """The nonzero SID stores of one call of ``program`` on zeroed memory."""
    mem, out = bytearray(0x10000), []

    def put(address, value):
        out.append((address, value))
        mem[address] = value

    compile_program(program)(mem.__getitem__, put, controls(program))
    return [w for w in out if 0xD400 <= w[0] < 0xD419 and w[1]]


def test_a_staging_copy_differs_from_direct_output_only_in_its_declaration():
    direct = f'spec d; output chip @ 0xD400 order "{O.ORDER}";' + BODY
    ghost = f'spec g; output chip @ L.g order "{O.ORDER}" copy to 0xD400;' + BODY
    results = []
    for text, bind in ((direct, {"g": 0}), (ghost, {"g": 0x3000})):
        prog = elaborate(parse(text), bind)
        assert prog.outorder == O.ORDER and prog.sid == 0xD400
        result = O.reorder(prog)
        assert result.native == frozenset()
        results.append(sid_writes(result.program))
    assert (
        results[0]
        == results[1]
        == [(0xD400 + r, v) for r, v in ((4, 0x11), (6, 0x33), (5, 0x22), (0, 7))]
    )


def test_reorder_without_an_order_or_a_declaration_is_refused():
    prog = build(
        "spec d; cell c: u8 @ 0x10; machine m { step { c := 1; } } schedule { m; }"
    )
    with pytest.raises(O.OrderError, match="no order"):
        O.reorder(prog)


# -- the unified cursor -----------------------------------------------------------------

CURSOR = """
spec cur;
layout { tab: addr; }
cell pos: u8 @ 0x10;
cell cnt: u8 @ 0x11;
cell ret: u8 @ 0x12;
cell run: u8 @ 0x13;
cell note: u8 @ 0x14;
cell inst: u8 @ 0x15;
cell pw: u16 @ 0x16;
cell done: u8 @ 0x18;
cell start: u8 @ 0x19;
event table_end: tick;
cursor t {
  table @ L.tab width 3;
  pos pos; count cnt; ret ret; run run; start start;
  advance post;
  on frame step; on note reset;
  wrap -> table_end;
  match c0 {
    0x00..0x3f => value(note = c0, inst = c1) then pw.lo = c2;
    0x40..0x5f => hold(c1) then note = c2;
    0x60..0x6f => sweep(into = pw, count = c1, step = c2);
    0x70 => jump(c1) hold 0x00..0x3f;
    0x71 => call(c1);
    0x72 => return;
    0x73 when c1 == 1 && c2 != 0 => value(inst = 99);
    0x80..0x8f => run(0x80) then note = c0;
    0xfe => end;
    _ => stop;
  }
}
counter ended: done { on table_end += 1; }
machine m { step { fire frame; if mem[0x30] != 0 { mem[0x30] := 0; fire note; } } }
schedule { m; }
"""


def tab_memory(seed):
    rng = random.Random(seed)
    mem = bytearray(0x10000)
    kinds = [0x05, 0x45, 0x62, 0x70, 0x71, 0x72, 0x73, 0x80, 0x82, 0xFE, 0xFF, 0x20]
    for i in range(0x40):
        mem[0x2000 + 3 * i] = rng.choice(kinds)
        mem[0x2001 + 3 * i] = (
            rng.randrange(0, 0x30)
            if mem[0x2000 + 3 * i] in (0x70, 0x71)
            else rng.randrange(4)
        )
        mem[0x2002 + 3 * i] = rng.randrange(256)
    for k in range(0x10, 0x1A):
        mem[k] = rng.randrange(0, 3)
    for k in (0x10, 0x12):
        mem[k] = rng.randrange(0, 0x30) // 3 * 3
    return mem


@pytest.mark.parametrize("seed", range(12))
def test_every_cursor_row_kind_plays_the_same_natively(seed):
    mem = tab_memory(seed)
    prog = build(CURSOR, tab=0x2000)
    if native.available():
        assert run_both(prog, mem, 60) is None
    else:
        frames(
            prog, {0: bytes(mem[:0x40]), 0x2000: bytes(mem[0x2000:0x2200])}, 60, (0x10,)
        )


HAND_CURSOR = """
spec cur;
layout { tab: addr; }
cell tab: u8[0x200] @ L.tab ro;
cell pos: u8 @ 0x10;
cell cnt: u8 @ 0x11;
cell ret: u8 @ 0x12;
cell note: u8 @ 0x14;
cell inst: u8 @ 0x15;
cell pw: u16 @ 0x16;
machine m {
  step {
    let at = pos;
    let again = true;
    while again bound 256 {
      again := false;
      let c0 = tab[at];
      let c1 = tab[at + 1];
      let c2 = tab[at + 2];
      if c0 < 0x40 {
        note := c0; inst := c1; pw := (pw & 0xFF00) | u16(c2); at += 3;
      } elif c0 < 0x60 {
        if c0 >= 0x40 && cnt != c1 { cnt += 1; } else { cnt := 0; at += 3; note := c2; }
      } elif c0 == 0x70 {
        if c1 != at { at := c1; if !(tab[at] < 0x40) { again := true; } else { cnt := 0; } }
      } elif c0 == 0x71 {
        ret := at + 3; at := c1; again := true;
      } elif c0 == 0x72 {
        at := ret; again := true;
      }
    }
    pos := at;
  }
}
schedule { m; }
"""


@pytest.mark.parametrize("seed", range(10))
def test_value_hold_jump_call_and_return_rows_are_the_hand_written_loop(seed):
    rng = random.Random(seed)
    mem = bytearray(0x10000)
    for i in range(0x30):
        kind = rng.choice((0x05, 0x45, 0x71, 0x72, 0x70, 0x20))
        mem[0x2000 + 3 * i : 0x2003 + 3 * i] = bytes(
            (
                kind,
                (
                    rng.randrange(3)
                    if kind != 0x71 and kind != 0x70
                    else 3 * rng.randrange(0x30)
                ),
                rng.randrange(256),
            )
        )
    subset = """
spec cur;
layout { tab: addr; }
cell pos: u8 @ 0x10; cell cnt: u8 @ 0x11; cell ret: u8 @ 0x12;
cell note: u8 @ 0x14; cell inst: u8 @ 0x15; cell pw: u16 @ 0x16;
cursor t {
  table @ L.tab width 3; pos pos; count cnt; ret ret; advance post; on frame step;
  match c0 {
    0x00..0x3f => value(note = c0, inst = c1) then pw.lo = c2;
    0x40..0x5f => hold(c1) then note = c2;
    0x70 => jump(c1) hold 0x00..0x3f;
    0x71 => call(c1);
    0x72 => return;
    _ => stop;
  }
}
machine m { step { fire frame; } }
schedule { m; }
"""
    image = {0: bytes(0x20), 0x2000: bytes(mem[0x2000:0x2200])}
    watch = (0x10, 0x11, 0x12, 0x14, 0x15, 0x16)
    fast = build(subset, tab=0x2000)
    hand = build(HAND_CURSOR, tab=0x2000)
    try:
        want = frames(hand, image, 30, watch)
    except Exception:  # a jump loop the hand-written bound also rejects
        with pytest.raises(Exception):
            frames(fast, image, 30, watch)
        return
    assert frames(fast, image, 30, watch) == want


def test_pre_advance_rests_on_the_last_byte_and_a_pending_run_holds_the_position():
    text = """
spec p;
cell pos: u8 @ 0x10; cell run: u8 @ 0x11; cell out: u8 @ 0x12;
cursor c { table @ 0x2000, 0x2001; pos pos; run run; advance pre; on frame step; on note reset;
  match c0 { 0xf0..0xf7 => run(0xf0) then out = c0; 0x80..0xef when c1 == 9 => value(out = c0) take 2; _ => value(out = c0); } }
machine m { step { if mem[0x30] != 0 { mem[0x30] := 0; fire note; } fire frame; } }
schedule { m; }
"""
    mem = {0x2000: [5, 0x81, 9, 0xF2, 7], 0x30: [1]}
    seen = frames(both(text, bytearray(0x10000), 1), mem, 7, (0x10, 0x11, 0x12))
    assert [s[0] for s in seen] == [0, 2, 3, 3, 3, 4, 5]
    assert [s[2] for s in seen] == [5, 0x81, 0, 0, 0, 7, 0]


def test_cursor_declarations_are_checked():
    head = "spec c; cell p: u8 @ 1; cell n: u8 @ 2; cell r: u8 @ 3; cell v: u8 @ 4; "
    cases = [
        ("pos p; table @ 0; match c0 { _ => call(c1); }", "needs a ret"),
        ("pos p; table @ 0; match c0 { _ => run(0); }", "needs a run"),
        ("pos p; table @ 0; match c0 { _ => hold(c1); }", "needs a cnt"),
        (
            "pos p; count n; table @ 0; advance pre; match c0 { _ => hold(c1); }",
            "advance post",
        ),
        ("pos p; table @ 0; on frame settle; match c0 { _ => stop; }", "advance pre"),
        ("pos p; table @ 0; match c0 { _ => value(v.hi = c0); }", "no high byte"),
        ("pos p; table @ 0; match c0 { 0x10 when c3 == 1 => stop; }", "tests one of"),
        ("table @ 0; match c0 { _ => stop; }", "exactly one pos"),
    ]
    for body, message in cases:
        with pytest.raises(SpecError, match=message):
            build(
                head
                + f"cursor t {{ {body} }} machine m {{ step {{ fire frame; }} }} schedule {{ m; }}"
            )


# -- emit and census --------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, binding",
    [
        (CURSOR, {"tab": 0x2000}),
        (ACCUM, {}),
        (CHAIN, {}),
        (VOICES, {"w": 0x100}),
        (BITS, {}),
    ],
)
def test_every_construct_emits_source_that_elaborates_to_the_same_program(
    text, binding
):
    check_roundtrip(text, **binding)


def test_cursors_and_accumulators_stay_objects_through_reorder_and_the_census():
    for text, binding, kinds in (
        (ACCUM, {}, {"accum": 2, "counter": 2}),
        (CURSOR, {"tab": 0x2000}, {"cursor": 2, "counter": 1}),
    ):
        census = O.census(O.reorder(build(text, **binding), O.ORDER))
        assert set(census["objects"]) == set(kinds) and census["statements"]["obj"] > 0


# -- random programs --------------------------------------------------------------------


@st.composite
def cursor_texts(draw):
    kinds = [
        "value(note = c0, inst = c1)",
        "value(note = c0) take 2",
        "hold(c1) then note = c2",
        "sweep(into = pw, count = c1, step = c2)",
        "jump(c1)",
        "call(c1)",
        "return",
        "run(0x80)",
        "end",
        "stop",
    ]
    cuts = sorted(
        draw(st.lists(st.integers(1, 255), min_size=2, max_size=6, unique=True))
    )
    arms, lo = [], 0
    for cut in cuts:
        act = draw(st.sampled_from(kinds))
        if act == "run(0x80)":
            lo = max(lo, 0x80)
            cut = max(cut, lo)
        test = draw(
            st.sampled_from(("", "", " when c1 != 0", " when c2 == 1 && c1 == 2"))
        )
        if lo <= cut - 1 or lo == cut:
            arms.append(f"{lo:#x}..{max(lo, cut - 1):#x}{test} => {act};")
        lo = cut
    arms.append("_ => stop;")
    adv = draw(st.sampled_from(("post", "pre")))
    hold = "count cnt;" if adv == "post" else ""
    arms = [
        a for a in arms if adv == "post" or not any(k in a for k in ("hold(", "sweep("))
    ]
    return f"""
spec r;
cell pos: u8 @ 0x10; cell cnt: u8 @ 0x11; cell ret: u8 @ 0x12; cell run: u8 @ 0x13;
cell note: u8 @ 0x14; cell inst: u8 @ 0x15; cell pw: u16 @ 0x16;
cell acc: u16 @ 0x20; cell dir: u8 @ 0x22; cell beat: u8 @ 0x23;
event turn: tick;
cursor t {{
  table @ 0x2000 width 3; pos pos; {hold} ret ret; run run; advance {adv};
  on frame step; on note reset; wrap -> tick;
  match c0 {{ {' '.join(arms)} }}
}}
accum a: acc {{ step 7; dir dir; on frame triangle; on turn flip; }}
counter b: beat {{ on tick += 1; wrap at 2 -> turn; }}
machine m {{ step {{ fire frame; if note == 3 {{ fire note; }} }} }}
schedule {{ m; }}
"""


@settings(max_examples=150, deadline=None, suppress_health_check=list(HealthCheck))
@given(cursor_texts(), st.binary(min_size=4, max_size=64))
def test_random_cursor_and_accumulator_specs_agree_natively_and_emit_back(text, seed):
    try:
        prog = build(text)
    except SpecError as exc:
        pytest.skip(str(exc)[:60])
    mem = table_memory(seed)
    if native.available():
        run_both(prog, mem, 6)
    check_roundtrip(text)
