"""Counters gated by a condition, walkers with carries, holds and guards, last-read cursors,
views and routine arguments, against the hand-written loops they replace."""

import random

import pytest

from pysidtracker.playspec import elaborate, parse
from pysidtracker.playspec import outorder as O
from pysidtracker.playspec.emit import emit, isomorphic
from pysidtracker.playspec.pyexec import run

from .test_playspec_events import frames

pytestmark = pytest.mark.strain_exempt(
    reason="asserts the values and equalities of the object semantics, which coverage cannot see"
)


@pytest.fixture(autouse=True)
def backend():
    """These tests do not use the 6502 engines the suite is otherwise run on."""
    return None


COUNTER = """
spec c;
cell n: u8 @ 0x10;
cell lim: u8 @ 0x11;
cell hits: u8 @ 0x12;
counter beat: n { on frame += 1; reload 1; wrap when n == lim || n == 0 -> row; }
counter rows: hits { on row += 1; }
machine m { step { fire frame; } }
schedule { m; }
"""
HAND_COUNTER = """
spec h;
cell n: u8 @ 0x10;
cell lim: u8 @ 0x11;
cell hits: u8 @ 0x12;
machine m {
  step {
    if n == lim || n == 0 { n := 1; hits += 1; } else { n += 1; }
  }
}
schedule { m; }
"""


@pytest.mark.parametrize("lim", [0, 1, 2, 7, 255])
def test_a_counter_wraps_when_its_condition_holds_before_the_step(lim):
    memory = {0x10: [0, lim, 0]}
    got = frames(elaborate(parse(COUNTER), {}), memory, 600, (0x10, 0x12))
    assert got == frames(elaborate(parse(HAND_COUNTER), {}), memory, 600, (0x10, 0x12))


HAND_TABLE = """
spec hand;
layout { tab: addr; kb: flag; }
cell tab: u8[256] @ L.tab ro;
cell pwlo: u8 @ 0x10;
cell pwhi: u8 @ 0x11;
cell pos: u8 @ 0x12;
cell cnt: u8 @ 0x13;
cell kbd: u8 @ 0x14;
cell carry: u8 @ 0x15;
machine m {
  proc set_pos(y: u8) { pos := y; cnt := 0; }
  step {
    let y = pos;
    let a = tab[y];
    let c = false;
    let advance = false;
    let set = false;
    if !neg(a) {
      y += 1;
      if a == cnt { c := true; advance := true; }
      else {
        cnt += 1;
        let step = tab[y];
        if neg(step) { pwhi -= 1; }
        let lo = pwlo;
        c := carry(step, lo);
        pwlo := step + lo;
        if c { pwhi += 1; }
      }
    } elif a == 0xFE {
      y += 1;
      let target = tab[y];
      c := target >= pos;
      if target != pos {
        y := target;
        a := tab[y];
        if !neg(a) { set_pos(y); } else { set := true; }
      }
    } elif a < 0xFE {
      set := true;
    } else {
      c := true;
    }
    if set {
      pwhi := a & 0x7F;
      y += 1;
      pwlo := tab[y];
      advance := true;
    }
    if advance {
      y += 1;
      if L.kb { kbd := tab[y]; }
      set_pos(y + 1);
    }
    carry := u8(c);
  }
}
schedule { m; }
"""
WALKED = """
spec walked;
layout { tab: addr; kb: flag; }
cell pw: u16 @ (0x10, 0x11);
cell pos: u8 @ 0x12;
cell cnt: u8 @ 0x13;
cell kbd: u8 @ 0x14;
cell carry: u8 @ 0x15;
event pulse: tick;
walker table {
  table @ L.tab width 3;
  pos pos; count cnt; values pw, kbd;
  on pulse step;
  row 0..0x7f: sweep(into=pw, count=c0, step=c1) then kbd = c2 if L.kb;
  row 0x80..0xfd: set(pw.hi = c0 & 0x7f, pw.lo = c1) then kbd = c2 if L.kb, cnt = 0;
  row 0xfe: jump(c1) hold 0..0x7f as 0x80;
  row 0xff: stop;
}
machine m { step { let c = false; react table on pulse() -> c; carry := u8(c); } }
schedule { m; }
"""


def table(rng):
    """A random three-byte-row table of sweeps, sets, jumps and stops."""
    rows = []
    for _ in range(rng.randrange(2, 14)):
        kind = rng.choice(("sweep", "sweep", "set", "jump", "stop"))
        rows += {
            "sweep": lambda: [
                rng.randrange(0, 4),
                rng.randrange(256),
                rng.randrange(256),
            ],
            "set": lambda: [
                rng.randrange(0x80, 0xFE),
                rng.randrange(256),
                rng.randrange(256),
            ],
            "jump": lambda: [
                0xFE,
                3 * rng.randrange(0, 7) + rng.choice((0, 0, 0, 1)),
                0,
            ],
            "stop": lambda: [0xFF, rng.randrange(256), 0],
        }[kind]()
    return rows + [0xFF, 0, 0]


@pytest.mark.parametrize("kb", [0, 1])
@pytest.mark.parametrize("seed", range(40))
def test_a_walker_leaves_the_state_and_carry_of_the_sw_pulse_table_routine(seed, kb):
    rng = random.Random(seed)
    binding = {"tab": 0x2000, "kb": kb}
    memory = {0x2000: table(rng), 0x10: [rng.randrange(256), rng.randrange(256)]}
    memory[0x13] = [rng.randrange(0, 3)]
    watch = (0x10, 0x11, 0x12, 0x13, 0x14, 0x15)
    walked = frames(elaborate(parse(WALKED), binding), memory, 50, watch)
    assert walked == frames(elaborate(parse(HAND_TABLE), binding), memory, 50, watch)


GUARDED = """
spec g;
layout { tab: addr; fine: flag; }
cell lo: u8 @ 0x10;
cell hi: u8 @ 0x11;
cell pos: u8 @ 0x12;
cell cnt: u8 @ 0x13;
cell fsw: u8 @ 0x14;
cell kbd: u8 @ 0x15;
cell word: u16 @ (0x10, 0x11);
walker t {
  table @ L.tab width 3;
  pos pos; count cnt; values word, hi, fsw, kbd;
  on frame step;
  row 0..0x7f if L.fine: sweep(into=word, count=c0, step=c1, frac=3)
    then fsw = c2 & 0x0f when (c2 & 0xf0) == 0x80, kbd = 0 when (c2 & 0xf0) == 0x80, kbd = c2 when (c2 & 0xf0) != 0x80;
  row 0..0x7f if !L.fine: sweep(into=hi, count=c0, step=c1)
    then fsw = c2 & 0x0f when (c2 & 0xf0) == 0x80, kbd = 0 when (c2 & 0xf0) == 0x80, kbd = c2 when (c2 & 0xf0) != 0x80;
  row 0x80..0xff: stop;
}
machine m { step { fire frame; } }
schedule { m; }
"""


@pytest.mark.parametrize("fine", [0, 1])
def test_static_ifs_pick_rows_and_whens_guard_assignments_on_a_row_byte(fine):
    rows = [3, 0xFE, 0x85, 3, 0x40, 0x12, 0xFF, 0, 0]
    memory = {0x2000: rows}
    got = frames(
        elaborate(parse(GUARDED), {"tab": 0x2000, "fine": fine}),
        memory,
        12,
        (0x14, 0x15),
    )
    assert got[3:5] == [(0x05, 0), (0x05, 0)] and got[8][1] == 0x12 and got[8][0] == 5


CURSOR = """
spec cur;
layout { base: addr; }
event advance: tick;
event pattern_end: tick;
cell ptnpos: u8 @ 0x10;
cell packcnt: u8 @ 0x11;
cell note: u8 @ 0x12;
cell ins: u8 @ 0x13;
cell insl: u8 @ 0x14;
cell fx: u8 @ 0x15;
cell val: u8 @ 0x16;
cell ends: u8 @ 0x17;
cursor rd @ L.base pos ptnpos last {
  on row next;
  on pattern_end reset;
  on advance settle;
  run 0x70..0x77 base 0x6F -> packcnt;
  field n -> note = n & 0x7F;
  field i when neg(n) -> ins = i & 0x7F when !neg(i), insl = i & 0x7F when neg(i);
  field f when neg(i) -> fx = f;
  field v when (f & 0xE0) == 0 -> val = v;
}
machine m {
  step {
    note := 0; ins := 0; insl := 0; fx := 0; val := 0;
    fire row;
    if rd.idle {
      if rd.ahead == 0xFF { fire pattern_end; ends += 1; } else { fire advance; }
    }
  }
}
schedule { m; }
"""
HAND_CURSOR = """
spec hand;
layout { base: addr; }
cell data: u8[256] @ L.base ro;
cell ptnpos: u8 @ 0x10;
cell packcnt: u8 @ 0x11;
cell note: u8 @ 0x12;
cell ins: u8 @ 0x13;
cell insl: u8 @ 0x14;
cell fx: u8 @ 0x15;
cell val: u8 @ 0x16;
cell ends: u8 @ 0x17;
machine m {
  step {
    note := 0; ins := 0; insl := 0; fx := 0; val := 0;
    let row = 0;
    let y = 0;
    if packcnt != 0 { packcnt -= 1; }
    else {
      row := data[ptnpos];
      if row >= 0x70 && row < 0x78 { packcnt := row - 0x6F; row := 0; }
    }
    if !neg(row) { note := row; }
    else {
      note := row & 0x7F;
      y += 1;
      let i = data[ptnpos + y];
      if !neg(i) { ins := i; }
      else {
        insl := i & 0x7F;
        y += 1;
        let f = data[ptnpos + y];
        fx := f;
        if f & 0xE0 == 0 { y += 1; val := data[ptnpos + y]; }
      }
    }
    ptnpos += y;
    let at = ptnpos;
    if packcnt == 0 {
      at += 1;
      if data[at] == 0xFF { at := 0; ends += 1; }
      ptnpos := at;
    }
  }
}
schedule { m; }
"""


def pattern(rng):
    """A random pattern of rows of one to four bytes and runs, ending in ``$FF``."""
    out = []
    for _ in range(rng.randrange(1, 12)):
        note, inst, fx = rng.randrange(1, 0x70), rng.randrange(0x80), rng.randrange(256)
        out += rng.choice(
            (
                [note],
                [0x80 | note, inst],
                [0x80 | note, 0x80 | inst, fx, rng.randrange(256)],
                [0x70 | rng.randrange(8)],
            )
        )
    return out + [0xFF]


@pytest.mark.parametrize("seed", range(30))
def test_a_last_read_cursor_with_settle_is_the_sw_row_reader(seed):
    rng = random.Random(seed)
    memory = {0x2000: pattern(rng)}
    watch = tuple(range(0x10, 0x18))
    got = frames(elaborate(parse(CURSOR), {"base": 0x2000}), memory, 60, watch)
    assert got == frames(
        elaborate(parse(HAND_CURSOR), {"base": 0x2000}), memory, 60, watch
    )


VIEWS = """
spec v;
layout { b: addr; }
cell ptr_: u16 @ 0x10;
cell out: u8 @ 0x20;
view inst @ ptr_ { ad @ 3; sr @ 4; }
machine m { proc f(a: u8, c: u8) { out := a + c + inst.ad; } step { out := inst.ad + inst.sr; } }
schedule { m; }
"""


def test_a_view_reads_the_byte_at_its_field_offset_from_a_pointer():
    memory = {0x10: [0x00, 0x30], 0x3000: [0, 0, 0, 7, 9]}
    assert frames(elaborate(parse(VIEWS), {}), memory, 1, (0x20,)) == [(16,)]


def test_a_routine_takes_registers_and_expressions_as_arguments():
    prog = elaborate(parse(VIEWS), {}, routine="m.f(a, inst.sr >> 1)")
    assert "cpu_a" in prog.inputs
    mem = bytearray(0x10000)
    mem[0x10:0x12] = bytes((0, 0x30))
    mem[0x3000:0x3005] = bytes((0, 0, 0, 7, 9))
    run(prog, mem, registers={"a": 5})
    assert mem[0x20] == 5 + 4 + 7


PER_PROC = """
spec p;
layout { lo, hi: addr; }
cell word: u16 @ (L.lo, L.hi) per_proc;
cell out: u8 @ 0x30;
counter c: word { on frame += 1; }
machine m { proc f() { out := lo(word); } step { fire frame; f(); } }
schedule { m; }
"""


def test_a_per_proc_cell_placed_at_two_layout_fields_moves_with_either():
    base = {"lo": 0x40, "hi": 0x41}
    prog = elaborate(parse(PER_PROC), {**base, "lo@c": 0x50, "hi@c": 0x51})
    cell = next(c for c in prog.objects.values() if c.name == "c")
    assert cell.locs["cell"] == (0x50, 0x51)
    other = elaborate(parse(PER_PROC), base)
    assert next(iter(other.objects.values())).locs["cell"] == (0x40, 0x41)


def test_emitted_events_and_gated_counters_elaborate_back():
    prog = elaborate(parse(COUNTER), {})
    result = O.reorder(prog, O.ORDER)
    text = emit(result.program, "gated")
    back = elaborate(parse(text), {})
    assert (
        isomorphic(
            result.program.body,
            O.inline(back.body, back.procs),
            (result.program.objects, back.objects),
        )
        is None
    )
    walked = elaborate(parse(WALKED), {"tab": 0x2000, "kb": 1})
    text = emit(O.reorder(walked, O.ORDER).program, "walked")
    assert "hold 0x0..0x7f as 0x80" in text and "event pulse: tick;" in text
