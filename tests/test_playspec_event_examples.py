"""The worked examples of docs/playspec.md against the hand-written specs they replace."""

import random
from pathlib import Path

import pytest

from pysidtracker.playspec import SpecRuntimeError, load
from pysidtracker.playspec import outorder as O
from pysidtracker.playspec import outproof
from pysidtracker.playspec import elaborate, parse

from .test_playspec_events import frames

pytestmark = pytest.mark.strain_exempt(
    reason="asserts the values, diagnostics and equalities of the object semantics, which coverage cannot see"
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "events"


@pytest.fixture(autouse=True)
def backend():
    """These tests do not use the 6502 engines the suite is otherwise run on."""
    return None


def example(name, **binding):
    """The example spec ``name`` elaborated against ``binding``."""
    return elaborate(load(EXAMPLES / f"{name}.play"), binding)


def trace(prog, memory, count, watch):
    """The watched bytes after each call, or the error the spec raised."""
    try:
        return frames(prog, memory, count, watch)
    except SpecRuntimeError as exc:
        return type(exc)


HAND_TEMPO = """
spec hand;
layout { w: addr; }
cell tempo: u8 @ L.w;
cell speed: u8 @ L.w + 1;
cell rows: u8 @ L.w + 2;
machine clock {
  step {
    if tempo == 0 { tempo := speed; rows += 1; } else { tempo -= 1; }
  }
}
schedule { clock; }
"""


@pytest.mark.parametrize("speed", [0, 1, 2, 5, 127])
def test_the_tempo_countdown_is_the_hand_written_one(speed):
    binding = {"w": 0x40}
    memory = {0x40: [speed, speed, 0]}
    rewritten = frames(example("tempo", **binding), memory, 300, (0x40, 0x42))
    by_hand = frames(elaborate(parse(HAND_TEMPO), binding), memory, 300, (0x40, 0x42))
    assert rewritten == by_hand
    assert rewritten[-1][1] == 300 // (speed + 1) % 256


HAND_PULSE = """
spec hand;
layout { tab: addr; }
cell tab: u8[256] @ L.tab ro;
cell pw: u16 @ (0x10, 0x11);
cell pos: u8 @ 0x12;
cell cnt: u8 @ 0x13;
machine m {
  step {
    let at = pos;
    let again = true;
    while again bound 256 {
      again := false;
      let tag = tab[at];
      if tag < 0x80 {
        if cnt == tag { cnt := 0; at += 3; pos := at; }
        else { cnt += 1; pw += sx(tab[at + 1]); }
      } elif tag < 0xFE {
        pw := word(tab[at + 1], tag & 0x7F);
        at += 3;
        pos := at;
      } elif tag == 0xFE {
        let target = tab[at + 1];
        if target != at { at := target; again := true; }
      }
    }
    pos := at;
  }
}
schedule { m; }
"""
PULSE = """
spec pulse;
layout { tab: addr; }
cell pw: u16 @ (0x10, 0x11);
cell pos: u8 @ 0x12;
cell cnt: u8 @ 0x13;
walker t {
  table @ L.tab width 3; pos pos; count cnt; values pw;
  on frame step;
  row 0..0x7f: sweep(into=pw, count=c0, step=c1);
  row 0x80..0xfd: set(pw.hi = c0 & 0x7f, pw.lo = c1);
  row 0xfe: jump(c1);
  row 0xff: stop;
}
machine m { step { fire frame; } }
schedule { m; }
"""


def pulse_table(rng):
    """A random pulse table of 3-byte rows."""
    rows = []
    for _ in range(rng.randrange(2, 20)):
        kind = rng.choice(("sweep", "sweep", "set", "jump", "stop"))
        rows += {
            "sweep": lambda: [rng.randrange(0, 5), rng.randrange(256), 0],
            "set": lambda: [rng.randrange(0x80, 0xFE), rng.randrange(256), 0],
            "jump": lambda: [0xFE, 3 * rng.randrange(0, 6), 0],
            "stop": lambda: [0xFF, 0, 0],
        }[kind]()
    return rows + [0xFF, 0, 0]


@pytest.mark.parametrize("seed", range(12))
def test_a_pulse_table_walker_is_the_hand_written_loop(seed):
    rng = random.Random(seed)
    binding = {"tab": 0x2000}
    memory = {0x2000: pulse_table(rng), 0x10: [rng.randrange(256), rng.randrange(256)]}
    watch = (0x10, 0x11, 0x12, 0x13)
    rewritten = trace(elaborate(parse(PULSE), binding), memory, 60, watch)
    by_hand = trace(elaborate(parse(HAND_PULSE), binding), memory, 60, watch)
    assert rewritten == by_hand


HAND_PATTERN = """
spec hand;
layout { w, orderlist, ptn_lo, ptn_hi: addr; }
cell ptn_lo: u8[16] @ L.ptn_lo ro;
cell ptn_hi: u8[16] @ L.ptn_hi ro;
cell ordpos: u8 @ L.w;
cell curptn: u8 @ L.w + 1;
cell ptnpos: u8 @ L.w + 2;
cell gap: u8 @ L.w + 3;
cell curnote: u8 @ L.w + 4;
cell curinst: u8 @ L.w + 5;
cell curfx: u8 @ L.w + 6;
machine player {
  step {
    stream s = word(ptn_lo[curptn], ptn_hi[curptn]) @ ptnpos;
    if gap != 0 { gap -= 1; curnote := 0; }
    else {
      let first = next s;
      if first >= 0xF0 && first < 0xF8 { gap := first - 0xF0; curnote := 0; }
      else {
        curnote := first & 0x7F;
        if neg(first) {
          let inst = next s;
          curinst := inst & 0x7F;
          if neg(inst) { curfx := next s; }
        }
      }
    }
    ptnpos := s.pos;
    if curnote == 0x7F {
      stream o = L.orderlist @ ordpos;
      curptn := next o;
      ordpos := o.pos;
      ptnpos := 0;
    }
  }
}
schedule { player; }
"""


def pattern(rng):
    """A random pattern: rows of one to three bytes and gaps, ending in a ``0x7F`` note."""
    out = []
    for _ in range(rng.randrange(1, 10)):
        kind = rng.choice(("note", "inst", "fx", "gap"))
        note, inst = rng.randrange(1, 0x7F), rng.randrange(0x7F)
        out += {
            "note": [note],
            "inst": [0x80 | note, inst],
            "fx": [0x80 | note, 0x80 | inst, rng.randrange(256)],
            "gap": [0xF0 | rng.randrange(8)],
        }[kind]
    return out + [0x7F]


@pytest.mark.parametrize("seed", range(8))
def test_orderlist_and_pattern_cursors_are_the_hand_written_stream_reads(seed):
    rng = random.Random(seed)
    binding = {"w": 0x40, "orderlist": 0x2800, "ptn_lo": 0x2900, "ptn_hi": 0x2910}
    memory = {0x2800: [rng.randrange(3) for _ in range(40)]}
    for k in range(3):
        at = 0x3000 + 0x40 * k
        memory[at] = pattern(rng)
        memory[0x2900 + k], memory[0x2910 + k] = [at & 0xFF], [at >> 8]
    watch = tuple(range(0x40, 0x47))
    rewritten = frames(example("pattern-cursor", **binding), memory, 120, watch)
    by_hand = frames(elaborate(parse(HAND_PATTERN), binding), memory, 120, watch)
    assert rewritten == by_hand
    assert len({row[1] for row in rewritten}) == 3


def test_the_pulse_and_filter_example_plays_and_is_proved_around_its_walkers():
    binding = {"w": 0x40, "pulse_table": 0x2000, "filter_table": 0x2100}
    prog = example("pulse-filter", **binding)
    memory = {
        0x2000: [2, 0x10, 0, 0x81, 0x20, 0, 0xFF, 0, 0],
        0x2100: [0x95, 0x40, 0, 1, 0x10, 0, 0xFF, 0, 0],
        0x4D: [2],
    }
    watch = (0x40, 0x41, 0x44, 0x45, 0x46, 0x4A)
    seen = frames(prog, memory, 3, watch)
    assert seen[:2] == [
        (0x10, 0, 0, 0x40, 0x10, 14),
        (0x20, 0, 0, 0x40, 0x10, 14),
    ]
    proof = outproof.prove(prog, O.ORDER, timeout_ms=30000)
    assert proof.holds, proof.open
