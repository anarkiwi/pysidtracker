"""Microseconds per call of the Python and the native executor on a spec of event-driven objects.

``python tools/objects_bench.py [--frames N]``: a spec with a counter cascade, a walker, a
cursor, an owner and a record is bound to random tables and played ``N`` calls by
``compile_program`` and by the native call machine; both must write the same memory.
"""

import argparse
import random
import time

from pysidtracker.playspec import compile_program, elaborate, native, parse
from pysidtracker.playspec.pyexec import controls

SPEC = """
spec bench;
layout { sid, w, tab: addr; }
cell chip: u8[25] @ L.sid out;
cell spdc: u8 @ L.w;
cell tempo: u8 @ L.w + 1;
cell beat: u8 @ L.w + 2;
cell wpos: u8 @ L.w + 3;
cell wcnt: u8 @ L.w + 4;
cell pw: u16 @ (L.w + 5, L.w + 6);
cell band: u8 @ L.w + 7;
cell cpos: u8 @ L.w + 8;
cell crun: u8 @ L.w + 9;
cell cnote: u8 @ L.w + 10;
cell cinst: u8 @ L.w + 11;
cell owned: u8 @ L.w + 12;
counter spd: spdc { on frame += 1; wrap at tempo -> tick; }
counter beats: beat { on tick += 1; wrap at 3 -> row; }
walker pwt {
  table @ L.tab width 3; pos wpos; count wcnt; values pw, band;
  on frame step; on note reset;
  row 0..0x3f: sweep(into=pw, count=c0 & 7, step=c1) then band = c2;
  row 0x40..0x7f: delay(count=c0 & 7);
  row 0x80..0xfd: set(pw.hi = c0 & 0x7f, pw.lo = c1, band = c2);
  row 0xfe: jump(c1 & 0x3f);
  row 0xff: stop;
}
cursor ptn @ L.tab + 0x100 pos cpos {
  on row next;
  run 0xF0..0xF7 base 0xF0 -> crun;
  field n -> cnote = n & 0x7f;
  field i when neg(n) -> cinst = i;
}
owner flt: owned { on note claim when cnote != 0; none 0xFF; }
machine voice[x in 0..3] {
  record v @ L.w + 0x20 stride 4 { dur: u8; freq: u16; }
  step {
    dur += 1;
    freq += 0x13;
    chip[7 * x] := lo(freq);
    chip[7 * x + 1] := hi(freq);
    chip[7 * x + 2] := lo(pw);
    chip[7 * x + 3] := hi(pw);
    if flt.mine { chip[21] := band; }
    chip[7 * x + 4] := 0x41;
  }
}
machine clock { step { fire frame; if beat == 1 && cnote != 0 { fire note; } } }
schedule { clock; for v in 0..3 { voice[v]; } }
"""
ENTRY, BANK = 0x1000, 4


def main() -> None:
    """Print the time per call of each executor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=100000)
    args = parser.parse_args()
    program = elaborate(parse(SPEC), {"sid": 0xD400, "w": 0x2000, "tab": 0x3000})
    rng = random.Random(1)
    memory = bytearray(0x10000)
    for addr in range(0x3000, 0x3200):
        memory[addr] = rng.choice(
            (0, 1, 2, 3, 0x41, 0x7F, 0x80, 0x95, 0xF2, 0xFE, 0xFF)
        )
    memory[0x2001] = 5
    python, fast = bytearray(memory), bytearray(memory)
    call, state = compile_program(program), controls(program)
    start = time.perf_counter()
    for _ in range(args.frames):
        call(python.__getitem__, python.__setitem__, state)
    py_us = (time.perf_counter() - start) / args.frames * 1e6
    core, out = native.Core(fast, ENTRY, ENTRY, 10**6), bytearray(25 * args.frames)
    core.add_bank(BANK, False, [], [ENTRY])
    core.select(BANK)
    core.add(BANK, ENTRY, *native.encode(program))
    core.set_cpu((ENTRY, 0xFF, 0, 0, 0, 0x24, 0))
    start = time.perf_counter()
    played, status, _ = core.run(args.frames, out, 0)
    nat_us = (time.perf_counter() - start) / args.frames * 1e6
    assert (played, status) == (args.frames, 0) and fast == python
    print(
        f"objects python {py_us:7.2f} native {nat_us:6.3f} us/call ({py_us / nat_us:5.0f}x)"
    )


if __name__ == "__main__":
    main()
