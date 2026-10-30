"""The playspec DSL: parsing, static checks, both backends, model checking and TV."""

import json
from collections import deque
import re

import pytest
import z3
from hypothesis import given, settings
from hypothesis import strategies as st

from pysidtracker import NTSC_CLOCK_HZ, PAL_CLOCK_HZ, SidFormat, SidParseError, irqhost
from pysidtracker.formats import SpecBinding
from pysidtracker.asm6502 import assemble
from pysidtracker.irqhost import InterruptPlayer
from pysidtracker.testing import whole_song
from pysidtracker.maincli import main as cli_main
from pysidtracker.oracle import EmuPlayer
from pysidtracker.playspec import (
    SpecError,
    SpecPlayer,
    SpecRuntimeError,
    compile_program,
    elaborate,
    load,
    parse,
    run,
)
from pysidtracker.playspec import pyexec, smt
from pysidtracker.playspec.pyexec import source as python_source
from pysidtracker.playspec.tv import TVError, replay, validate
from pysidtracker.playspec.validate import StreamValidator

from .helpers import build_psid

HEADER = "spec t;\nlayout { a, b, c, d: addr; f: flag; n: byte; l: list; }\n"


def lower(body, binding=None):
    """Elaborate ``HEADER`` + ``body`` with ``binding``."""
    return elaborate(parse(HEADER + body), binding)


BIND = {"a": 0x1000, "b": 0x1100, "c": 0x1200, "d": 0x00FB, "f": 1, "n": 3, "l": (2, 0)}


def mem64(**cells):
    """A zeroed 64 KiB memory with ``cells`` (``0x1000=...`` style keys ``m1000``)."""
    mem = bytearray(0x10000)
    for key, value in cells.items():
        addr = int(key[1:], 16)
        mem[addr : addr + len(value)] = bytes(value)
    return mem


def test_syntax_error_is_a_spec_error():
    with pytest.raises(SpecError, match="playspec syntax"):
        parse("spec x; cell")


@pytest.mark.parametrize(
    "body, message",
    [
        ("machine m { step { y := 1; } }\nschedule { m; }", "undefined name 'y'"),
        (
            "cell w: u16 @ L.a;\ncell v: u8 @ L.b;\nmachine m { step { v := w; } }\n"
            "schedule { m; }",
            "width error: u16 value where u8",
        ),
        (
            "cell v: u8 @ L.a;\nmachine m { step { v := 300; } }\nschedule { m; }",
            "300 does not fit",
        ),
        (
            "cell v: u8 @ L.a;\nmachine m { step { if v { } } }\nschedule { m; }",
            "not a truth value",
        ),
        (
            "cell t: u8[4] @ L.a ro;\nmachine m { step { t[1] := 2; } }\nschedule { m; }",
            "read-only",
        ),
        (
            "machine p { signal s: bool = false; step { s := true; } }\n"
            "machine q { step { if p.s { } } }\nschedule { q; p; }",
            "causality: q reads p.s before p runs",
        ),
        (
            "cell k: u8 @ L.a latched;\nmachine p { step { k := 1; } }\n"
            "machine q { step { let v = k; } }\nschedule { p; q; }",
            "latched cell k after p wrote it",
        ),
        (
            "machine p { signal s: bool = false; step { } }\n"
            "machine q { step { p.s := true; } }\nschedule { p; q; }",
            "cannot write p.s",
        ),
        (
            "machine m { proc f() { f(); } step { f(); } }\nschedule { m; }",
            "recursive call of f",
        ),
        ("machine m { step { return; let v = 1; } }\nschedule { m; }", "unreachable"),
        (
            "machine m { step { let v = old(1); } }\nschedule { m; }",
            "old() is only allowed",
        ),
        ("machine m { step { goto s; } }\nschedule { m; }", "goto s: no control state"),
        (
            "machine m { step { let v = L.zz; } }\nschedule { m; }",
            "L.zz is not declared",
        ),
        ("machine m { step { } }\nschedule { m[1]; }", "is not indexed"),
        (
            "machine m { proc g() -> u8 { return 1; } step { let v = g() + 1; } }\nschedule { m; }",
            "returns only as a whole",
        ),
        (
            "machine m[x in 0..2] { cell v: u8 @ L.a; step { } }\nschedule { m[0]; }",
            None,
        ),
        (
            "machine m { step { while true bound 0 { } } }\nschedule { m; }",
            "bound must be positive",
        ),
        (
            "cell v: u8 @ L.a;\nmachine m { step { let q = v / 2; } }\nschedule { m; }",
            "/ needs static",
        ),
    ],
)
def test_static_checks(body, message):
    if message is None:
        with pytest.raises(
            SpecError, match="private cell v of m\\[0\\] and m\\[1\\] share"
        ):
            lower(body, BIND)
        return
    with pytest.raises(SpecError, match=re.escape(message)):
        lower(body)


def test_aliasing_with_a_read_only_table_is_noted():
    program = lower(
        "cell t: u8[4] @ L.a ro;\ncell v: u8 @ L.a + 2;\nmachine m { step { v := 1; } }\n"
        "schedule { m; }",
        BIND,
    )
    assert program.notes == ["shared.v aliases read-only t"]


SEMANTICS = """
const K = 0xF0;
cell w: u16 @ (L.a, L.a + 0x10);
cell arr: u8[8] @ L.b;
cell ptrs: ptr[2] @ (L.c, L.c + 2) ro;
cell out: u8[16] @ L.a + 0x20 out;
machine m {
  control phase { idle, busy = 7 }
  step {
    out[0] := arr[7] + 0x81;
    w += 0x0101;
    out[1] := u8(carry(0xF0, arr[0]));
    out[2] := u8(borrow(arr[1], 0x80, true));
    out[3] := u8(slt(arr[2], 1));
    out[4] := ind(zp(L.d), arr[3]);
    out[5] := mem[L.b + u16(arr[4])];
    out[6] := lo(sx(arr[5])) ^ hi(sx(arr[5]));
    out[7] := rol(arr[6], true);
    out[8] := ror(arr[6], true);
    out[9] := ite(member(2, L.l) && L.f, L.n, 0);
    stream s = ptrs[1] @ 0xFF;
    let first = next s;
    out[10] := first + s[0] + s.pos;
    in phase {
      idle => { goto busy; out[11] := 1; }
      busy => { goto idle; out[11] := 2; }
    }
    let total: u16 = 0;
    for i in 0..4 {
      total += arr[i];
    }
    out[12] := lo(total);
    out[13] := u8(bit(K, 4)) + u8(neg(arr[7]));
    out[14] := u8(cpu.c) + cpu.y;
  }
}
schedule { m; }
"""


def semantics_memory():
    """Memory for :data:`SEMANTICS`."""
    mem = mem64(
        m1000=[0xFF],
        m1010=[0x12],
        m1100=[0x20, 0x7F, 0xFF, 0x05, 0x02, 0x80, 0x81, 0x80],
        m1200=[0, 0x00],
        m1202=[0, 0x30],
        m00fb=[0x00, 0x40],
        m4005=[0x99],
    )
    mem[0x30FF], mem[0x3000] = 0x11, 0x22
    return mem


def test_concrete_semantics_wrap_carry_and_streams():
    program = lower(SEMANTICS, BIND)
    mem = run(program, semantics_memory(), registers={"c": True, "y": 3})
    out = mem[0x1020:0x1030]
    assert mem[0x1000] | mem[0x1010] << 8 == 0x1400
    assert list(out[:15]) == [
        0x01,
        1,
        1,
        1,
        0x99,
        0xFF,
        0x7F,
        0x03,
        0xC0,
        3,
        0x33,
        1,
        0xA3,
        2,
        4,
    ]
    run(program, mem)
    assert mem[0x102B] == 1


def test_hidden_control_state_persists_across_frames():
    program = lower(SEMANTICS, BIND)
    frame = compile_program(program)
    mem = semantics_memory()
    state = [0]
    regs = {"c": False, "y": 0}
    frame(mem.__getitem__, mem.__setitem__, state, None, regs)
    assert (state, mem[0x102B]) == ([7], 1)
    frame(mem.__getitem__, mem.__setitem__, state, None, regs)
    assert (state, mem[0x102B]) == ([0], 2)


def test_returns_and_bounded_loops():
    program = lower(
        "cell v: u8[4] @ L.a;\nmachine m { proc f(k: u8) -> u8 { if k > 2 { return k; } "
        "v[0] += 1; return 9; }\n step { let r = f(v[1]); v[2] := r; "
        "while v[3] != 0 bound 3 { v[3] -= 1; } } }\nschedule { m; }",
        BIND,
    )
    mem = run(program, mem64(m1000=[0, 5, 0, 2]))
    assert list(mem[0x1000:0x1004]) == [0, 5, 5, 0]
    mem = run(program, mem64(m1000=[0, 1, 0, 0]))
    assert list(mem[0x1000:0x1003]) == [1, 1, 9]
    with pytest.raises(SpecRuntimeError, match="ran past 3"):
        run(program, mem64(m1000=[0, 1, 0, 9]))


def test_assert_failure_raises():
    program = lower(
        "cell v: u8 @ L.a;\nmachine m { step { assert v == 0; } }\nschedule { m; }",
        BIND,
    )
    run(program, mem64())
    with pytest.raises(SpecRuntimeError, match="assert at line"):
        run(program, mem64(m1000=[1]))


SCHEDULED = """
cell order: u8[8] @ L.a;
cell k: u8 @ L.b;
machine clock { signal tick: bool = false; step { k += 1; tick := k & 1 != 0; } }
machine v[x in 0..3] {
  cell seen: u8 @ L.c + x;
  step { if clock.tick { seen += 1; } order[x] := k + x; }
}
schedule {
  clock;
  if clock.tick { v[0]; v[1]; v[2]; } else { for i in L.l { v[i]; } }
}
"""


def test_schedule_order_and_signals():
    program = lower(SCHEDULED, BIND)
    assert program.order == ["clock", "v[0]", "v[1]", "v[2]", "v[2]", "v[0]"]
    mem = run(program, mem64(), frames=2)
    assert list(mem[0x1200:0x1203]) == [1, 1, 1]
    assert list(mem[0x1000:0x1003]) == [2, 2, 4]


def test_python_source_has_a_function_per_proc():
    program = lower(SCHEDULED, BIND)
    text = python_source(program)
    assert text.count("def ") == len(program.procs) + 1
    assert (
        "def _p0_clock_step(" in text
        and "\ndef frame(R, W, C, G=None, P=None, N=None):" in text
    )
    with pytest.raises(Exception, match="without a binding"):
        compile_program(lower(SCHEDULED))


@settings(max_examples=25, deadline=None)
@given(st.binary(min_size=16, max_size=16), st.booleans(), st.integers(0, 255))
def test_symbolic_frame_equals_concrete(data, carry_in, y):
    program = lower(SEMANTICS, BIND)
    mem = semantics_memory()
    mem[0x1100:0x1108] = data[:8]
    mem[0x1000], mem[0x1010] = data[8], data[9]
    mem[0x4000 + data[3]] = data[10]
    expected = run(program, bytearray(mem), registers={"c": carry_in, "y": y})
    env = {n: z3.BitVecVal(d, w) for n, w, d in program.controls}
    env.update(cpu_c=z3.BoolVal(carry_in), cpu_y=z3.BitVecVal(y, 8))
    state = smt.step(program, smt.concrete_memory(bytes(mem)), env)
    for addr in list(range(0x1020, 0x102F)) + [0x1000, 0x1010]:
        assert (
            z3.simplify(smt.select(state.mem, z3.BitVecVal(addr, 16))).as_long()
            == expected[addr]
        )


COUNTER = """
cell n: u8 @ L.a;
cell top: u8 @ L.b;
machine m { step { if n < top { n += 1; } else { n := 0; } } }
schedule { m; }
invariant bounded: n <= top || n == 0;
invariant steps: n == old(n) + 1 || n == 0;
"""


def test_bmc_proves_and_refutes():
    program = lower(COUNTER, BIND)
    assert smt.bmc(program, 3).holds
    broken = lower(COUNTER + "invariant small: n < 5;\n", BIND)
    result = smt.bmc(broken, 3)
    assert not result.holds and result.what == "invariant small"
    assert result.model["n"] >= 4 or result.frame > 0


def test_bmc_reports_loop_bounds_and_assumes():
    spec = (
        "cell v: u8 @ L.a;\nmachine m { step { while v != 0 bound 2 { v -= 1; } } }\n"
        "schedule { m; }\n"
    )
    result = smt.bmc(lower(spec, BIND), 1)
    assert not result.holds and "runs past its bound" in result.what
    assert smt.bmc(lower(spec + "assume v < 3;\n", BIND), 1).holds


def test_transition_relation_is_functional():
    program = lower(COUNTER, BIND)
    pre, post, rel = smt.transition(program)
    solver = z3.Solver()
    solver.add(rel, z3.Select(pre, 0x1000) == 2, z3.Select(pre, 0x1100) == 5)
    solver.add(z3.Select(post, 0x1000) != 3)
    assert solver.check() == z3.unsat


ROUTINE = """
  DEC $1000,X
  BPL skip
reload:
  LDA #$04
  STA $1000,X
  LDY $1010,X
  LDA ($FB),Y
  CLC
  ADC $1020,X
  STA $1020,X
  BCC skip
  INC $1030,X
skip:
  JSR out
  RTS
out:
  LDA $1020
  STA $D400
  RTS
"""

TV_SPEC = """
spec tv;
layout { cnt, idx, acc, hi, reload, sid: addr; }
cell reload: u8 @ L.reload operand;
cell acc0: u8 @ L.acc;
cell sid: u8[32] @ L.sid out;
machine v[x in 0..2] {
  cell cnt: u8 @ L.cnt + x;
  cell idx: u8 @ L.idx + x;
  cell acc: u8 @ L.acc + x;
  cell hi: u8 @ L.hi + x;
  proc tick() {
    cnt -= 1;
    if neg(cnt) {
      cnt := reload;
      let b = ind(0xFB, idx);
      let c = carry(acc, b);
      acc += b;
      if c { hi += 1; }
    }
    sid[0] := acc0;
  }
  step { tick(); }
}
schedule { v[0]; v[1]; }
"""


def _routine():
    code, labels = assemble(ROUTINE, 0x2000)
    image = bytearray(0x10000)
    image[0x2000 : 0x2000 + len(code)] = code
    return bytes(image), labels


TV_BIND = {"cnt": 0x1000, "idx": 0x1010, "acc": 0x1020, "hi": 0x1030, "sid": 0xD400}


def test_translation_validation_proves_a_routine():
    image, labels = _routine()
    binding = dict(TV_BIND, reload=labels["reload"] + 1)
    for x in (0, 1):
        result = validate(
            parse(TV_SPEC),
            binding,
            f"v[{x}].tick",
            image,
            0x2000,
            registers={"X": x},
            preserve=("X",),
        )
        assert result.proven, result.counterexample
        assert result.paths >= 1


def test_translation_validation_counterexample_replays():
    image, labels = _routine()
    binding = dict(TV_BIND, reload=labels["reload"] + 1)
    broken = parse(TV_SPEC.replace("if c { hi += 1; }", ""))
    result = validate(broken, binding, "v[0].tick", image, 0x2000, registers={"X": 0})
    assert not result.proven
    cex = result.counterexample
    assert cex["address"] == 0x1030 and cex["code"] == (cex["spec"] + 1) & 0xFF
    diffs = replay(broken, binding, "v[0].tick", cex["memory"], 0x2000, {"X": 0})
    assert diffs == [0x1030]


def test_translation_validation_reads_operand_cells_live():
    image, labels = _routine()
    binding = dict(TV_BIND, reload=labels["reload"] + 1)
    constant = TV_SPEC.replace("cnt := reload;", "cnt := 4;")
    frozen = constant.replace(
        "cell reload: u8 @ L.reload operand;", "cell reload: u8 @ L.reload;"
    )
    live = validate(
        parse(constant), binding, "v[0].tick", image, 0x2000, registers={"X": 0}
    )
    assert not live.proven and live.counterexample["address"] == 0x1000
    assert validate(
        parse(frozen), binding, "v[0].tick", image, 0x2000, registers={"X": 0}
    ).proven


PLAY_SPEC = """
spec counter;
layout { freq, sid: addr; }
cell sid: u8[32] @ L.sid out;
machine m {
  cell freq: u8 @ L.freq;
  step { freq += 3; sid[0] := freq; sid[4] := 0x41; }
}
schedule { m; }
"""


def _tune(wrapped):
    code, labels = assemble(
        """
init:
  lda #$10
  sta $1300
  rts
play:
  lda $1300
  clc
  adc #$03
  sta $1300
  sta $d400
  lda #$41
  sta $d404
  rts
wrap:
  jsr play
  rts
""",
        0x1000,
    )
    play = labels["wrap" if wrapped else "play"]
    return build_psid(code, 0x1000, init=labels["init"], play=play), labels


@pytest.mark.parametrize("wrapped", [False, True])
def test_spec_player_matches_the_6502(wrapped):
    data, labels = _tune(wrapped)
    image = data[0x7C:]
    play = labels["wrap" if wrapped else "play"]
    spec = parse(PLAY_SPEC)
    player = SpecPlayer(
        spec,
        image,
        0x1000,
        labels["init"],
        play,
        bind=lambda v, e: {"freq": 0x1300, "sid": 0xD400},
        entries=lambda view: {labels["play"]},
        validate=True,
    )
    reference = EmuPlayer(image, 0x1000, labels["init"], play)
    frames = whole_song(40)
    assert player.render_grid(frames) == reference.render_grid(frames)
    assert player.entry == labels["play"]
    assert player.program().name == "counter"
    assert len(player.view()) == 0x10000


def test_unreached_first_call_falls_back_to_the_lowest_entry():
    data, labels = _tune(False)
    image = data[0x7C:]

    def make(entries):
        return SpecPlayer(
            parse(PLAY_SPEC),
            image,
            0x1000,
            labels["init"],
            labels["init"],  # a play call that never enters the modelled routine
            bind=lambda v, e: {"freq": 0x1300, "sid": 0xD400},
            entries=entries,
        )

    assert make(lambda view: {labels["play"], labels["wrap"]}).entry == labels["play"]
    with pytest.raises(SidParseError, match="no modelled entry"):
        make(lambda view: set())


def test_stream_validator_reports_malformed_streams():
    spec = parse(
        "spec g;\nlayout { p: addr; }\ncell q: u8 @ L.p;\nmachine m { step { "
        'stream s = 0x2000 @ 0 grammar "tests.test_bytegrammar:TOY:seq"; let v = next s; q := v; '
        'stream t = 0x3000 @ 0 grammar "tests.test_bytegrammar:TOY:seq"; } }\nschedule { m; }'
    )
    program = elaborate(spec, {"p": 0x1000})
    mem = mem64(m2000=[1, 2, 0xFF])
    validator = StreamValidator()
    frame = compile_program(program)
    frame(
        mem.__getitem__, mem.__setitem__, [], validator.hook(program, mem.__getitem__)
    )
    assert list(validator.failures()) == [("tests.test_bytegrammar:TOY:seq", 0x3000)]
    strict = StreamValidator(strict=True)
    with pytest.raises(Exception, match="is not tests.test_bytegrammar"):
        strict.check("tests.test_bytegrammar:TOY:seq", 0x3000, mem.__getitem__)


def test_load_reads_a_spec_file(tmp_path):
    path = tmp_path / "s.play"
    path.write_text(PLAY_SPEC)
    assert load(path).name == "counter" and load(str(path)).machines


def test_cli_check_python_bmc(tmp_path, capsys):
    path = tmp_path / "c.play"
    path.write_text(HEADER + COUNTER)
    binding = tmp_path / "b.json"
    binding.write_text(json.dumps(BIND))
    cli_main(["playspec", "check", str(path), "--binding", str(binding)])
    cli_main(["playspec", "python", str(path), "--binding", str(binding)])
    cli_main(["playspec", "bmc", str(path), "--binding", str(binding), "--frames", "2"])
    out = capsys.readouterr().out
    assert (
        "ok: t; order m" in out and "def frame" in out and "holds for 2 frames" in out
    )
    path.write_text(HEADER + COUNTER + "invariant small: n < 5;\n")
    cli_main(["playspec", "bmc", str(path), "--binding", str(binding), "--frames", "2"])
    assert "invariant small fails" in capsys.readouterr().out


def test_cli_run_and_tv(tmp_path, capsys):
    data, labels = _tune(False)
    tune = tmp_path / "t.sid"
    tune.write_bytes(data)
    spec = tmp_path / "s.play"
    spec.write_text(PLAY_SPEC)
    binder = "tests.test_playspec:cli_binding"
    cli_main(
        ["playspec", "run", str(spec), str(tune), "--binder", binder, "--frames", "5"]
    )
    assert "5 frames, grid digest" in capsys.readouterr().out
    cli_main(
        [
            "playspec",
            "tv",
            str(spec),
            str(tune),
            "--binder",
            binder,
            "--routine",
            "m.step",
            "--pc",
            f"{labels['play']:x}",
            "--reg",
            "X=0",
            "--preserve",
            "X",
        ]
    )
    assert "m.step vs $1006: proven" in capsys.readouterr().out


def cli_binding(view, entry):
    """The binding the CLI tests pass by module:attr."""
    del view, entry
    return {"freq": 0x1300, "sid": 0xD400}


def test_stream_grammar_rule_follows_layout():
    spec = parse(
        "spec g;\nlayout { p: addr; ext: flag; }\ncell q: u8 @ L.p;\nmachine m { step { "
        'stream s = 0x2000 @ 0 grammar "tests.test_bytegrammar:TOY:ext" if L.ext, '
        '"tests.test_bytegrammar:TOY:seq"; let v = next s; q := v; } }\nschedule { m; }'
    )
    for ext, rule in ((1, "ext"), (0, "seq")):
        program = elaborate(spec, {"p": 0x1000, "ext": ext})
        assert program.grammars == [
            (f"tests.test_bytegrammar:TOY:{rule}", program.grammars[0][1])
        ]


NATIVE_SPEC = """
spec hooked;
layout { freq, sid, hook, tail: addr; }
cell sid: u8[32] @ L.sid out;
machine m {
  cell freq: u8 @ L.freq;
  step {
    let v = freq;
    native L.hook(a = v) resumes L.tail;
    freq += 3; sid[0] := freq; sid[4] := 0x41;
  }
}
schedule { m; }
"""


@pytest.mark.parametrize("wrapped", [False, True])
def test_native_hook_resumes_or_hands_the_call_to_the_cpu(wrapped):
    code, labels = assemble(
        """
init:
  lda #$10
  sta $1300
  rts
play:
  lda $1300
  jmp hook
tail:
  clc
  adc #$03
  sta $1300
  sta $d400
  lda #$41
  sta $d404
  rts
hook:
  cmp #$40
  beq away
  jmp tail
away:
  lda #$00
  sta $1300
  sta $d400
  rts
wrap:
  jsr play
  rts
""",
        0x1000,
    )
    play = labels["wrap" if wrapped else "play"]
    image = build_psid(code, 0x1000, init=labels["init"], play=play)[0x7C:]
    binding = {
        "freq": 0x1300,
        "sid": 0xD400,
        "hook": labels["hook"],
        "tail": labels["tail"],
    }
    player = SpecPlayer(
        parse(NATIVE_SPEC),
        image,
        0x1000,
        labels["init"],
        play,
        bind=lambda v, e: binding,
        entries=lambda view: {labels["play"]},
    )
    reference = EmuPlayer(image, 0x1000, labels["init"], play)
    frames = whole_song(60)
    assert player.render_grid(frames) == reference.render_grid(frames)


HOSTED = """
init: SEI
  LDA #$7F
  STA $DC0D
  LDA $DC0D
  LDA #$10
  STA $1300
  LDA #<irq
  STA $0314
  LDA #>irq
  STA $0315
  LDA #$1B
  STA $D011
  LDA #$80
  STA $D012
  LDA #$01
  STA $D01A
  CLI
  RTS
irq: INC $D019
  JSR play
  JMP $EA31
play: LDA $1300
  CLC
  ADC #$03
  STA $1300
  STA $D400
  LDA #$41
  STA $D404
  RTS
"""


def _hosted_tune():
    code, labels = assemble(HOSTED, 0x1000)
    data = build_psid(code, 0x1000, init=labels["init"], play=0, magic=b"RSID")
    return data, labels, code


@pytest.mark.parametrize("container", [True, False])
def test_spec_player_traps_entries_under_the_interrupt_host(container, monkeypatch):
    monkeypatch.setattr(irqhost, "poweron_writes", lambda: None)
    data, labels, code = _hosted_tune()
    player = SpecPlayer(
        parse(PLAY_SPEC),
        code,
        0x1000,
        labels["init"],
        0,
        bind=lambda v, e: {"freq": 0x1300, "sid": 0xD400},
        entries=lambda view: {labels["play"]},
        psid=False,
        sid=data if container else None,
    )
    reference = InterruptPlayer(data)
    frames = whole_song(40)
    grid = player.render_grid(frames)
    assert grid == reference.render_grid(frames)
    assert len({row[0] for row in grid}) > 20
    assert player.entry == labels["play"]
    assert player.program().name == "counter"
    assert len(player.view()) == len(player.memory) == 0x10000
    more = SpecPlayer(
        parse(PLAY_SPEC),
        code,
        0x1000,
        labels["init"],
        0,
        bind=lambda v, e: {"freq": 0x1300, "sid": 0xD400},
        entries=lambda view: {labels["play"]},
        psid=False,
        sid=data,
    )
    assert more.play_frame() and more.play_frame()
    assert more.snapshot()[0] == more.regs[0]


def test_hosted_spec_replaces_the_trapped_entry(monkeypatch):
    monkeypatch.setattr(irqhost, "poweron_writes", lambda: None)
    _, labels, code = _hosted_tune()
    player = SpecPlayer(
        parse(PLAY_SPEC.replace("freq += 3", "freq += 5")),
        code,
        0x1000,
        labels["init"],
        0,
        bind=lambda v, e: {"freq": 0x1300, "sid": 0xD400},
        entries=lambda view: {labels["play"]},
        psid=False,
    )
    grid = player.render_grid(whole_song(10))
    steps = {(b[0] - a[0]) & 0xFF for a, b in zip(grid[1:], grid[2:])}
    assert steps == {5}


def test_tv_counted_loop_with_unconditional_back_edge():
    routine = """
shift: LDY $1000
loop:  DEY
       BMI done
       LSR $1011
       ROR $1010
       JMP loop
done:  RTS
"""
    spec = """
spec t;
layout { cnt, val: addr; }
cell count: u8 @ L.cnt;
cell low: u8 @ L.val;
cell high: u8 @ L.val + 1;
machine m {
  proc shift() {
    let k = count - 1;
    while !neg(k) bound 128 {
      let w = word(low, high) >> 1;
      low := lo(w);
      high := hi(w);
      k -= 1;
    }
  }
  step { shift(); }
}
schedule { m; }
"""
    code, _ = assemble(routine, 0x2000)
    image = bytearray(0x10000)
    image[0x2000 : 0x2000 + len(code)] = code
    binding = {"cnt": 0x1000, "val": 0x1010}
    assert validate(parse(spec), binding, "m.shift", bytes(image), 0x2000)


STRETCH_SPEC = """spec stretch;
layout { src, dst: addr; }
cell src: u8 @ L.src;
cell dst: u8 @ L.dst;
machine m[x in 0..1] {
  step { copy(); }
  proc copy() { dst := src; }
}
schedule { m[0]; }
"""


def test_translation_validation_of_a_stretch():
    image = bytearray(0x10000)
    code = bytes.fromhex("A5208521A5228523E62460")
    image[0x1000 : 0x1000 + len(code)] = code
    binding = {"src": 0x20, "dst": 0x21}
    args = (parse(STRETCH_SPEC), binding, "m[0].copy", bytes(image), 0x1000)
    assert not validate(*args).proven
    assert validate(*args, stop=0x1004).proven
    with pytest.raises(TVError):
        validate(*args, stop=0x1100)


PUSH_SPEC = """
spec pushy;
layout { out, idx: addr; }
cell out: u8 @ L.out;
cell idx: u8 @ L.idx;
machine m {
  proc go() { out := ind(0xFB, idx); }
  step { go(); }
}
"""


def test_translation_validation_pha_push_is_not_aliased_by_reads():
    code, _ = assemble("PHA\nLDY $1010\nLDA ($FB),Y\nSTA $1000\nPLA\nRTS", 0x2000)
    image = bytearray(0x10000)
    image[0x2000 : 0x2000 + len(code)] = code
    binding = {"out": 0x1000, "idx": 0x1010}
    result = validate(parse(PUSH_SPEC), binding, "m.go", bytes(image), 0x2000)
    assert result.proven, result.counterexample


def test_translation_validation_assume_constrains_initial_memory():
    image, labels = _routine()
    binding = dict(TV_BIND, reload=labels["reload"] + 1)
    constant = TV_SPEC.replace("cnt := reload;", "cnt := 4;")
    args = (parse(constant), binding, "v[0].tick", image, 0x2000)
    assert not validate(*args, registers={"X": 0}).proven
    cell = labels["reload"] + 1
    assumed = validate(
        *args, registers={"X": 0}, assume=[lambda mem, regs: z3.Select(mem, cell) == 4]
    )
    assert assumed.proven, assumed.counterexample


_PHA_PLA = """
split: LDA $1000
       PHA
       AND #$0F
       STA $1010
       PLA
       AND #$F0
       STA $1011
       RTS
"""
_PHA_PLA_SPEC = """
spec t;
layout {{ src, dst: addr; }}
cell a: u8 @ L.src;
cell low: u8 @ L.dst;
cell high: u8 @ L.dst + 1;
machine m {{
  proc split() {{ low := a & 0x0F; high := {high}; }}
  step {{ split(); }}
}}
schedule {{ m; }}
"""


@pytest.mark.parametrize("high,proven", [("a & 0xF0", True), ("(a & 0xF0) ^ 1", False)])
def test_pha_pla_routine_is_not_vacuously_proven(high, proven):
    code, _ = assemble(_PHA_PLA, 0x2000)
    image = bytearray(0x10000)
    image[0x2000 : 0x2000 + len(code)] = code
    spec = parse(_PHA_PLA_SPEC.format(high=high))
    binding = {"src": 0x1000, "dst": 0x1010}
    result = validate(spec, binding, "m.split", bytes(image), 0x2000)
    assert result.proven is proven


SPLIT = """
cell p: u8 @ L.a;
cell q: u8 @ L.b;
cell r: u8[4] @ L.c;
machine m {
  step {
    let t = p;
    let u = q + 1;
    for i in 0..4 { r[i] := t + i; t += u; }
    while u != 0 bound 20 { u -= 1; q := q + t; }
    if p == 7 { q := 99; finish; }
    r[0] := t;
    q := q + u;
  }
}
schedule { m; }
"""


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 255), st.integers(0, 15), st.integers(1, 3))
def test_split_frame_matches_unsplit(p, q, chunk):
    program = lower(SPLIT, BIND)
    mem = mem64(m1000=[p], m1100=[q])
    whole = run(program, bytearray(mem))
    saved, pyexec.CHUNK = pyexec.CHUNK, chunk
    try:
        text = python_source(program)
        assert "def _f0(" in text and "V[0]" in text
        assert run(program, bytearray(mem)) == whole
    finally:
        pyexec.CHUNK = saved


def test_split_frame_finish_skips_the_rest(monkeypatch):
    monkeypatch.setattr(pyexec, "CHUNK", 1)
    out = run(lower(SPLIT, BIND), mem64(m1000=[7], m1100=[0]))
    assert (out[0x1100], out[0x1200]) == (99, 7)


CALLS = """
cell p: u8 @ L.a;
cell q: u8 @ L.b;
cell r: u8 @ L.c;
machine m {
  proc pick(n: u8) -> u8 { if n == 1 { return 5; } if n == 2 { finish; } }
  proc twice() { r += 1; }
  step {
    let a = pick(p);
    twice();
    let b = pick(q);
    twice();
    r := r + a + b;
  }
}
schedule { m; }
"""


@pytest.mark.parametrize(
    "p, q, expected",
    [(1, 0, 2 + 5), (0, 1, 2 + 5), (1, 1, 2 + 10), (0, 0, 2), (2, 1, 0), (1, 2, 1)],
)
def test_a_proc_lowered_once_serves_every_call(p, q, expected):
    program = lower(CALLS, BIND)
    assert len(program.procs) == 3
    mem = mem64(m1000=[p], m1100=[q])
    assert run(program, mem)[0x1200] == expected


VOLATILE = """spec vol;
layout { osc, pw: addr; }
cell osc3: u8 @ L.osc volatile;
cell pw: u8 @ L.pw out;
cell tmp: u8 @ 0x10 scratch;
machine m { step { %s } }
schedule { m; }
"""
VOLATILE_BINDING = {"osc": 0xD41B, "pw": 0xD402}


def _prove_volatile(step: str, source: str):
    code, _ = assemble(source, 0x2000)
    image = bytearray(0x10000)
    image[0x2000 : 0x2000 + len(code)] = code
    return validate(
        parse(VOLATILE % step), VOLATILE_BINDING, "m.step", bytes(image), 0x2000
    )


def test_a_volatile_read_matches_one_6502_read():
    result = _prove_volatile(
        "pw := osc3 & 0x0F;", "lda $d41b\nand #$0f\nsta $d402\nrts"
    )
    assert result.proven


@pytest.mark.parametrize(
    "step, code",
    [
        ("pw := osc3 + osc3;", "lda $d41b\nand #$0f\nsta $d402\nrts"),
        ("pw := osc3 & 0x0F;", "lda $d41b\nlda $d41b\nand #$0f\nsta $d402\nrts"),
        ("pw := osc3 & 0x0F;", "lda $d41c\nand #$0f\nsta $d402\nrts"),
    ],
)
def test_volatile_reads_that_differ_in_number_or_address_are_refuted(step, code):
    assert not _prove_volatile(step, code).proven


def test_a_volatile_read_is_not_shared_between_uses():
    both = "lda $d41b\nsta $10\nlda $d41b\nclc\nadc $10\nsta $d402\nrts"
    assert _prove_volatile("pw := osc3 + osc3;", both).proven


def test_a_volatile_read_on_one_branch_counts_only_where_it_runs():
    step = "if osc3 & 1 != 0 { pw := osc3; }"
    code = "lda $d41b\nand #1\nbeq done\nlda $d41b\nsta $d402\ndone: rts"
    assert _prove_volatile(step, code).proven


def test_a_store_to_a_volatile_cell_is_refused():
    text = VOLATILE % "osc3 := 1;"
    with pytest.raises(SpecError, match="write to volatile cell"):
        elaborate(parse(text), VOLATILE_BINDING)


def test_each_volatile_read_is_a_separate_bus_read():
    reads = []
    program = elaborate(
        parse(VOLATILE % "pw := osc3 + osc3; pw := pw + osc3;"), VOLATILE_BINDING
    )
    frame = compile_program(program)
    ram = bytearray(0x10000)

    def read(addr):
        reads.append(addr)
        return ram[addr]

    frame(read, ram.__setitem__, [], None, {})
    assert reads.count(0xD41B) == 3


def test_spec_player_replays_the_6502_volatile_reads():
    code, labels = assemble(
        """init: lda #$ff
        sta $d40e
        sta $d40f
        lda #$22
        sta $d413
        lda #$f0
        sta $d414
        lda #$81
        sta $d412
        rts
play:   lda $d41b
        and #$0f
        sta $d402
        lda $d41c
        sta $d403
        rts""",
        0x1000,
    )
    spec = parse(
        "spec o;\nlayout { osc, env, pw, pw2: addr; }\n"
        "cell osc3: u8 @ L.osc volatile;\ncell env3: u8 @ L.env volatile;\n"
        "cell pw: u8 @ L.pw out;\ncell pw2: u8 @ L.pw2 out;\n"
        "machine m { step { pw := osc3 & 0x0F; pw2 := env3; } }\nschedule { m; }"
    )
    fed = deque()
    player = SpecPlayer(
        spec,
        code,
        0x1000,
        labels["init"],
        labels["play"],
        bind=lambda v, e: {"osc": 0xD41B, "env": 0xD41C, "pw": 0xD402, "pw2": 0xD403},
        entries=lambda view: {labels["play"]},
        inputs=lambda addr: fed.popleft(),
    )
    reference = EmuPlayer(
        code, 0x1000, labels["init"], labels["play"], record_reads=(0xD41B, 0xD41C)
    )
    seen = set()
    for _ in range(60):
        reference.play_frame()
        assert [a for a, _v in reference.reads] == [0xD41B, 0xD41C]
        fed.extend(v for _a, v in reference.reads)
        seen.update(v for _a, v in reference.reads)
        player.play_frame()
        assert player.snapshot() == reference.snapshot()
    assert not fed
    assert len(seen) > 8


def test_parser_survives_an_unwritable_table_cache(monkeypatch):
    from pysidtracker.playspec import syntax

    real = syntax.Lark

    def lark(text, **options):
        if options.get("cache"):
            raise OSError("read-only cache directory")
        return real(text, **options)

    monkeypatch.setattr(syntax, "Lark", lark)
    syntax._parser.cache_clear()
    try:
        assert syntax._parser().parse("spec x;") is not None
    finally:
        syntax._parser.cache_clear()


def _hosted_format(sid):
    _, labels, code = _hosted_tune()
    binding = SpecBinding(
        parse(PLAY_SPEC),
        lambda v, e: {"freq": 0x1300, "sid": 0xD400},
        code,
        entries=lambda view: {labels["play"]},
        psid=False,
        sid=sid,
    )
    model = type(
        "Model",
        (),
        dict(
            name="n",
            author="a",
            released="r",
            load_addr=0x1000,
            init_addr=labels["init"],
            play_addr=0,
        ),
    )
    return SidFormat("hosted", None, None, spec=lambda m: binding), model()


def test_spec_player_runs_a_play_address_0_tune_on_its_real_header(monkeypatch):
    monkeypatch.setattr(irqhost, "poweron_writes", lambda: None)
    data, labels, code = _hosted_tune()
    fmt, model = _hosted_format(data)
    by_hand = SpecPlayer(
        parse(PLAY_SPEC),
        code,
        0x1000,
        labels["init"],
        0,
        bind=lambda v, e: {"freq": 0x1300, "sid": 0xD400},
        entries=lambda view: {labels["play"]},
        psid=False,
        sid=data,
        clock="PAL",
    )
    frames = whole_song(40)
    got = fmt.spec_player(model).render_grid(frames)
    assert (
        got == by_hand.render_grid(frames) == InterruptPlayer(data).render_grid(frames)
    )
    assert len({row[0] for row in got}) > 20


def test_spec_player_clock_follows_the_tune_header(monkeypatch):
    monkeypatch.setattr(irqhost, "poweron_writes", lambda: None)
    code, labels = assemble(HOSTED, 0x1000)
    kwargs = dict(init=labels["init"], play=0, magic=b"RSID")
    pal = build_psid(code, 0x1000, **kwargs)
    ntsc = build_psid(code, 0x1000, flags=0b1000, **kwargs)

    def hz(data, **overrides):
        fmt, model = _hosted_format(data)
        player = fmt.spec_player(model, **overrides)
        player.render_grid(whole_song(3))
        return player._hosted.host.clock_hz  # pylint: disable=protected-access

    assert hz(pal) == PAL_CLOCK_HZ and hz(ntsc) == NTSC_CLOCK_HZ
    assert hz(ntsc, clock="PAL") == PAL_CLOCK_HZ
