"""``unnative``: a spec's native hooks replaced by spec statements, same output."""

import pytest
import z3

from pysidtracker.asm6502 import assemble
from pysidtracker.errors import SidError
from pysidtracker.maincli import main as cli_main
from pysidtracker.playspec import SpecPlayer, elaborate, parse, run
from pysidtracker.playspec import smt
from pysidtracker.playspec.unnative import count_natives, equivalent, unnative

from .helpers import build_psid
from .test_playspec import NATIVE_SPEC

BASE = 0x1000


def tune(source, wrapped=False):
    """``(psid, labels, play address)`` of the assembled ``source``."""
    code, labels = assemble(source, BASE)
    play = labels["wrap" if wrapped else "play"]
    return build_psid(code, BASE, init=labels["init"], play=play), labels, play


def players(source, binding, wrapped=False):
    """``(image, make(spec))`` playing ``source`` under a spec."""
    psid, labels, play = tune(source, wrapped)
    image = psid[0x7C:]
    binding = dict(binding, **{k: labels[v] for k, v in binding.items() if v in labels})

    def make(spec):
        return SpecPlayer(
            spec,
            image,
            BASE,
            labels["init"],
            play,
            bind=lambda v, e: binding,
            entries=lambda view: {labels["play"]},
        )

    return make, binding, labels


HOOK = """
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
"""


@pytest.mark.parametrize("wrapped", [False, True])
def test_hook_replaced_by_branch_and_finish(wrapped):
    make, binding, _ = players(
        HOOK,
        {"freq": 0x1300, "sid": 0xD400, "hook": "hook", "tail": "tail"},
        wrapped,
    )
    old = parse(NATIVE_SPEC)
    new = unnative(old, binding, make(old).view())
    assert count_natives(new.machines) == 0
    assert "native" not in new.source and "finish;" in new.source
    equivalent(make, old, new, 60)
    elaborate(new, binding)  # a native-free spec also checks and bounded-checks
    smt.transition(elaborate(new, binding))


LOOP_SPEC = """
spec looped;
layout { buf, out, hook, tail: addr; }
machine m {
  cell n: u8 @ L.buf;
  cell last: u8 @ L.out;
  step {
    native L.hook(x = n, y = 2) resumes L.tail;
    last += 1;
  }
}
schedule { m; }
"""

LOOP = """
init:
  lda #$03
  sta $1300
  rts
play:
  jmp hook
tail:
  rts
hook:
  lda #$00
  sta $1400,x
  inc $1301,x
  txa
  clc
  adc #$05
  sta $1500
  lda $1302
  and #$0f
  beq skip
  lda #$99
  sta $1500,x
skip:
  jmp tail
"""


def test_indexed_stores_and_reads_match():
    make, binding, _ = players(
        LOOP, {"buf": 0x1300, "out": 0x1600, "hook": "hook", "tail": "tail"}
    )
    old = parse(LOOP_SPEC)
    new = unnative(old, binding, make(old).view())
    assert "native" not in new.source
    equivalent(make, old, new, 40)


def test_a_hook_that_always_leaves_the_call_has_no_branch():
    spec = LOOP_SPEC.replace("last += 1;", "last += 1; finish;")
    src = LOOP.replace("jmp tail", "rts")
    make, binding, _ = players(
        src, {"buf": 0x1300, "out": 0x1600, "hook": "hook", "tail": "tail"}
    )
    old = parse(spec)
    new = unnative(old, binding, make(old).view())
    assert "if " not in new.source.split("step")[1].split("last")[0]
    equivalent(make, old, new, 10)


def test_finish_ends_the_call_in_every_backend():
    text = (
        "spec f;\nlayout { a, b: addr; }\ncell p: u8 @ L.a;\ncell q: u8 @ L.b;\n"
        "machine m { step { p := 1; if p == 1 { finish; } q := 2; } }\n"
        "machine n { step { q := 3; } }\nschedule { m; n; }"
    )
    program = elaborate(parse(text), {"a": 0x10, "b": 0x11})
    mem = run(program, bytearray(0x10000))
    assert (mem[0x10], mem[0x11]) == (1, 0)
    pre, post, relation = smt.transition(program)
    solver = z3.Solver()
    solver.add(relation, z3.Select(post, 0x11) != z3.Select(pre, 0x11))
    assert (
        solver.check() == z3.unsat
    )  # the first machine finishes; the second never runs


@pytest.mark.parametrize(
    "source, message",
    [
        ("hook:\n  txa\n  sta $1300\n  jmp tail\n", "do not give it"),
        ("hook:\n  sta $1300\n  jsr hook\n  jmp tail\n", "never|budget|past"),
    ],
)
def test_unstateable_hooks_are_refused(source, message):
    spec = "spec r;\nlayout { f, hook, tail: addr; }\ncell c: u8 @ L.f;\n" + (
        "machine m { step { native L.hook(a = c) resumes L.tail; c := 1; } }\nschedule { m; }"
    )
    src = "init:\n  rts\nplay:\n  jmp hook\ntail:\n  rts\n" + source
    make, binding, _ = players(src, {"f": 0x1300, "hook": "hook", "tail": "tail"})
    old = parse(spec)
    with pytest.raises(SidError, match=message):
        unnative(old, binding, make(old).view())


def test_unreached_hook_is_refused():
    spec = (
        "spec u;\nlayout { f, hook, tail: addr; flag: flag; }\ncell c: u8 @ L.f;\n"
        "machine m { step { if L.flag { native L.hook(a = c) resumes L.tail; } c := 1; } }\n"
        "schedule { m; }"
    )
    src = "init:\n  rts\nplay:\n  jmp hook\ntail:\n  rts\nhook:\n  jmp tail\n"
    make, binding, _ = players(
        src, {"f": 0x1300, "hook": "hook", "tail": "tail", "flag": 0}
    )
    old = parse(spec)
    with pytest.raises(SidError, match="never reached"):
        unnative(old, binding, make(old).view())


def cli_binding(view, entry):
    """The binding the CLI test passes by module:attr."""
    del view, entry
    _psid, labels, _play = tune(HOOK)
    return {
        "freq": 0x1300,
        "sid": 0xD400,
        "hook": labels["hook"],
        "tail": labels["tail"],
    }


def test_cli_unnative(tmp_path, capsys):
    psid, _labels, _play = tune(HOOK)
    sid = tmp_path / "t.sid"
    sid.write_bytes(psid)
    spec = tmp_path / "s.play"
    spec.write_text(NATIVE_SPEC)
    out = tmp_path / "o.play"
    binder = "tests.test_playspec_unnative:cli_binding"
    cli_main(
        ["playspec", "unnative", str(spec), str(sid), "--binder", binder]
        + ["-o", str(out), "--frames", "20"]
    )
    assert "native-free, same output over 20 frames" in capsys.readouterr().out
    assert "native" not in out.read_text() and "finish;" in out.read_text()


ARITH_SPEC = LOOP_SPEC.replace("x = n, y = 2", "a = n, x = n")

ARITH = """
init:
  lda #$c3
  sta $1300
  lda #$55
  sta $1302
  rts
play:
  jmp hook
tail:
  rts
hook:
  clc
  adc $1302
  sta $1303
  bcc nocarry
  lda $1303
  eor #$ff
  sta $1303
nocarry:
  asl
  asl
  ora #$01
  lsr
  sta $1304
  sec
  sbc #$20
  bmi neg
  sta $1305
  txa
  cmp $1303
  bcs big
  sta $1306,x
big:
  lda $1303
  sec
  sbc $1304
  sta $1307
  jmp tail
neg:
  dec $1308
  eor #$80
  sta $1309
  jmp tail
"""


@pytest.mark.parametrize("value", ["00", "7f", "c3", "ff"])
def test_arithmetic_carries_shifts_and_compares_match(value):
    make, binding, _ = players(
        ARITH.replace("#$c3", f"#${value}"),
        {"buf": 0x1300, "out": 0x1600, "hook": "hook", "tail": "tail"},
    )
    old = parse(ARITH_SPEC)
    new = unnative(old, binding, make(old).view())
    assert "native" not in new.source
    equivalent(make, old, new, 20)
