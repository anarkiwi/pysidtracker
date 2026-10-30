"""Spec derivation: a variant restates only what differs from its base."""

import pytest

from pysidtracker.asm6502 import assemble
from pysidtracker.playspec import SpecError, SpecPlayer, elaborate, load, parse, run
from pysidtracker.playspec.syntax import Loc

from .helpers import build_psid

BASE = """
spec base;
layout { a, b: addr; }
const STEP = 1;
cell out: u8[8] @ L.b;
proc bump(v: u8) -> u8 {
  return v + STEP;
}
machine voice[x in 0..2] {
  cell n: u8 @ L.a + x;
  proc tick() {
    advance: n := bump(n);
    emit: {
      out[x] := n;
      check: if n == 0x10 {
        n := 0;
      }
    }
  }
  step {
    tick();
  }
}
schedule { voice[0]; voice[1]; }
"""

BIND = {"a": 0x1000, "b": 0x1100}


def frames(spec, count=4, binding=None):
    """Memory after ``count`` calls of ``spec`` from a fixed start."""
    mem = bytearray(0x10000)
    mem[0x1000:0x1002] = bytes([0x0E, 0x40])
    return bytes(run(elaborate(spec, binding or BIND), mem, count))


def derived(text):
    return parse(text, {"base": parse(BASE)})


def test_a_variant_equals_the_fork_it_replaces():
    variant = derived("""
spec variant extends base;
layout { c: addr; }
override const STEP = 2;
cell trace: u8 @ L.c;
extend machine voice {
  patch tick {
    before advance { trace += 1; }
    replace check { if n >= 0x10 { n := 0x80; } }
    after emit { out[4 + x] := n; }
  }
}
""")
    fork = parse(
        BASE.replace("spec base;", "spec fork;")
        .replace("a, b: addr;", "a, b, c: addr;")
        .replace("STEP = 1", "STEP = 2")
        .replace("cell out", "cell trace: u8 @ L.c;\ncell out")
        .replace("advance:", "trace += 1;")
        .replace("n == 0x10 {\n        n := 0;", "n >= 0x10 {\n        n := 0x80;")
        .replace("  }\n  step", "    out[4 + x] := n;\n  }\n  step")
    )
    binding = dict(BIND, c=0x1200)
    assert frames(variant, binding=binding) == frames(fork, binding=binding)
    assert frames(variant, binding=binding) != frames(parse(BASE), binding=binding)
    assert variant.name == "variant" and variant.base == "base"
    assert set(variant.sources) == {"base", "variant"}


def test_replaced_statements_stay_labelled_for_the_next_variant():
    first = derived(
        "spec first extends base;\n"
        "extend machine voice { patch tick { replace check { n := 0x20; } } }"
    )
    second = parse(
        "spec second extends first;\n"
        "extend machine voice { patch tick { replace check { n := 0x30; } } }",
        {"first": first},
    )
    assert frames(second, 1)[0x1000] == 0x30
    assert frames(first, 1)[0x1000] == 0x20


def test_group_opens_no_scope_so_a_replacement_can_bind_lets():
    spec = derived("""
spec lets extends base;
extend machine voice {
  patch tick {
    replace advance { let nn = n + 3; n := nn; }
    after emit { out[6] := nn; }
  }
}
""")
    assert frames(spec, 1)[0x1106] == 0x43


def test_override_machine_schedule_proc_and_drop():
    spec = derived("""
spec whole extends base;
override proc bump(v: u8) -> u8 {
  return v ^ 0xFF;
}
override schedule { voice[1]; }
invariant small: out[0] < 0x80;
drop STEP;
""")
    mem = frames(spec, 1)
    assert mem[0x1101] == 0x40 ^ 0xFF and mem[0x1100] == 0
    assert [i[2] for i in spec.invariants] == ["small"]
    assert "STEP" not in {c[2] for c in spec.consts}


def test_an_override_keeps_its_base_position():
    spec = derived(
        "spec order extends base;\nconst EXTRA = 3;\noverride const STEP = EXTRA;"
    )
    assert [c[2] for c in spec.consts] == ["STEP", "EXTRA"]
    with pytest.raises(SpecError, match="undefined|EXTRA"):
        elaborate(spec, BIND)


@pytest.mark.parametrize(
    "text,message",
    [
        ("const STEP = 2;", "already in base base: say override"),
        ("override const NONE = 2;", "override of const NONE: base base has none"),
        ("override proc out() {}", "override of proc out: base base has none"),
        ("extend machine voice { override proc n() {} }", "machine voice has a cell"),
        ("override signal s: bool = false;", "outside extend machine"),
        ("extend machine nobody { }", "no such machine"),
        ("extend machine voice { cell n: u8 @ 0; }", "already in machine voice"),
        ("extend machine voice { patch nope { replace check {} } }", "no such proc"),
        (
            "extend machine voice { patch tick { after missing {} } }",
            "has 0 statements",
        ),
        ("patch bump { replace check {} }", "proc bump has 0 statements"),
        ("drop nothing;", "drop nothing: base base has 0"),
        ("extend machine voice { drop absent; }", "machine voice has 0"),
        ("layout { a: byte; }", "layout a: byte here, addr in base"),
        ("schedule { voice[0]; }", "the base has a schedule: say override"),
    ],
)
def test_edits_that_do_not_apply_exactly_are_refused(text, message):
    with pytest.raises(SpecError, match=message):
        derived(f"spec bad extends base;\n{text}")


def test_ambiguous_labels_are_refused():
    twice = BASE.replace("advance:", "check:")
    with pytest.raises(SpecError, match="has 2 statements so labelled"):
        parse(
            "spec bad extends base;\n"
            "extend machine voice { patch tick { replace check {} } }",
            {"base": parse(twice)},
        )


def test_override_schedule_needs_one_in_the_base():
    bare = parse("spec bare;")
    with pytest.raises(SpecError, match="the base has none"):
        parse("spec s extends bare;\noverride schedule { }", {"bare": bare})


def test_edits_need_a_base_and_the_base_must_resolve():
    with pytest.raises(SpecError, match="needs a base"):
        parse("spec s;\ndrop x;")
    with pytest.raises(SpecError, match="no base spec 'gone'"):
        parse("spec s extends gone;", {})
    with pytest.raises(SpecError, match="no base spec 'gone'"):
        parse("spec s extends gone;")
    with pytest.raises(SpecError, match="extends itself"):
        parse("spec base extends base;", {"base": parse(BASE)})


def test_load_resolves_bases_beside_the_file(tmp_path):
    (tmp_path / "base.play").write_text(BASE)
    (tmp_path / "mid.play").write_text(
        "spec mid extends base;\noverride const STEP = 4;"
    )
    (tmp_path / "top.play").write_text("spec top extends mid;\nconst X = 1;")
    spec = load(tmp_path / "top.play")
    assert frames(spec, 1)[0x1000] == 0x12 and set(spec.sources) == {
        "base",
        "mid",
        "top",
    }
    (tmp_path / "loop1.play").write_text("spec loop1 extends loop2;")
    (tmp_path / "loop2.play").write_text("spec loop2 extends loop1;")
    with pytest.raises(SpecError, match="extends itself"):
        load(tmp_path / "loop1.play")
    with pytest.raises(SpecError, match="no base spec 'nowhere'"):
        (tmp_path / "orphan.play").write_text("spec orphan extends nowhere;")
        load(tmp_path / "orphan.play")


def test_diagnostics_name_the_spec_each_line_is_from():
    spec = derived(
        "spec where extends base;\n"
        "extend machine voice { patch tick { after emit { undefined_cell := 1; } } }"
    )
    with pytest.raises(SpecError, match=r"line where:2: undefined"):
        elaborate(spec, BIND)
    assert Loc(3, "a") != Loc(3, "b") and Loc(3, "a") == Loc(3, "a") == 3
    assert f"{Loc(3)}" == "3" and f"{Loc(3, 'a'):>3}" == "  3"


def test_labels_and_groups_lower_like_plain_statements():
    plain = parse(
        BASE.replace("advance: ", "").replace("emit: {", "{").replace("check: ", "")
    )
    assert frames(parse(BASE)) == frames(plain)
    labelled_return = parse(
        BASE.replace(
            "    tick();", "    early: { if n == 0x40 { return; } }\n    tick();"
        )
    )
    mem = frames(labelled_return, 1)
    assert mem[0x1001] == 0x40 and mem[0x1000] == 0x0F


VARIANT_TUNE = """
init:
  lda #$10
  sta $1300
  rts
play:
  lda $1300
  clc
  adc #$02
  sta $1300
  sta $d400
  rts
"""

ONE = """
spec one;
layout { freq, sid: addr; }
const STEP = 1;
cell sid: u8[32] @ L.sid out;
machine m {
  cell freq: u8 @ L.freq;
  step { freq += STEP; sid[0] := freq; }
}
schedule { m; }
"""


def test_a_binder_picks_a_variant_per_instance():
    code, labels = assemble(VARIANT_TUNE, 0x1000)
    image = build_psid(code, 0x1000, init=labels["init"], play=labels["play"])[0x7C:]
    one = parse(ONE)
    two = parse("spec two extends one;\noverride const STEP = 2;", {"one": one})
    binding = {"freq": 0x1300, "sid": 0xD400}

    def player(bind):
        return SpecPlayer(one, image, 0x1000, labels["init"], labels["play"], bind=bind)

    plain, variant = player(lambda v, e: binding), player(lambda v, e: (two, binding))
    for _ in range(3):
        plain.play_frame()
        variant.play_frame()
    assert variant.regs[0] == 0x16 and plain.regs[0] == 0x13
