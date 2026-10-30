"""Generic playspec constructs on synthetic memory, and every elaboration diagnostic."""

import re

import pytest

from pysidtracker.playspec import SpecError, elaborate, parse, run

HEADER = "spec t;\nlayout { a, b, c: addr; l: list; }\n"
BIND = {"a": 0x1000, "b": 0x1100, "c": 0x1200, "l": (1, 0)}

CONSTRUCTS = """
const HALF = 7 / 2 + 7 % 2;
cell tab: ptr[2] @ (L.a, L.a + 2) ro;
cell owner: u8 @ L.b;
cell log: u8[8] @ L.c;
cell wide: u16 @ L.b + 0x10;
proc first(s: stream) -> u8 {
  let v = next s;
  return v;
}
machine src {
  step {
    owner += 1;
    wide := mem16[L.a + 2];
  }
}
machine voice[x in 0..2] {
  cell note: u8 @ L.b + 0x20 + x;
  control gate on note mask 0x01 { off = 0, on = 1 }
  step {
    stream s = tab[x] @ 0;
    let v = first(s);
    log[x] := v + owner;
    in gate {
      off => { goto on; }
      on => { goto off; }
    }
    log[2 + x] := voice[0].note;
    log[4 + x] := u8(sgt(v, 0x10)) + u8(sge(v, v)) + u8(sle(v, v));
    for k in 0..2 {
      log[6] += 1;
    }
    log[7] := lo(zp(u16(owner) + 0xFF)) + HALF;
  }
}
schedule {
  src;
  for i in L.l {
    voice[i];
  }
}
"""


def test_generic_constructs():
    mem = bytearray(0x10000)
    mem[0x1002:0x1004] = bytes([0x30, 0x31])
    mem[0x3000], mem[0x3100] = 0x05, 0x90
    run(elaborate(parse(HEADER + CONSTRUCTS), BIND), mem)
    assert mem[0x1100] == 1 and mem[0x1110] | mem[0x1111] << 8 == 0x3130
    assert list(mem[0x1200:0x1208]) == [0x06, 0x91, 0x01, 0x00, 0x02, 0x02, 4, 4]
    assert list(mem[0x1120:0x1122]) == [1, 1]


def test_constructs_type_check_without_a_binding():
    program = elaborate(parse(HEADER + CONSTRUCTS))
    assert program.order == ["src", "voice[?]"]


M = "machine m {{ {} step {{ {} }} }}\nschedule {{ m; }}\n"


def body(decls="", stmts="", top=""):
    """A one-machine spec."""
    return top + M.format(decls, stmts)


@pytest.mark.parametrize(
    "spec, message",
    [
        (body(top="cell q: bool @ L.a;\n"), "bool is not a memory type"),
        (body(top="cell q: u8 @ (L.a, L.b);\n"), "split place needs a ptr or u16"),
        (body(top="cell q: u8 @ L.a ro out;\n"), "cannot be both ro and out"),
        (
            body("cell t: u8[2] @ L.a; control c on t mask 1 { p }"),
            "is not a scalar u8 cell",
        ),
        (body("control c { p }", "in c { zz => { } }"), "has no state 'zz'"),
        (body("control c { p } control d { p }", "goto p;"), "ambiguous control state"),
        (body("cell v: u8 @ L.a; signal s: u8 = v;"), "default must be static"),
        (body("", "foo();"), "undefined proc 'foo'"),
        (body("proc f(a: u8) { }", "f();"), "takes 1 arguments, 0 given"),
        (body("proc f(s: stream) { }", "f(1);"), "needs a stream"),
        (
            body("cell v: u8 @ L.a;", "stream s = 0 @ 0; v += next s;"),
            "next only assigns with :=",
        ),
        (body("", "stream s = 0 @ 0; let q = s.foo;"), "has no 'foo'"),
        (body("", "stream s = 0 @ 0; let q = s;"), "is a stream"),
        (body("", "stream s = 0 @ 0; s.base := 1;"), "cannot assign to s.base"),
        (body("", 'stream s = 0 @ 0 grammar "abc";'), "grammar is"),
        (body("", "native 0x1000(b = 1) resumes 0x1000;"), "native arguments"),
        (
            body("cell t: u8[2] @ L.a;", "let q = t[true];"),
            "a truth value cannot index",
        ),
        (body("", "let q: u8 = ~3;"), "~ of an untyped constant"),
        (body("", "let q = true << 1;"), "shift needs a typed integer"),
        (body("", "let q = L.l;"), "L.l is a list"),
        (body("", "let q = len(5);"), "expected a static list"),
        (
            body("cell v: u8 @ L.a;", "let q = member(v, L.l);"),
            "member() needs a static value",
        ),
        (body("", "let q = foo(1);"), "undefined function 'foo'"),
        (body("", "let q = lo(1, 2);"), "lo() takes 1 arguments"),
        (body("", "let q = neg(true);"), "neg() needs a typed integer"),
        (body("", "let q = ite(1, 2, 3);"), "ite() condition is not a truth value"),
        (
            body("cell v: u8 @ L.a;", "let q = carry(v, v, 5);"),
            "carry-in is not a truth value",
        ),
        (body("cell v: u8 @ L.a;", "let q = rol(v, 5);"), "rol/ror need"),
        (
            body("cell v: u8 @ L.a;", "let q = bit(v, v);"),
            "bit() needs a static bit number",
        ),
        (body("", "let q = true < false;"), "< of truth values"),
        (body("", "let q = true + 1;"), "truth value mixed with an integer"),
        (body("", "let q = true + false;"), "+ of truth values"),
        (body("proc f() { return 1; }", "f();"), "return value does not match"),
        (body("proc f() { }", "let q = f();"), "returns no value"),
        (body("", "let q: ptr = 1;"), "ptr is not a local type"),
        (body("", "HALF := 1;", top="const HALF = 1;\n"), "cannot assign to HALF"),
        (body("", "zz[1] := 2;"), "cannot assign to zz[...]"),
        (body("", "zz := 2;"), "undefined name 'zz'"),
        (body("", "let q = zz[1];"), "undefined cell 'zz'"),
        (body("", "let q = m[0];"), "needs a member"),
        (body("", "let q = cpu.q;"), "entry registers are"),
        (body("cell t: u8[2] @ L.a;", "t := 1;"), "cell t needs an index"),
        (body("cell v: u8 @ L.a;", "v[1] := 1;"), "cell v is not an index"),
        (body("", "let q = 1 + zz.s;"), "undefined machine 'zz'"),
        (body(top="const C = mem[1];\n"), "const C is not static"),
        (
            "spec t;\nlayout { a: addr; }\nmachine m { step { } }\n",
            "spec has no schedule",
        ),
        (
            "spec t;\nlayout { a: addr; }\nmachine m { }\nschedule { m; }\n",
            "has no step",
        ),
        (
            "spec t;\nlayout { a: addr; }\nmachine m[x in 0..2] { step { } }\nschedule { m; }\n",
            "must be indexed",
        ),
        (
            "spec t;\nlayout { a: addr; }\nmachine m[x in 0..2] { step { } }\nschedule { m[5]; }\n",
            "m[5] is outside 0..2",
        ),
        (
            "spec t;\nlayout { a: addr; }\nmachine p { cell c: u8 @ L.a; step { } }\n"
            "machine m { step { let q = p.zz; } }\nschedule { p; m; }\n",
            "has no signal or cell",
        ),
        (
            "spec t;\nlayout { a: addr; }\nmachine p[x in 0..2] { cell c: u8 @ L.a + x; step { } }\n"
            "machine m { cell v: u8 @ L.a + 4; step { let q = p[v].c; } }\nschedule { m; }\n",
            "needs a static index",
        ),
    ],
)
def test_diagnostics(spec, message):
    text = (
        spec
        if spec.startswith("spec")
        else "spec t;\nlayout { a, b: addr; l: list; }\n" + spec
    )
    with pytest.raises(SpecError, match=re.escape(message)):
        elaborate(parse(text), {"a": 0x1000, "b": 0x1100, "l": (0,)})


def test_binding_must_supply_declared_fields():
    with pytest.raises(SpecError, match="binding has no layout field 'b'"):
        elaborate(parse(HEADER + CONSTRUCTS), {"a": 0x1000, "c": 0x1200, "l": (0,)})
