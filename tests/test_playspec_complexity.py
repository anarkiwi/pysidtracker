"""playspec.complexity: static measures, statement probes and corpus coverage."""

import dataclasses
import json
import re
from collections import Counter

import pytest
from jennings import memory as jennings_memory

from pysidtracker import jobs
from pysidtracker.maincli import main
from pysidtracker.playspec import SpecPlayer, chains, complexity, elaborate, parse
from pysidtracker.playspec.pyexec import compile_program, controls

from .conftest import BACKENDS
from .helpers import assemble, build_psid

ENGINE = "native" if "native" in BACKENDS and jennings_memory.NATIVE else "python"

STATIC = """spec t;
layout { base, tab_lo, tab_hi, out: addr; fx, alt: flag; }
cell ptr: u16[4] @ (L.tab_lo, L.tab_hi) ro;
cell o: u8 @ L.out;
state p: u16 = 0;
state a: u8 = 0;
state b: u8 = 0;
table data @ p stride 1 { v: u8; }
machine m {
  proc one() { if L.fx { a := a + 1; o := (a & 0x0f) | 0x10; } }
  proc two() { if L.fx { a := a + 1; o := (a & 0x0f) | 0x10; } }
  proc three() { if L.alt { b := b + 1; o := (b & 0x0f) | 0x10; } }
  step {
    p := ptr[a] + L.base;
    one(); two(); three();
    if L.alt && a == 3 { p := word(a, b); }
  }
}
schedule { m; }
"""

PLAY = """spec p;
layout { out: addr; on: flag; }
cell o: u8 @ L.out;
state n: u8 = 0;
machine m {
  step {
    n += 1;
    if n == 2 {
      o := 7;
    } else {
      o := 1;
    }
    if L.on { o := o + 1; }
  }
}
schedule { m; }
"""


@pytest.fixture(autouse=True)
def one_engine(backend):
    """The measures do not depend on jennings' engine: run each test on one."""
    if backend != ENGINE:
        pytest.skip("measured on the other engine")


def lines_of(text: str, needle: str) -> list:
    return [i for i, line in enumerate(text.splitlines(), 1) if needle in line]


def test_static_measures():
    analysis = complexity.analyse(parse(STATIC))
    units = analysis.units
    assert set(units) == {"decls", "m.one", "m.two", "m.three", "m.step", "schedule"}
    assert (units["m.one"].decisions, units["m.step"].decisions) == (1, 2)
    assert units["m.one"].depth == 1
    assert units["m.one"].nodes == units["m.two"].nodes
    assert units["m.one"].statements == ["t:10:assign", "t:10:if"]
    totals = analysis.totals()
    assert totals["decisions"] == sum(u.decisions for u in units.values())
    assert totals["statements"] == sum(len(u.statements) for u in units.values())
    assert totals["multi_writer_state"] == 2
    assert analysis.layout["fx"] == {
        "kind": "flag",
        "guard": 2,
        "code": 0,
        "decl": 0,
        "units": ["m.one", "m.two"],
    }
    assert analysis.layout["alt"]["guard"] == 2
    assert analysis.layout["base"]["code"] == 1
    assert analysis.layout["tab_lo"]["decl"] == 1
    assert analysis.coupling["a"]["writers"] == ["m.one", "m.two"]
    assert analysis.coupling["p"]["writers"] == ["m.step"]
    assert "m.step" in analysis.coupling["a"]["readers"]
    assert Counter(f["kind"] for f in analysis.addressing) == {
        "relocation": 1,
        "word assembly": 1,
        "split pointer table": 1,
        "pointer state": 1,
    }
    text = "\n".join(complexity.render(analysis, complexity.report(analysis), top=3))
    assert text.startswith("spec t: units 6")
    assert re.search(r"split pointer table +1 +lines 3\n", text)


def test_clones_are_maximal_and_shape_matches_renamed_state():
    exact = complexity.analyse(parse(STATIC)).clones
    top = exact[0]
    assert top["count"] == 2 and not top["nested"]
    assert {u for u, _ in top["at"]} == {"m.one", "m.two"}
    assert top["savings"] == top["size"]
    assert all(c["count"] != 2 or c is top or c["nested"] for c in exact)
    shaped = complexity.analyse(parse(STATIC), shape=True).clones
    assert shaped[0]["count"] == 3
    assert {u for u, _ in shaped[0]["at"]} == {"m.one", "m.two", "m.three"}
    assert not complexity.analyse(parse(STATIC), min_nodes=10_000).clones


def _frames(program, count):
    memory = bytearray(0x10000)
    slots = {n: i for i, (n, _w, _d) in enumerate(program.controls)}
    frame, state = compile_program(program, slots), controls(program)
    probes = [n for n in slots if n.startswith(complexity.PROBE)]
    runs = []
    for _ in range(count):
        frame(memory.__getitem__, memory.__setitem__, state, None, {})
        runs.append({n for n in probes if state[slots[n]]})
        for n in probes:
            state[slots[n]] = 0
    return memory, runs


@pytest.mark.strain_exempt(
    reason="asserts probing leaves the output unchanged and marks only statements run"
)
def test_probes_mark_the_statements_run_and_change_nothing():
    spec = parse(PLAY)
    binding = {"out": 0x1300, "on": 0}
    with complexity.probing():
        probed = elaborate(spec, binding)
    plain = elaborate(spec, binding)
    assert not any(n.startswith(complexity.PROBE) for n, _w, _d in plain.controls)
    memory, runs = _frames(probed, 3)
    assert memory == _frames(plain, 3)[0]
    assert memory[0x1300] == 1
    seven, one, off = (
        f"_{lines_of(PLAY, text)[0]}_assign" for text in ("o := 7", "n += 1", "o + 1")
    )
    names = [{n[n.index("_", 4) :] for n in run} for run in runs]
    assert seven in names[1] and seven not in names[0] | names[2]
    assert all(one in run for run in names)
    assert not any(off in run for run in names)


@dataclasses.dataclass(frozen=True)
class Tune:
    """A format stand-in: one tune whose play entry is the spec."""

    name: str
    spec: object
    image: bytes = b""

    def load(self, data, subtune):
        del data, subtune
        return self

    def spec_player(self, model, native=False):
        binding = self.spec(model)
        code, labels = assemble("init:\n  rts\nplay:\n  rts\n", 0x1000)
        image = build_psid(code, 0x1000, init=labels["init"], play=labels["play"])
        return SpecPlayer(
            binding.spec,
            image[0x7C:],
            0x1000,
            labels["init"],
            labels["play"],
            bind=binding.bind,
            native=native,
        )


@dataclasses.dataclass
class Binding:
    """The part of a ``SpecBinding`` :func:`cover_tune` uses."""

    spec: object
    bind: object


def _format(name="p"):
    spec = parse(PLAY.replace("spec p;", f"spec {name};"))
    return Tune(
        "fake", lambda m: Binding(spec, lambda view, entry: {"out": 0x1300, "on": 1})
    )


def test_cover_counts_frames_and_tunes(tmp_path, monkeypatch):
    path = tmp_path / "p.play"
    path.write_text(PLAY)
    tunes = [tmp_path / "a.sid", tmp_path / "b.sid"]
    for tune in tunes:
        tune.write_bytes(b"x")
    monkeypatch.setattr("pysidtracker.formats.find_format", lambda data: _format())
    monkeypatch.setattr(
        jobs,
        "imap",
        lambda fn, tasks, n=None: ((i, fn(*t), None) for i, t in enumerate(tasks)),
    )
    corpus = complexity.cover(str(path), [str(t) for t in tunes], frames=4)
    one = corpus[str(tunes[0])]
    assert one["flags"] == {"on": 1}
    seven = f"p:{lines_of(PLAY, 'o := 7')[0]}:assign"
    assert one["hits"][seven] == 1
    table = complexity.report(complexity.analyse(parse(PLAY)), corpus, rare=1)
    step = next(u for u in table["units"] if u["name"] == "m.step")
    assert step["unreached"] == []
    assert table["corpus"] == {
        "tunes": 2,
        "skipped": 0,
        "errors": {},
        "flags": {"on": 2},
        "combinations": 1,
    }
    assert len(step["statements"]) == 6
    assert step["frames"] == 2 * (4 * 4 + 1 + 3)


def test_cover_skips_other_specs_and_reports_errors(tmp_path, monkeypatch):
    path = tmp_path / "p.play"
    path.write_text(PLAY)
    tune = tmp_path / "a.sid"
    tune.write_bytes(b"x")
    monkeypatch.setattr("pysidtracker.formats.find_format", lambda data: _format("q"))
    assert complexity.cover_tune(str(tune), str(path), 1) == {
        "format": "fake",
        "skipped": "q",
    }
    corpus = {"x": {"error": "boom"}, "y": {"format": "fake", "skipped": "q"}}
    table = complexity.report(complexity.analyse(parse(PLAY)), corpus)
    assert table["corpus"]["errors"] == {"x": "boom"}
    assert table["corpus"]["skipped"] == 1


def test_cli_reports_and_compares(tmp_path, capsys):
    spec, older, out = tmp_path / "p.play", tmp_path / "o.play", tmp_path / "r.json"
    spec.write_text(PLAY)
    older.write_text(PLAY.replace("    n += 1;\n", "    n += 1;\n    n += 1;\n"))
    main(["playspec", "complexity", str(spec), "--diff", str(older), "--out", str(out)])
    table = json.loads(out.read_text())
    assert table["delta"]["totals"]["statements"] == [7, 6]
    assert table["delta"]["units"]["m.step"]["nodes"][0] > table["units"][1]["nodes"]
    shown = capsys.readouterr().out
    assert "units by decisions" in shown and re.search(r"statements +7 -> 6", shown)


CHAINS = """spec ch;
layout { out: addr; }
cell o: u8 @ L.out;
state lo: u8 = 0;
state hi: u8 = 0;
state dlo: u8 = 0;
state dhi: u8 = 0;
state keep: bool = false;
machine m {
  proc add16() {
    let c = carry(lo, dlo);
    lo += dlo;
    hi := hi + dhi + u8(c);
  }
  proc sub16(c: bool) {
    let l = lo;
    lo := l - dlo - ite(c, 0, 1);
    hi := hi - dhi - ite(borrow(l, dlo, !c), 1, 0);
  }
  proc dec() {
    let c = !borrow(lo, dlo, true);
    lo := lo - dlo - 1;
    hi := hi - dhi - u8(!c);
  }
  proc off() {
    let c = carry(lo, dlo);
    lo += dlo;
    lo += 1;
    hi += u8(c);
  }
  proc raw() {
    let c = carry(lo, mem[L.out]);
    lo += mem[L.out];
    hi += u8(c);
  }
  proc mark() { keep := carry(lo, dhi); }
  proc take() { hi += u8(keep); }
  proc shadow() {
    let keep = carry(lo, dhi);
    keep := carry(lo, dlo);
    hi += u8(keep);
  }
  step {
    sub16(borrow(lo, dhi));
    if carry(dlo, dhi) { o := 1; }
    lo := lo + dlo + u8(carry(lo, dlo));
    add16(); dec(); off(); raw(); mark(); take(); shadow();
  }
}
schedule { m; }
"""


def _chain(found, unit):
    return next(c for c in found if unit in c["proofs"])


def test_carry_chains_are_followed_classified_proved_and_reported():
    found = chains.find(parse(CHAINS))
    add = _chain(found, "m.add16")
    assert add["kind"] == "width" and add["ends"] == ["internal"]
    assert add["pairs"] == [["lo", "hi"]] and add["writes"] == {"hi": 7, "lo": 7}
    assert add["proofs"]["m.add16"]["status"] == "proved: hi:lo = hi:lo + dhi:dlo"
    assert add["saved"] > 0
    sub = _chain(found, "m.sub16")
    assert sub["ends"] == ["call"] and set(sub["units"]) == {"m.step", "m.sub16"}
    assert sub["proofs"]["m.sub16"]["status"] == "proved: hi:lo = hi:lo - dhi:dlo - !c"
    assert sub["proofs"]["m.step"]["status"] == "not tried: not one byte pair"
    assert _chain(found, "m.dec")["proofs"]["m.dec"]["status"].endswith("- dhi:dlo - 1")
    assert _chain(found, "m.off")["proofs"]["m.off"] == {
        "status": "refuted",
        "saved": 0,
    }
    raw = _chain(found, "m.raw")["proofs"]["m.raw"]["status"]
    assert raw == "not tried: outside the subset (syntax mem)"
    escape = found[0]
    assert escape["escapes"] == ["keep"] and escape["ends"] == ["escape"]
    assert set(escape["units"]) == {"m.mark", "m.take"}
    kinds = Counter(c["kind"] for c in found if c["units"] == ["m.step"])
    assert kinds == {"flag": 1, "n-ary": 1}
    shadowed = _chain(found, "m.shadow")
    assert shadowed["ends"] == ["internal"] and not shadowed["escapes"]
    condition = next(c for c in found if "condition" in c["ends"])
    assert condition["links"] == ["carry(dlo, dhi)"]
    analysis = complexity.analyse(parse(CHAINS))
    totals = analysis.totals()
    assert totals["chains"] == len(analysis.chains) == 10
    assert totals["chains_proved"] == 3 and totals["escaping_chains"] == 1
    assert totals["chain_savings"] == sum(c["saved"] for c in analysis.chains)
    text = "\n".join(complexity.render(analysis, complexity.report(analysis)))
    shown = text[text.index("carry chains") :].splitlines()
    assert shown[1].startswith("flag") and "escape via keep" in shown[1]
    section = "\n".join(shown)
    assert "m.off: refuted" in section and "m.raw" not in section
    assert (
        chains.src(parse(CHAINS).machines["m"][6][4][-1][0][4])
        == "carry(lo, mem(L.out))"
    )
