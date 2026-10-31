"""Rescheduling a spec's SID writes into a canonical order, on synthetic specs."""

import argparse
import json
import random

import numpy as np
import pytest

from pysidtracker.playspec import SpecPlayer, cli, elaborate, parse
from pysidtracker.playspec.cli import add_playspec_commands
from pysidtracker.playspec import outorder as O
from pysidtracker.playspec.pyexec import compile_program, controls

ORDER = (
    "fc_lo,fc_hi,res_filt,mode_vol,voice(ad,sr,ctrl:2,ctrl,freq_lo,freq_hi,pw_lo,pw_hi)"
)
BIND = {"sid": 0xD400, "w": 0x1000}
HEAD = """spec t;
layout { sid, w: addr; }
cell chip: u8[25] @ L.sid out;
cell tick: u8 @ L.w + 9;
"""
PLAYER = HEAD + """
cell regs: u8[4] @ L.w;
cell tmp: u8 @ L.w + 8;
machine filter {
  step {
    tmp := tick + 1;
    regs[0] := tmp ^ regs[1];
    regs[3] := 0x0F;
  }
}
machine voice[x in 0..3] {
  cell freq: u8 @ L.w + 16 + x;
  cell pw: u8 @ L.w + 20 + x;
  cell gate: u8 @ L.w + 24 + x;
  step {
    chip[7 * x + 2] := pw;
    if tick & 1 == 0 {
      chip[7 * x + 4] := gate & 0xFE;
      chip[7 * x + 5] := 0;
      return;
    }
    freq += 1;
    chip[7 * x + 0] := freq;
    chip[7 * x + 4] := gate;
    if neg(chip[7 * x + 4]) {
      chip[7 * x + 6] := 0xF0;
    }
    chip[7 * x + 4] := gate | 1;
  }
}
machine out {
  step {
    for r in 0..4 {
      chip[21 + r] := regs[r];
    }
    tick += 1;
    if tick == 0x80 {
      finish;
    }
    gate_all: chip[4] := 0;
  }
}
schedule {
  filter;
  for v in 0..3 {
    voice[v];
  }
  out;
}
"""
STALE = HEAD + """
cell pw: u8 @ L.w;
machine m {
  step {
    chip[2] := pw;
    pw += 1;
    chip[5] := pw;
  }
}
schedule { m; }
"""


def program(text):
    """``text`` elaborated against :data:`BIND`."""
    return elaborate(parse(text), BIND)


def play(prog, memory, frames):
    """``(memory, per-call SID writes)`` of ``frames`` calls from ``memory``."""
    mem, frame, state, calls = (
        bytearray(memory),
        compile_program(prog),
        controls(prog),
        [],
    )

    def store(addr, value):
        if 0xD400 <= addr < 0xD419:
            calls[-1].append((addr - 0xD400, value))
        mem[addr] = value

    for _ in range(frames):
        calls.append([])
        frame(mem.__getitem__, store, state)
    return mem, calls


@pytest.mark.parametrize("seed", range(4))
def test_reordered_calls_write_the_kept_writes_in_order(seed):
    rng = random.Random(seed)
    memory = bytes(rng.randrange(256) for _ in range(0x10000))
    original = program(PLAYER)
    result = O.reorder(original, ORDER)
    slots = result.slots
    mem, calls = play(original, memory, 300)
    new_mem, new_calls = play(result.program, memory, 300)
    assert [O.kept(c, slots) for c in calls] == new_calls
    differs = np.flatnonzero(
        np.frombuffer(mem, np.uint8) != np.frombuffer(new_mem, np.uint8)
    )
    assert set(differs.tolist()) <= set(result.promoted)
    assert any(len(c) > len(n) for c, n in zip(calls, new_calls))


def test_reorder_removes_staging_and_scratch():
    result = O.reorder(program(PLAYER), ORDER)
    assert result.promoted == [0x1000, 0x1003, 0x1008]
    assert result.staged == []
    assert result.after.cells == result.inlined.cells - 3
    assert result.paths == {result.slots}


def test_check_proves_the_reordered_call():
    original = program(PLAYER)
    assert O.check(original, O.reorder(original, ORDER)) is None


def test_check_finds_a_wrong_schedule():
    original = program(PLAYER)
    result = O.reorder(original, ORDER)
    body = O.rewrite(
        result.program.body,
        lambda e: e,
        lambda s: [] if s[0] == "wr" and s[1] == O.K(0xD400 + 5, 16) else None,
    )
    result.program = O.replace(result.program, body=body)
    assert O.check(original, result) is not None


def test_a_value_clobbered_before_its_slot_is_staged():
    original = program(STALE)
    result = O.reorder(original, ORDER)
    assert result.staged == ["o1_2"]
    assert O.check(original, result) is None
    _, calls = play(result.program, bytes(0x10000), 2)
    assert calls == [[(5, 1), (2, 0)], [(5, 2), (2, 1)]]


def test_order_that_needs_no_staging():
    order = (
        "fc_lo,fc_hi,res_filt,mode_vol,voice(pw_lo,ad,sr,ctrl,freq_lo,freq_hi,pw_hi)"
    )
    assert O.reorder(program(STALE), order).staged == []


@pytest.mark.parametrize(
    "text, slots",
    [
        (ORDER, 28),
        (
            "interleave(freq_lo,freq_hi,pw_lo,pw_hi,ctrl,ad,sr),fc_lo,fc_hi,res_filt,mode_vol",
            25,
        ),
    ],
)
def test_parse_order(text, slots):
    parsed = O.parse_order(text)
    assert len(parsed) == slots and {r for r, _ in parsed} == set(range(25))


def test_interleave_runs_register_by_register():
    parsed = O.parse_order(
        "interleave(freq_lo,freq_hi,pw_lo,pw_hi,ctrl,ad,sr),fc_lo,fc_hi,res_filt,mode_vol"
    )
    assert parsed[:3] == ((0, 1), (7, 1), (14, 1))


@pytest.mark.parametrize(
    "text",
    [
        "fc_lo,voice(ad)",
        "fc_lo,fc_hi,res_filt,mode_vol,voice(ad,sr,ctrl,ctrl:2,freq_lo,freq_hi,pw_lo,pw_hi)",
        "fc_lo,fc_hi,res_filt,mode_vol,voice(ad,sr,ctrl:3,ctrl,freq_lo,freq_hi,pw_lo,pw_hi)",
        "fc_lo,fc_lo,fc_hi,res_filt,mode_vol,voice(ad,sr,ctrl,freq_lo,freq_hi,pw_lo,pw_hi)",
    ],
)
def test_parse_order_refuses(text):
    with pytest.raises(O.OrderError):
        O.parse_order(text)


def test_kept_takes_the_last_writes_in_slot_order():
    slots = O.parse_order(ORDER)
    writes = [(2, 1), (4, 8), (5, 3), (4, 9), (4, 10), (2, 7), (24, 15)]
    assert O.kept(writes, slots) == [(24, 15), (5, 3), (4, 9), (4, 10), (2, 7)]


@pytest.mark.parametrize(
    "stmts, message",
    [
        ("mem[L.sid + u16(tick)] := 1;", "may reach the SID"),
        ("tick := mem[L.sid + u16(tick)];", "may read the SID"),
        ("mem[mem16[L.w]] := 1;", "unbounded pointer"),
        ("native 0x1000() resumes 0x1003;", "opaque"),
    ],
)
def test_reorder_refuses(stmts, message):
    text = HEAD + "machine m { step { " + stmts + " } }\nschedule { m; }\n"
    with pytest.raises(O.OrderError, match=message):
        O.reorder(program(text), ORDER)


def test_volatile_reads_stay_put():
    text = HEAD + """cell osc: u8 @ L.sid + 0x1B volatile;
machine m { step { let a = osc; let b = osc; chip[0] := a; chip[1] := b; } }
schedule { m; }
"""
    original = program(text)
    result = O.reorder(original, ORDER)
    reads = [s for s in O.statements(result.program.body) if s[0] == "set"]
    assert len(reads) == 2
    assert O.check(original, result) is None


def spec_player(transform=None):
    """A :class:`SpecPlayer` running :data:`PLAYER` at a bare RTS play routine."""
    image = bytes([0x60, 0x60])
    return SpecPlayer(
        parse(PLAYER),
        image,
        0x1000,
        0x1000,
        0x1001,
        bind=lambda view, entry: {"sid": 0xD400, "w": 0x2000},
        transform=transform,
    )


def test_compare_plays_both_ways():
    verdict, results = O.compare(spec_player, ORDER, 200)
    assert verdict is None
    assert len(results) == 1 and results[0].staged == []


def test_compare_reports_unordered_writes():
    verdict, results = O.compare(lambda transform: spec_player(), ORDER, 5)
    assert verdict.startswith("frame 0: writes") and results == []


class FakeFormat:
    """An installed format whose model plays :data:`PLAYER`."""

    name = "fake"

    @staticmethod
    def load(data, subtune):
        """The model of ``data``."""
        return (data, subtune)

    @staticmethod
    def spec_player(model, transform=None):
        """The spec player of ``model``."""
        assert model[0] == b"tune"
        return spec_player(transform)


def test_survey_and_cli(tmp_path, monkeypatch):
    tune = tmp_path / "a.sid"
    tune.write_bytes(b"tune")
    monkeypatch.setattr("pysidtracker.formats.find_format", lambda data: FakeFormat)
    report = O.survey(str(tune), ORDER, 50)
    assert report["format"] == "fake" and report["verdict"] is None
    assert (
        report["programs"][0]["after"]["cells"]
        < report["programs"][0]["inlined"]["cells"]
    )
    tunes = tmp_path / "tunes.csv"
    tunes.write_text(f"path,player\n{tune},fake\n")
    monkeypatch.setattr(
        cli,
        "imap",
        lambda fn, tasks, jobs: ((i, fn(*t), None) for i, t in enumerate(tasks)),
    )
    out = tmp_path / "report.json"
    parser = argparse.ArgumentParser()
    add_playspec_commands(parser)
    args = parser.parse_args(
        ["reorder", str(tunes), str(tune), "--frames", "20", "--out", str(out)]
    )
    args.func(args)
    assert set(json.loads(out.read_text())) == {str(tune)}
