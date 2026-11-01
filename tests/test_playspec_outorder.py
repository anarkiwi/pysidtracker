"""Rescheduling a spec's SID writes into a canonical order, on synthetic specs."""

import argparse
import json
import random

import numpy as np
import pytest

from pysidtracker.playspec import SpecPlayer, cli, elaborate, parse
from pysidtracker.playspec.cli import add_playspec_commands
from pysidtracker.playspec import outorder as O
from pysidtracker.playspec import outproof
from pysidtracker.playspec.pyexec import compile_program, controls

ORDER = O.ORDER
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
    assert [O.kept(c, slots, result.native) for c in calls] == new_calls
    differs = np.flatnonzero(
        np.frombuffer(mem, np.uint8) != np.frombuffer(new_mem, np.uint8)
    )
    assert set(differs.tolist()) <= set(result.promoted)
    assert any(c != n for c, n in zip(calls, new_calls))


def test_reorder_removes_staging_and_scratch():
    result = O.reorder(program(PLAYER), ORDER)
    assert result.promoted == [0x1000, 0x1003, 0x1008]
    assert result.staged and all(n.endswith("_4") for n in result.staged)
    assert result.after.cells == result.inlined.cells - 3
    assert result.paths == {result.slots}


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


def test_order_that_needs_no_staging():
    order = (
        "fc_lo,fc_hi,res_filt,mode_vol,voice(pw_lo,ad,sr,ctrl,freq_lo,freq_hi,pw_hi)"
    )
    assert O.reorder(program(STALE), order).staged == []


@pytest.mark.parametrize(
    "text, slots",
    [
        (ORDER, 52),
        (
            "fc_lo,fc_hi,res_filt,mode_vol,voice(ad,sr,ctrl,freq_lo,freq_hi,pw_lo,pw_hi)",
            25,
        ),
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


SONG = "fc_lo,fc_hi,res_filt,mode_vol,"


@pytest.mark.parametrize(
    "text",
    [
        "fc_lo,voice(ad)",
        SONG + "voice(segments(sr,ad,ctrl):2,freq_lo,freq_hi,pw_lo,pw_hi),fc_lo",
        SONG + "voice(ad,sr,ctrl,ctrl:2,freq_lo,freq_hi,pw_lo,pw_hi)",
        SONG + "voice(ad,sr,ctrl:2,ctrl,freq_lo,freq_hi,pw_lo,pw_hi)",
        SONG + "voice(sr:2,ad,sr,ctrl,freq_lo,freq_hi,pw_lo,pw_hi)",
        SONG + "voice(segments(sr,ad,ctrl):2,freq_lo:2,freq_lo,freq_hi,pw_lo,pw_hi)",
        "fc_lo:2,fc_lo,fc_hi,res_filt,mode_vol,voice(segments(sr,ad,ctrl):2,"
        "freq_lo,freq_hi,pw_lo,pw_hi)",
        "fc_lo,fc_hi,res_filt,mode_vol,voice(ad,sr,ctrl:x,freq_lo,freq_hi,pw_lo,pw_hi)",
    ],
)
def test_parse_order_refuses(text):
    with pytest.raises(O.OrderError):
        O.parse_order(text)


@pytest.mark.parametrize(
    "stmts, message",
    [
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


def test_a_store_at_a_variable_sid_offset_is_split():
    text = (
        HEAD
        + "machine m { step { mem[L.sid + u16(tick & 7)] := tick; tick := mem[L.sid + 2]; } }\nschedule { m; }\n"
    )
    original = program(text)
    result = O.reorder(original, ORDER)
    assert result.after.sid_writes == 8
    assert O.check(original, result) is None


def gate(stmts):
    """A one-voice program running ``stmts`` on the SID registers of voice 0."""
    return HEAD + "machine m { step { " + stmts + " } }\nschedule { m; }\n"


def writes_of(text, order=ORDER):
    """``(original, result, original writes, reordered writes)`` of one call."""
    original = program(text)
    result = O.reorder(original, order)
    return (
        original,
        result,
        play(original, bytes(0x10000), 1)[1][0],
        play(result.program, bytes(0x10000), 1)[1][0],
    )


GATE_EDGES = gate(
    "chip[5] := 0x11; chip[6] := 0x22; chip[0] := 7; chip[4] := 0x41;"
    "chip[5] := 0x33; chip[6] := 0x44; chip[4] := 0x40; chip[1] := 8;"
)


def test_two_gate_edges_come_out_as_two_rounds():
    original, result, native, ordered = writes_of(GATE_EDGES)
    assert result.native == frozenset()
    assert ordered == [
        (6, 0x22),
        (5, 0x11),
        (4, 0x41),
        (6, 0x44),
        (5, 0x33),
        (4, 0x40),
        (0, 7),
        (1, 8),
    ]
    assert O.kept(native, result.slots) == ordered
    assert O.check(original, result) is None


def test_envelope_values_before_a_gate_on_land_before_its_ctrl():
    text = gate("chip[4] := 0x40; chip[5] := 0x55; chip[4] := 0x41; chip[6] := 0x66;")
    original, result, _native, ordered = writes_of(text)
    assert ordered == [(4, 0x40), (5, 0x55), (4, 0x41), (6, 0x66)]
    assert O.check(original, result) is None


def test_writes_that_leave_the_gate_alone_collapse_to_the_last():
    text = gate("chip[4] := 0x41; chip[4] := 0x21; chip[4] := 0x11; chip[5] := 3;")
    original, result, _native, ordered = writes_of(text)
    assert ordered == [(4, 0x41), (5, 3), (4, 0x11)]
    assert O.check(original, result) is None


def test_a_test_bit_toggle_ends_a_segment():
    text = gate("chip[4] := 0x09; chip[4] := 0x01; chip[4] := 0x09;")
    _original, _result, _native, ordered = writes_of(text)
    assert ordered == [(4, 0x09), (4, 0x01), (4, 0x09)]


PULSES = gate(
    "chip[5] := 1; chip[4] := 0x09; chip[4] := 0x01; chip[4] := 0x09;"
    "chip[4] := 0x01; chip[6] := 2; chip[4] := 0x09; chip[0] := 9;"
)


def test_repeated_test_pulses_keep_the_native_order():
    original, result, native, ordered = writes_of(PULSES)
    assert result.native == frozenset({4, 5, 6})
    assert ordered == [w for w in native if w[0] != 0] + [(0, 9)]
    assert O.kept(native, result.slots, result.native) == ordered
    assert O.check(original, result) is None


def test_a_loop_storing_ctrl_keeps_the_native_order():
    text = gate("for i in 0..8 { chip[4] := tick ^ 0x08; tick += 1; } chip[5] := 1;")
    original, result, native, ordered = writes_of(text)
    assert result.native == frozenset({4, 5, 6})
    assert ordered == native
    assert O.check(original, result) is None


def test_native_stores_are_proved_without_the_log():
    text = gate("for i in 0..300 { chip[4] := tick ^ 0x08; tick += 1; }")
    original = program(text)
    result = O.reorder(original, ORDER)
    with pytest.raises(O.OrderError, match="use prove"):
        O.check(original, result)
    assert outproof.prove(original, ORDER, timeout_ms=10_000).holds


def test_a_read_of_a_segmented_register_sees_its_last_store():
    text = gate("chip[4] := 0x41; chip[4] := 0x40; tick := chip[4]; chip[5] := tick;")
    original, result, native, ordered = writes_of(text)
    assert native[-1] == (5, 0x40) and (5, 0x40) in ordered
    assert O.check(original, result) is None


def observed(writes):
    """The gate edges of a call with the AD and SR in force at each and at its end."""
    ad = sr = gate_bit = edges = None
    out = []
    for reg, value in writes:
        if reg == 5:
            ad = value
        elif reg == 6:
            sr = value
        elif reg == 4:
            if (gate_bit, edges) == (None, None) or (value ^ edges) & 9:
                out.append((value & 9, ad, sr))
            gate_bit, edges = value & 1, value
    return out, (ad, sr, edges)


@pytest.mark.parametrize("seed", range(6))
def test_kept_commands_the_same_envelope(seed):
    rng = random.Random(seed)
    slots = O.parse_order(ORDER)
    for _ in range(300):
        trace = [
            (rng.choice((4, 4, 5, 6, 0)), rng.choice((0, 1, 8, 9, 0x41, 0x40, 0x21, 7)))
            for _ in range(rng.randrange(2, 7))
        ]
        if sum(r == 4 for r, _ in trace) >= 4:
            continue
        assert observed(O.kept(trace, slots)) == observed(trace)
