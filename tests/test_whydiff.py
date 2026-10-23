"""Tests for whydiff: the Python line trace, the 6502 history, and their dataflow."""

import inspect
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pysidtracker import gridcmp, testing, whydiff
from pysidtracker.asm6502 import assemble
from pysidtracker.header import write_psid
from pysidtracker.player import MemPlayer

ORG = 0x1000

# CMP leaves C set from frame 2 on, and ADC #0 carries it into the stacked $D400 value.
TOY = (
    "init:\n  LDA #0\n  STA cnt\n  RTS\n"
    "play:\n  LDA tbl\n  PHA\n  LDA cnt\n  CMP #2\n  PLA\n  ADC #0\n  STA $D400\n"
    "  INC cnt\n  RTS\n"
    "tbl:\n  .byte $10\ncnt:\n  .byte $00\n"
)
CODE, LABELS = assemble(TOY, ORG)
SID = write_psid(load=ORG, init=LABELS["init"], play=LABELS["play"], image=CODE)


class Toy(MemPlayer):
    """A transcription of :data:`TOY`; ``threshold`` 2 is faithful, 3 drops a carry."""

    threshold = 3

    def _init(self, subtune):
        self.cnt = 0

    def _frame(self):
        carry = self.cnt >= self.threshold
        self._wr(0xD400, 0x10 + carry)
        self.cnt += 1


class Faithful(Toy):
    threshold = 2


def make_toy(data, song):  # pylint: disable=unused-argument
    """The ``module:callable`` CLI hook: the carry-dropping transcription."""
    return Toy(b"", 0)


def _toy(_data):
    return Toy(b"", 0)


def test_explains_the_dropped_carry():
    why = whydiff.explain_divergence(SID, _toy, name="toy")
    py, cpu = why.left, why.right
    assert (why.frame, why.register, py.frame, cpu.frame) == (2, 0, 2, 2)
    assert why.values == {"python": 0x10, "6502": 0x11, "reference": 0x11}
    assert "jennings agrees with the reference to here" in why.notes
    lines = [py.events[i] for i in py.keep]
    cond = next(e for e in lines if e.text.startswith("carry ="))
    assert cond.sets == [("carry", "False")]
    assert ("self.cnt", "2") in cond.reads
    assert py.live == {"self.cnt": "2", "self.threshold": "3"}
    assert py.events[py.root].text.startswith("self._mem[addr & 0xFFFF] = val")
    assert not any(py.events[i].kind == "return" for i in py.keep)  # _wr's own return
    texts = [cpu.events[i].text for i in cpu.keep]
    assert [t.split()[0] for t in texts] == [
        "LDA",
        "PHA",
        "LDA",
        "CMP",
        "PLA",
        "ADC",
        "STA",
    ]
    assert cpu.live == {
        "SP": "entry",
        f"${LABELS['cnt']:04X}": "$02",
        f"${LABELS['tbl']:04X}": "$10",
    }
    report = why.text(80)
    assert "WHY" in report and "CMP #$02" in report and "FULL HISTORIES" in report
    assert ">>" in report and "** SID $00 v1_freq_lo = $10" in report


def test_agreement_is_none_and_slices_skip_unwritten_registers():
    assert whydiff.explain_divergence(SID, lambda d: Faithful(b"", 0)) is None
    why = whydiff.explain_divergence(SID, _toy, frame=1, register=4)
    assert why.left.keep == why.right.keep == [] and why.left.root is None
    assert "(not stored on the traced frame)" in why.text()
    picked = whydiff.explain_divergence(SID, _toy, frame=2)
    assert picked.register == 0


def test_summary_is_json():
    why = whydiff.explain_divergence(SID, _toy)
    out = json.loads(json.dumps(why.summary()))
    assert out["register_name"] == "v1_freq_lo" and out["left"]["slice"]
    right = out["right"]
    assert right["events"][right["slice"][-1]]["sid_writes"] == [[0, 0x11]]


def test_assert_matches_reports(tmp_path):
    testing.assert_matches(SID, lambda d: Faithful(b"", 0), None)
    with pytest.raises(AssertionError, match="WHY"):
        testing.assert_matches(SID, _toy, None, name="A/toy.sid", report_dir=tmp_path)
    assert "CMP" in (tmp_path / "A_toy.txt").read_text()
    assert json.loads((tmp_path / "A_toy.json").read_text())["frame"] == 2


def test_assert_oracle_match(tmp_path, monkeypatch):
    tune = tmp_path / "toy.sid"
    tune.write_bytes(SID)
    seen = {}
    monkeypatch.setattr(testing, "resolve_tune", lambda rel, **kw: tune)

    def grid(path, **kw):
        seen.update(kw)
        return whydiff.register_grid(path.read_bytes(), kw["frames"])

    monkeypatch.setattr(testing, "oracle_grid", grid)
    with pytest.raises(AssertionError, match="toy.sid"):
        testing.assert_oracle_match(
            "A/toy.sid", _toy, hvsc_cache=tmp_path, oracle_cache=tmp_path, frames=8
        )
    assert seen["song"] == 1 and seen["seconds"] is None and seen["frames"] == 8
    monkeypatch.setattr(testing, "resolve_tune", lambda rel, **kw: None)
    with pytest.raises(testing.TuneFetchError):
        testing.assert_oracle_match(
            "A/toy.sid", _toy, hvsc_cache=tmp_path, oracle_cache=tmp_path
        )


def test_oracle_fixture_with_player_explains(pytester):
    pytester.makepyfile(f"""
        from pathlib import Path

        from pysidtracker import testing, whydiff
        from pysidtracker.player import MemPlayer

        import pytest

        SID = {SID!r}

        @pytest.fixture(autouse=True)
        def _offline(monkeypatch):
            tune = Path("toy.sid")
            tune.write_bytes(SID)
            monkeypatch.setattr(testing, "resolve_tune", lambda rel, **kw: tune)
            monkeypatch.setattr(
                testing, "oracle_grid", lambda path, **kw: whydiff.register_grid(SID, 6)
            )

        class Toy(MemPlayer):
            def _init(self, subtune):
                self.cnt = 0

            def _frame(self):
                self._wr(0xD400, 0x10 + (self.cnt >= 3))
                self.cnt += 1

        tune_id, oracle_match = testing.make_oracle_fixtures(
            {{"toy": "A/toy.sid"}}, hvsc_cache=".", oracle_cache=".",
            player=lambda data: Toy(b"", 0), max_lead=0, report_dir="why",
        )

        def test_match(oracle_match):
            oracle_match()
        """)
    result = pytester.runpytest_inprocess("-p", "no:cacheprovider")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*WHY: the steps the diverging value came from*"])
    assert (pytester.path / "why" / "A_toy.txt").exists()


def _sample(a, b=2):
    total = 0
    for i in range(a):
        if i % b:
            total += i
        else:
            total -= 1
    items = [total, a]
    items[0] = items[0] + 1
    return _leaf(items[0])


def _leaf(x):
    return x * 2


def _boom():
    raise ValueError("bad")


def test_pytrace_records_lines_branches_values_and_calls():
    before = sys.gettrace()
    with whydiff.PyTrace([__file__]) as trace:
        assert _sample(3) == 0
        with pytest.raises(ValueError):
            _boom()
    assert sys.gettrace() is before
    ev = trace.events
    calls = [e for e in ev if e.kind == "call"]
    assert calls[0].func.endswith("_sample") and ("a", "3") in calls[0].reads
    leaf = next(e for e in calls if e.func.endswith("_leaf"))
    assert leaf.depth == calls[0].depth + 1
    conds = [e.cond for e in ev if e.text == "if i % b:"]
    assert conds == [False, True, False]
    aug = [e for e in ev if e.text == "total += i"]
    assert aug[0].sets == [("total", "0")]
    item = next(e for e in ev if e.text.startswith("items[0] ="))
    assert item.sets == [("items[0]", "0")] and ("items[0]", "-1") in item.reads
    ret = [e for e in ev if e.kind == "return" and e.func.endswith("_sample")]
    assert ret[0].sets == [("", "0")]
    assert any(e.kind == "exception" and "ValueError: bad" in e.text for e in ev)
    root = ev.index(ret[0])
    keep = trace.slice(root)
    assert keep[-1] == root and any(ev[i].func.endswith("_leaf") for i in keep)
    assert any(ev[i].text == "total -= 1" for i in keep)


def test_python_side_skips_untraced_frames():
    side = whydiff.python_side(lambda: Toy(b"", 0), 2, 0, progress=True)
    events = side.events
    writes = [e for e in events if e.kind == "write"]
    assert [(e.register, e.value) for e in writes] == [(0, 0x10)]
    assert events[writes[0].site].text.startswith("self._wr")
    assert {e.func for e in events if e.kind == "call"} >= {
        "MemPlayer.play_frame",
        "Toy._frame",
    }


# A generic player in the style of a code-generating VM: no MemPlayer, its register file
# a list filled by slice assignment in exec'd code.
GENERIC = (
    "def tick(p):\n"
    "    lift = 1 if p.n >= p.at else 0\n"
    "    out = list(p.regs)\n"
    "    out[0:2] = [0x10 + lift, 0]\n"
    "    p.regs = out\n"
    "    p.n += 1\n"
)
GENERIC_NAME = "<generic tick>"


class Generic:
    """Plays :data:`GENERIC`; ``at`` is the tick its carry first lifts the register."""

    def __init__(self, at):
        self.n, self.at, self.regs = 0, at, [0] * 25
        space = {}
        exec(compile(GENERIC, GENERIC_NAME, "exec"), space)  # pylint: disable=exec-used
        self._tick = space["tick"]

    def play_frame(self):
        self._tick(self)

    def render_grid(self, n):
        rows = []
        for _ in range(n):
            self.play_frame()
            rows.append(list(self.regs))
        return rows


def test_two_python_players_side_by_side():
    def side(at, label):
        return whydiff.python_side(
            lambda: Generic(at),
            2,
            0,
            label=label,
            cell=lambda p, r: (p.regs, r),
            roots=[GENERIC_NAME],
            sources={GENERIC_NAME: GENERIC},
        )

    generic, native = side(3, "generic"), side(2, "native")
    grids = (Generic(3).render_grid(6), Generic(2).render_grid(6))
    why = whydiff.explain_cell(generic, native, grids, frame=2, register=0, name="toy")
    assert why.values == {"generic": 0x10, "native": 0x11}
    for s, lift in ((generic, "0"), (native, "1")):
        texts = [s.events[i].text for i in s.keep]
        assert texts == [
            "lift = 1 if p.n >= p.at else 0",
            "out[0:2] = [0x10 + lift, 0]",
        ]
        assert s.events[s.keep[0]].sets == [("lift", lift)]
        assert s.live == {
            "p.n": "2",
            "p.at": generic.live["p.at"] if s is generic else "2",
        }
    text = why.text(70)
    assert text.count("GENERIC") == 2 and "out[0:2] = [0x10 + lift, 0]" in text


def test_repr_and_line_formatting():
    rep = whydiff._Repr()  # pylint: disable=protected-access
    obj = object()

    class Plain:
        pass

    plain = Plain()
    assert rep.repr(5) == "5" and rep.repr(31) == "31/$1F" and rep.repr(True) == "True"
    assert rep.repr(np.int64(255)) == "255/$FF" and rep.repr(1.5) == "1.5"
    assert rep.repr(bytes(range(14))).startswith(
        "bytes[14] 00 01"
    ) and "..." in rep.repr(bytes(14))
    assert rep.repr(np.zeros((2, 5))).startswith("ndarray[2, 5][0.0")
    assert rep.repr(plain) == "<Plain#0>" and rep.repr(plain) == "<Plain#0>"
    assert rep.repr(obj).startswith("<object")
    assert rep.repr(None) == "None" and rep.repr(gridcmp.Verdict()) == "<Verdict#2>"
    line = whydiff.PyEvent("line", 1, "f", "/x/a.py", 7, "x = y", [("y", "1")])
    line.sets = [("x", "1")]
    assert whydiff.py_line(line) == "      7  x = y  | y=1  := x=1"
    exc = whydiff.PyEvent("exception", 0, "f", "/x/a.py", 7, "raise E: m")
    assert whydiff.py_line(exc) == "!! raise E: m  [a.py:7]"


def test_cpu_slice_follows_the_stack_and_carry():
    steps = whydiff.cpu_history(SID, 2)
    by = {s.text.split()[0]: s for s in steps}
    cell = 0x100 + by["PHA"].sp + 1
    assert by["PHA"].flow[cell] == {"A"} and cell in by["PLA"].flow["A"]
    mirror = whydiff.CpuStep(0, b"\x8d", "STA", 0, 0, 0, 0xFF, 0, 4, [], [(0xD420, 1)])
    assert mirror.sid_writes() == [(0, 1)]
    root = next(i for i, s in enumerate(steps) if s.sid_writes())
    keep, live = whydiff.cpu_slice(steps, root)
    assert [steps[i].text.split()[0] for i in keep] == [
        "LDA",
        "PHA",
        "LDA",
        "CMP",
        "PLA",
        "ADC",
        "STA",
    ]
    assert live == {"SP": None, LABELS["cnt"]: 2, LABELS["tbl"]: 0x10}


def test_cli(tmp_path, capsys):
    tune = tmp_path / "toy.sid"
    tune.write_bytes(SID)
    out = tmp_path / "why.json"
    assert (
        whydiff.main(
            [str(tune), f"{__name__}:make_toy", "--frames", "6", "--json", str(out)]
        )
        == 1
    )
    assert "divergence: toy.sid" in capsys.readouterr().out
    assert json.loads(out.read_text())["frame"] == 2
    ref = tmp_path / "ref.csv.zst"
    gridcmp.write_grid(np.full((6, 25), 0, np.int16), ref)
    assert (
        whydiff.main(
            [
                str(tune),
                f"{__name__}:make_toy",
                "--frames",
                "2",
                "--reference",
                str(ref),
            ]
        )
        == 1
    )
    assert whydiff.main([str(tune), f"{__name__}:make_toy", "--frames", "2"]) == 0
    assert "verified" in capsys.readouterr().out


def _probe(a):
    b = a + 1
    yield sys._getframe()  # pylint: disable=protected-access
    c = b * 2
    yield c


def _wr(self, addr, value):  # pylint: disable=unused-argument
    return sys._getframe()  # pylint: disable=protected-access


def test_trace_callbacks_by_hand():
    """The callbacks run where coverage cannot see them, so drive them directly."""
    gen = _probe(3)
    frame = next(gen)
    trace = whydiff.PyTrace([__file__], [_wr.__code__])
    assert (
        trace._global(frame, "call", None) == trace._local
    )  # pylint: disable=protected-access
    trace._local(frame, "line", None)  # pylint: disable=protected-access
    trace._local(frame, "line", None)  # pylint: disable=protected-access
    next(gen)
    trace._local(frame, "line", None)  # pylint: disable=protected-access
    trace._local(
        frame, "exception", (ValueError, ValueError("x"), None)
    )  # pylint: disable=protected-access
    trace._local(frame, "return", 8)  # pylint: disable=protected-access
    kinds = [e.kind for e in trace.events]
    assert kinds == ["call", "line", "line", "exception", "return"]
    assert trace.events[0].reads == [("a", "3")]
    assert trace.events[1].text.startswith("yield sys._getframe()")
    assert trace.events[2].text == "yield c" and trace.events[2].live == [("c", "8")]
    assert trace.events[4].deps == [2] and trace.events[4].sets == [("", "8")]
    trace._global(
        _wr(None, 0xD404, 0x41), "call", None
    )  # pylint: disable=protected-access
    trace._global(
        _wr(None, 0x1000, 1), "call", None
    )  # pylint: disable=protected-access
    trace._global(_wr(None, "x", 1), "call", None)  # pylint: disable=protected-access
    writes = [e for e in trace.events if e.kind == "write"]
    assert [(e.register, e.value) for e in writes] == [(4, 0x41)]
    assert (
        trace._global(sys._getframe(), "call", None) is not None
    )  # pylint: disable=protected-access
    other = whydiff.PyTrace(["/nonexistent"])
    assert (
        other._global(frame, "call", None) is None
    )  # pylint: disable=protected-access


def test_statement_index(tmp_path):
    src = tmp_path / "m.py"
    src.write_text(
        "import os\n"
        "def f(x, *a, k=1, **kw):\n"
        "    if (x >\n"
        "            1):\n"
        "        y = [x][0]\n"
        "    for i in range(x):\n"
        "        x += i\n"
        "    with open(os.devnull) as fh:\n"
        "        pass\n"
        "    try:\n"
        "        z: int = x\n"
        "    except ValueError:\n"
        "        pass\n"
        "    while x:\n"
        "        x -= 1\n"
        "    match x:\n"
        "        case 0:\n"
        "            pass\n"
        "    assert x == 0\n"
        "    t = x[1:2]\n"
        "    a, (b, c) = 1, (2, 3)\n"
        "    return f(x)\n"
    )
    index = whydiff._index(str(src))  # pylint: disable=protected-access
    assert index[3] is index[4] and index[3].text == "if (x > 1):" and index[3].cond
    assert index[5].control == 3 and index[7].control == 6 and index[3].control is None
    assert [t for t, _, _ in index[5].sets] == ["y"]
    assert [t for t, _, _ in index[6].sets] == ["i"]
    assert [t for t, _, _ in index[8].sets] == ["fh"]
    assert [t for t, _, _ in index[11].sets] == ["z"]
    assert index[19].cond is not None and index[16].reads
    assert [t for t, _, _ in index[20].reads] == ["x[1:2]"]
    assert [t for t, _, _ in index[21].sets] == ["a", "b", "c"]
    assert [t for t, _, _ in index[22].reads] == ["x"]
    bad = tmp_path / "bad.py"
    bad.write_text("def (:\n")
    assert whydiff._index(str(bad)) == {}  # pylint: disable=protected-access
    roots = whydiff.player_roots(Toy(b"", 0))
    assert roots == sorted(
        [str(__import__("tests").__path__[0]), inspect.getsourcefile(MemPlayer)]
    )


class Sparse(Toy):
    """Stores $D400 on even frames only, so odd frames carry the last value."""

    def _frame(self):
        if self.cnt % 2 == 0:
            self._wr(0xD400, self.cnt)
        self.cnt += 1


SPARSE = (
    "init:\n  LDA #0\n  STA cnt\n  RTS\n"
    "play:\n  LDA cnt\n  AND #1\n  BNE skip\n  LDA cnt\n  STA $D400\n"
    "skip:\n  INC cnt\n  RTS\n"
    "cnt:\n  .byte $00\n"
)


def test_a_side_without_a_store_traces_its_last_store():
    grid = Sparse(b"", 0).render_grid(6)
    side = whydiff.python_side(lambda: Sparse(b"", 0), 3, 0, grid=grid)
    assert (side.frame, side.traced) == (3, 2) and side.root is not None
    assert whydiff.python_side(lambda: Sparse(b"", 0), 3, 0).root is None
    code, labels = assemble(SPARSE, ORG)
    sid = write_psid(load=ORG, init=labels["init"], play=labels["play"], image=code)
    jen = whydiff.register_grid(sid, 6)
    cpu = whydiff.cpu_side(sid, 3, 0, grid=jen)
    assert (cpu.frame, cpu.traced) == (3, 2) and cpu.events[cpu.root].text.startswith(
        "STA"
    )
    ram = whydiff.cpu_side(sid, 2, 0, address=labels["cnt"])
    assert [ram.events[i].text.split()[0] for i in ram.keep] == ["INC"]
    assert ram.live == {f"${labels['cnt']:04X}": "$02"}
    never = whydiff.cpu_side(sid, 3, 4, grid=jen)
    assert never.root is None and never.traced == 3
    assert (
        whydiff.last_change(jen, 5, 0) == 4 and whydiff.last_change(jen, 1, 4) is None
    )
    left = whydiff.python_side(lambda: Sparse(b"", 0), 3, 0, grid=grid)
    why = whydiff.explain_cell(left, cpu, (grid, jen), frame=3, register=0)
    assert "python frame 3 (traced 2: its last store)" in why.notes[0]


def _rows(rows):
    total = 0
    for v in rows:
        total += v[1]
    grid = np.zeros((2, 3))
    first = grid[:, 1]
    return total, first


def test_earlier_items_are_state_and_2d_slices_trace():
    rows = [[0, 5], [0, 7]]
    with whydiff.PyTrace([__file__]) as trace:
        _rows(rows)
    adds = [e for e in trace.events if e.text == "total += v[1]"]
    assert [e.live for e in adds] == [[("v[1]", "5")], [("v[1]", "7")]]
    first = next(e for e in trace.events if e.text == "first = grid[:, 1]")
    assert first.sets and first.sets[0][0] == "first"


class GenericToy(Generic):
    """:data:`GENERIC` as a whole tune's player: the 6502 lifts the register a frame earlier."""

    def __init__(self, _data=b""):
        super().__init__(3)
        self.regs[24] = 0x0F


def test_explain_divergence_takes_an_execd_player():
    why = whydiff.explain_divergence(
        SID,
        GenericToy,
        frame=2,
        register=0,
        label="generic",
        cell=lambda p, r: (p.regs, r),
        roots=[GENERIC_NAME],
        sources={GENERIC_NAME: GENERIC},
    )
    assert why.values == {"generic": 0x10, "6502": 0x11, "reference": 0x11}
    texts = [why.left.events[i].text for i in why.left.keep]
    assert texts[-1] == "out[0:2] = [0x10 + lift, 0]"


# The gate cell is never stored by the 6502; frame 3 outputs $41 EOR gate.
GATED = (
    "init:\n  LDA #0\n  STA cnt\n  RTS\n"
    "play:\n  INC cnt\n  LDA cnt\n  CMP #4\n  BNE out\n  LDA #$41\n  EOR gate\n"
    "  STA $D404\nout:\n  RTS\n"
    "cnt:\n  .byte $00\ngate:\n  .byte $00\n"
)
GCODE, GLABELS = assemble(GATED, ORG)
GSID = write_psid(load=ORG, init=GLABELS["init"], play=GLABELS["play"], image=GCODE)


class Gated(MemPlayer):
    """:data:`GATED` with a model state bug from ``bug`` on: frame 1 sets the gate."""

    bug = True

    def _init(self, subtune):
        self.cnt = self.gate = 0

    def _frame(self):
        self.cnt += 1
        if self.cnt == 2 and self.bug:
            self.gate = 1
        if self.cnt == 4:
            self._wr(0xD404, 0x41 ^ self.gate)


class Ungated(Gated):
    bug = False


def test_state_is_chased_to_where_it_parts():
    why = whydiff.explain_divergence(
        GSID, lambda d: Gated(b"", 0), frames=8, state=True
    )
    assert (why.frame, why.register) == (3, 4)
    state = why.state
    assert state.frames == (0, 2)
    assert ("player.gate", f"${GLABELS['gate']:04X}", 1) in state.pairs
    assert ("player.cnt", f"${GLABELS['cnt']:04X}", None) in state.pairs
    onset = state.why
    assert onset.frame == 1 and onset.cell.startswith("state player.gate ~ $")
    assert onset.values == {"python": 1, "6502": 0}
    left = onset.left
    assert left.events[left.root].text == "self.gate = 1"
    assert onset.right.root is None and onset.right.traced == 1
    text = why.text(80)
    assert "parts at frame 1" in text and "STATE ONSET" in text
    assert json.loads(json.dumps(why.summary()))["state"]["why"]["frame"] == 1


def test_state_between_two_python_players_and_its_edges():
    grids = (Gated(b"", 0).render_grid(6), Ungated(b"", 0).render_grid(6))
    left = whydiff.python_side(lambda: Gated(b"", 0), 3, 4, grid=grids[0])
    right = whydiff.python_side(
        lambda: Ungated(b"", 0), 3, 4, label="native", grid=grids[1]
    )
    why = whydiff.explain_cell(left, right, grids, frame=3, register=4)
    state = whydiff.explain_state(why)
    assert ("player.gate", "player.gate", 1) in state.pairs
    assert state.why.right.root is None and state.why.left.root is not None
    agree = whydiff.explain_cell(right, right, grids[::-1], frame=3, register=4)
    assert whydiff.explain_state(agree).why is None
    first = whydiff.python_side(lambda: Gated(b"", 0), 0, 4)
    alone = whydiff.explain_cell(first, first, grids, frame=0, register=4)
    assert whydiff.explain_state(alone) is None


def test_first_break_and_object_paths():
    assert whydiff.first_break([0, 1, 2], [0, 1, 3]) == 2
    assert whydiff.first_break([0, 1, 0], [0xFF, 0, 0xFF]) == 3
    assert whydiff.first_break([0, 1, 2], [5, 6, 6]) == 2
    assert whydiff.first_break([1, 1], [1, 1]) == 2
    player = Gated(b"", 0)
    box = {"k": [player]}
    assert whydiff.object_path(box, player) == (("S", "k"), ("S", 0))
    assert whydiff.object_path(player, player) == ()
    assert whydiff.object_path(player, object()) is None
    assert (
        whydiff.path_name((("A", "voices"), ("S", 2), ("A", "gate")))
        == "player.voices[2].gate"
    )


def test_cli_format_and_state(tmp_path, capsys, monkeypatch):
    from pysidtracker import formats

    tune = tmp_path / "gated.sid"
    tune.write_bytes(GSID)
    parsed = []

    def parse(data, subtune):
        parsed.append(subtune)
        return data

    fmt = formats.SidFormat(
        name="gated",
        parser=SimpleNamespace(parse=parse, recognize=lambda image: True),
        player=lambda model: Gated(b"", 0),
    )
    monkeypatch.setattr(formats, "discover_formats", lambda: [fmt])
    argv = [str(tune), "--format", "gated", "--frames", "8", "--song", "2", "--state"]
    assert whydiff.main(argv) == 1
    assert "STATE ONSET" in capsys.readouterr().out
    assert whydiff.main([str(tune), "--format", "auto", "--frames", "8"]) == 1
    assert set(parsed) == {1, 0}
    with pytest.raises(SystemExit):
        whydiff.main([str(tune), "--format", "nope"])
    with pytest.raises(SystemExit):
        whydiff.main([str(tune)])


# Three voices' gate cells at a stride of 7; voice 3's is never stored, as is the decoy.
VOICED = (
    "init:\n  LDA #0\n  STA cnt\n  RTS\n"
    "play:\n  INC cnt\n  LDA decoy\n  LDA cnt\n  AND #1\n  STA g0\n  STA g1\n"
    "  LDA cnt\n  CMP #6\n  BNE out\n  LDA #$41\n  EOR g2\n  EOR decoy\n  STA $D412\n"
    "out:\n  RTS\n"
    "decoy:\n  .byte 0\ncnt:\n  .byte 0\n"
    "g0:\n  .byte 0, 0, 0, 0, 0, 0, 0\ng1:\n  .byte 0, 0, 0, 0, 0, 0, 0\ng2:\n  .byte 0\n"
)
VCODE, VLABELS = assemble(VOICED, ORG)
VSID = write_psid(load=ORG, init=VLABELS["init"], play=VLABELS["play"], image=VCODE)


class Voiced(MemPlayer):
    """:data:`VOICED` whose voice 3 gate goes stale from frame 1."""

    def _init(self, subtune):
        self.cnt = 0
        self.voices = [SimpleNamespace(gate=0) for _ in range(3)]

    def _frame(self):
        self.cnt += 1
        for v in self.voices[:2]:
            v.gate = self.cnt & 1
        if self.cnt == 2:
            self.voices[2].gate = 1
        if self.cnt == 6:
            self._wr(0xD412, 0x41 ^ self.voices[2].gate)


def test_a_list_member_pairs_with_its_strided_cell():
    why = whydiff.explain_divergence(
        VSID, lambda d: Voiced(b"", 0), frames=10, state=True
    )
    assert (why.frame, why.register) == (5, 0x12)
    pairs = {a: (b, f) for a, b, f in why.state.pairs}
    assert pairs["player.voices[2].gate"] == (f"${VLABELS['g2']:04X}", 1)
    assert VLABELS["g2"] - VLABELS["g1"] == VLABELS["g1"] - VLABELS["g0"] == 7
    ram = np.zeros((4, 0x10000), np.int64)
    ram[:, 0x100], ram[:, 0x107] = [1, 0, 1, 0], [0, 1, 0, 1]
    sib = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]])
    assert whydiff.fit_family(sib, 2, ram, [0x10E, 0x200]) == 0x10E


class Named:
    """A generic player whose commands set a voice's registers by name, as a document
    player's ``!ad``/``!sr`` do, and whose register file reads them back."""

    def __init__(self):
        self.voice = SimpleNamespace(ad=0, sr=0, wave=0)
        self.regs = [0] * 7
        self.level, self.tick = 0x40, 0
        self.script = {1: [("ad", 0x0A)], 2: [("sr", 3), ("wave", 1)]}

    def _command(self, key, value):
        level = self.level + value
        setattr(self.voice, key, level & 0xFF)

    def play_frame(self):
        for key, value in self.script.get(self.tick, ()):
            if key == "wave":
                vars(self.voice)[key] = value
                self.voice.__dict__["wave"] += getattr(self.voice, key)
            else:
                self._command(key, value)
        self.regs[5] = self.voice.ad
        self.regs[6] = getattr(self.voice, "sr")
        self.regs[4] = self.voice.wave
        self.tick += 1


def _named(frame, register, cell):
    side = whydiff.python_side(Named, frame, register, cell=cell, roots=[__file__])
    return side, [side.events[i] for i in side.keep]


def test_a_store_through_setattr_roots_a_slice_to_its_source():
    for frame, register, attr, value in (
        (1, 5, "ad", "74/$4A"),
        (2, 6, "sr", "67/$43"),
    ):
        side, keep = _named(frame, register, lambda p, _, a=attr: (p.voice, a))
        assert keep[-1].text == "setattr(self.voice, key, level & 0xFF)"
        assert keep[-1].sets == [("getattr(self.voice, key)", value)]
        assert any(e.text == "level = self.level + value" for e in keep)
        assert side.live["self.level"] == "64/$40"
        _, keep = _named(frame, register, lambda p, r: (p.regs, r))
        assert keep[-1].text.startswith(f"self.regs[{register}] =")
        assert "setattr(self.voice, key, level & 0xFF)" in [e.text for e in keep]


def test_vars_and_dict_items_store_the_attribute():
    _, keep = _named(2, 4, lambda p, _: (p.voice, "wave"))
    assert [e.text for e in keep][-2:] == [
        "vars(self.voice)[key] = value",
        'self.voice.__dict__["wave"] += getattr(self.voice, key)',
    ]
    assert keep[-1].sets == [("self.voice.wave", "2")]


def test_dynamic_attributes_in_the_statement_index(tmp_path):
    src = tmp_path / "d.py"
    src.write_text(
        "def f(o, k, v):\n"
        "    setattr(o, k, v)\n"
        "    setattr(o, 'ad', v)\n"
        "    delattr(o, k)\n"
        "    del o.x, v\n"
        "    vars(o)[k] = getattr(o, k, 0) + o.__dict__[k]\n"
        "    setattr(o, 'class', v)\n"
        "    setattr(*v)\n"
        "    [setattr(o, k, i) for i in v]\n"
        "    setattr(o.f(), k, v)\n"
    )
    index = whydiff._index(str(src))  # pylint: disable=protected-access
    sets = {n: [t for t, _, _ in index[n].sets] for n in index}
    assert sets[2] == ["getattr(o, k)"] and sets[3] == ["o.ad"]
    assert sets[4] == ["getattr(o, k)"] and sets[5] == ["o.x"]
    assert sets[6] == ["getattr(o, k)"]
    assert [t for t, _, _ in index[6].reads] == ["getattr(o, k)", "k", "o"]
    assert sets[7] == ["getattr(o, 'class')"]
    assert sets[8] == sets[9] == sets[10] == []
    assert index[2].sets[0][2][0] == "A"


def test_dynamic_stores_by_hand():
    """A dynamic name is evaluated in the frame; a deleted attribute is still a store."""
    obj = SimpleNamespace(ad=1)
    gen = _dyn(obj, "ad")
    frame = next(gen)
    trace = whydiff.PyTrace([__file__])
    trace._global(frame, "call", None)  # pylint: disable=protected-access
    for _ in range(4):
        next(gen)
        trace._local(frame, "line", None)  # pylint: disable=protected-access
    assert [e.text for e in trace.events[1:]] == [
        "yield setattr(o, k, 7)",
        "yield o",
        "yield delattr(o, k)",
        "yield o",
    ]
    assert trace.events[1].sets == [("getattr(o, k)", "7")]
    assert trace.events[3].sets == [("getattr(o, k)", "<unset>")]
    assert trace.store(obj, "ad") == 3
    spec = ("A", compile("o", "", "eval"), compile("1", "", "eval"))
    assert (
        trace._keys(frame, spec, frame.f_locals) == []
    )  # pylint: disable=protected-access


def _dyn(o, k):
    yield sys._getframe()  # pylint: disable=protected-access
    yield setattr(o, k, 7)
    yield o
    yield delattr(o, k)
    yield o
