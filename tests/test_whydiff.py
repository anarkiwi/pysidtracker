"""Tests for whydiff: the Python line trace, the 6502 history, and their dataflow."""

import json
import sys

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
    assert (why.frame, why.register, why.lead, why.cpu_lead) == (2, 0, 0, 0)
    assert (why.render_value, why.reference_value, why.cpu_value) == (0x10, 0x11, 0x11)
    assert why.cpu_agrees
    keep, live = why.py_slice()
    lines = [why.py[i] for i in keep]
    cond = next(e for e in lines if e.text.startswith("carry ="))
    assert cond.sets == [("carry", "False")]
    assert ("self.cnt", "2") in cond.reads
    assert live == {"self.cnt": "2", "self.threshold": "3"}
    texts = [why.cpu[i].text for i in why.cpu_dataflow()[0]]
    assert any(t.startswith("CMP") for t in texts)
    assert any(t.startswith("ADC") for t in texts)
    assert texts[-1].startswith("STA")
    cnt = LABELS["cnt"]
    assert why.cpu_dataflow()[1][cnt] == 2
    report = why.text(80)
    assert "WHY" in report and "CMP #$02" in report and "FULL HISTORIES" in report
    assert ">>" in report and "** SID $00 v1_freq_lo = $10" in report


def test_agreement_is_none_and_slices_skip_unwritten_registers():
    assert whydiff.explain_divergence(SID, lambda d: Faithful(b"", 0)) is None
    why = whydiff.explain_divergence(SID, _toy, frame=1, register=4)
    assert why.py_slice() == ([], {}) and why.cpu_dataflow() == ([], {})
    assert "(register not written)" in why.text()
    picked = whydiff.explain_divergence(SID, _toy, frame=2)
    assert picked.register == 0


def test_summary_is_json():
    why = whydiff.explain_divergence(SID, _toy)
    out = json.loads(json.dumps(why.summary()))
    assert out["register_name"] == "v1_freq_lo" and out["py_slice"]
    assert out["cpu"][out["cpu_slice"][-1]]["sid_writes"] == [[0, 0x11]]


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
    assert seen["song"] == 1 and seen["seconds"] == 2
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
    result.stdout.fnmatch_lines(["*WHY: the lines and instructions*"])
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


def test_trace_frame_skips_untraced_frames():
    events = whydiff.trace_frame(lambda: Toy(b"", 0), 2)
    writes = [e for e in events if e.kind == "write"]
    assert [(e.register, e.value) for e in writes] == [(0, 0x10)]
    assert writes[0].deps and events[writes[0].deps[0]].text.startswith("self._wr")
    assert all(e.path == __file__ for e in events if e.kind == "line")


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
    assert whydiff.player_roots(Toy(b"", 0)) == [str(__import__("tests").__path__[0])]
