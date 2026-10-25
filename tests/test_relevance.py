"""relevance: control dependence, and keying on the departure the register depends on."""

import csv
import dataclasses
import hashlib

import numpy as np
import pytest

from pysidtracker import faultloc as fl
from pysidtracker import pathdiff as pd
from pysidtracker import paths
from pysidtracker import relevance as rv
from pysidtracker import survey
from pysidtracker.asm6502 import assemble
from pysidtracker.whydiff import CpuStep, PyEvent, cpu_slice

from .helpers import DATA_OFFSET, build_psid

LOOP = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "A"), ("D", "E")]
LOOP += [("E", rv.EXIT)]


def test_postdominators_of_a_diamond_in_a_loop():
    ipdom, branches = rv.postdominators(LOOP)
    assert branches == {"A", "D"}
    assert {n: ipdom[n] for n in "ABCDE"} == dict(A="D", B="D", C="D", D="E", E=-1)
    ipdom, _ = rv.postdominators([("X", "X"), ("Y", rv.EXIT)])
    assert "X" not in ipdom


def test_control_parents_follow_regions_loops_and_calls():
    ipdom, branches = rv.postdominators(LOOP)
    nodes = list("ABDACDE")
    got = rv.control_parents(nodes, [0] * 7, [False] * 7, ipdom, branches)
    assert got.tolist() == [-1, 0, -1, 2, 3, 2, -1]
    nodes = ["A", "x", "y", "B", "D", "E"]
    depth = [0, 1, 1, 0, 0, 0]
    leave = [False, False, True, False, False, False]
    got = rv.control_parents(nodes, depth, leave, ipdom, branches)
    assert got.tolist() == [-1, 0, 0, 0, -1, -1]


def _step(pc, raw, flow):
    return CpuStep(pc, bytes(raw), "", 0, 0, 0, 0xFF, 0, 0, flow=flow)


def test_cpu_control_keeps_the_guarding_branch_and_its_condition():
    steps = [
        _step(0x10, [0xA5, 0x02], {"A": frozenset({2}), "Z": frozenset({2})}),
        _step(0x12, [0xF0, 0x02], {"PC": frozenset({"Z"})}),
        _step(0x14, [0xA9, 0x05], {"A": frozenset()}),
        _step(0x16, [0x8D, 0x00, 0xD4], {0xD400: frozenset({"A"})}),
        _step(0x19, [0x60], {}),
    ]
    steps[3].writes = [(0xD400, 5)]
    control = rv.cpu_control()(steps)
    assert control.tolist() == [-1, -1, 1, -1, -1]
    assert cpu_slice(steps, 3)[0] == [2, 3]
    keep, live = cpu_slice(steps, 3, control=control)
    assert keep == [0, 1, 2, 3] and 2 in live
    assert rv.cpu_control()([]).size == 0


def test_next_lines_stay_in_their_call():
    kinds = ["call", "line", "call", "line", "return", "line", "return"]
    depth = [0, 0, 1, 1, 1, 0, 0]
    events = [PyEvent(k, d, "f", "p.py", 1, "") for k, d in zip(kinds, depth)]
    assert rv.next_lines(events).tolist() == [-1, 5, -1, -1, -1, -1, -1]


DECOY = (
    "mode:\n  .byte {mode}\ntrig:\n  .byte {trig}\nfthr:\n  .byte {fthr}\n"
    "cnt:\n  .byte 0\nflag:\n  .byte 0\n"
    "init:\n  LDA #0\n  STA cnt\n  LDA #$0F\n  STA $D418\n  RTS\n"
    "play:\n  LDA cnt\n  CMP fthr\n  BCC nofilt\n  LDA #1\n  STA flag\n  JMP root\n"
    "nofilt:\n  LDA #0\n  STA flag\n"
    "root:\n  LDA cnt\n  CMP trig\n  BCC early\n"
    "  LDA cnt\n  ASL A\n  CLC\n  ADC mode\n  STA $D400\n  JMP fin\n"
    "early:\n  LDA cnt\n  STA $D400\n"
    "fin:\n  INC cnt\n  RTS\n"
)

PLAYERS = '''
from pysidtracker.paths import PySubject
from pysidtracker.player import MemPlayer

LIFT_NAME = "<playroutine decoy>"
LIFT = """
def tick(regs, st):
    if st["cnt"] > st["fthr"]:
        st["flag"] = 1
    else:
        st["flag"] = 0
    if st["cnt"] > st["trig"]:
        regs[0] = (st["cnt"] * 2 + st["mode"]) & 0xFF
    else:
        regs[0] = st["cnt"]
    st["cnt"] += 1
"""


class Decoy(MemPlayer):
    def _init(self, subtune):
        self._wr(0x1003, 0)
        self._wr(0xD418, 0x0F)

    def _frame(self):
        cnt = self._rd(0x1003)
        if cnt {op} self._rd(0x1002):
            self._wr(0x1004, 1)
        else:
            self._wr(0x1004, 0)
        if cnt {op} self._rd(0x1001):
            self._wr(0xD400, (cnt * 2 + self._rd(0x1000)) & 0xFF)
        else:
            self._wr(0xD400, cnt)
        self._wr(0x1003, cnt + 1)


class Lift:
    def __init__(self, data):
        space = {{}}
        exec(compile(LIFT, LIFT_NAME, "exec"), space)
        self.tick = space["tick"]
        self.regs = [0] * 25
        self.regs[24] = 0x0F
        st = data[{off}:{off} + 3]
        self.st = dict(mode=st[0], trig=st[1], fthr=st[2], cnt=0, flag=0)

    def play_frame(self):
        self.tick(self.regs, self.st)
        return list(self.regs)


def native(data, song):
    return Decoy(data[{off}:], 0x1000, song - 1)


def lift(data, song):
    return PySubject(
        lambda: Lift(data),
        cell=lambda p, r: (p.regs, r),
        sources={{LIFT_NAME: LIFT}},
        label="lift",
        starts=lambda p: (0, None, None),
    )
'''

TUNES = [(16, t, t) for t in (3, 4, 5, 6, 7, 8)] + [(0, 0, f) for f in (2, 3, 4, 5)]


@pytest.fixture(name="decoy")
def _decoy(tmp_path):
    """A corpus whose failing tunes depart first at a branch only RAM depends on."""
    hvsc, work = tmp_path / "hvsc", tmp_path / "work"
    (hvsc / "D").mkdir(parents=True)
    work.mkdir()
    rows = []
    for i, (mode, trig, fthr) in enumerate(TUNES):
        code, lab = assemble(DECOY.format(mode=mode, trig=trig, fthr=fthr), 0x1000)
        data = build_psid(code, 0x1000, init=lab["init"], play=lab["play"])
        rel = f"D/t{i}.sid"
        (hvsc / rel).write_bytes(data)
        key = f"{hashlib.md5(data).hexdigest()}_1"
        verdict = "differs" if mode else "exact"
        rows.append(dict(key=key, path=rel, song=1, frames=30, verdict=verdict))
    with open(work / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    root = tmp_path / "players"
    root.mkdir()
    specs = {}
    for name, op in (("good", ">="), ("bad", ">")):
        path = root / f"{name}.py"
        path.write_text(PLAYERS.format(op=op, off=DATA_OFFSET))
        specs[name] = str(path)
    survey.load_render.cache_clear()
    return hvsc, work, specs


def _line(fragment: str) -> int:
    lines = PLAYERS.format(op=">", off=0).splitlines()
    return next(i for i, t in enumerate(lines, 1) if fragment in t and "if " in t)


def _where(spec: str, fragment: str) -> str:
    """The key of the ``if`` line holding ``fragment`` in the player at ``spec``."""
    return paths.source_of(spec).where(_line(fragment))


def _plain(an):
    """The partings with every departure counted, as before the relevance filter."""
    return fl.partings(dataclasses.replace(an.spec, relevant=None), an.corr)


def test_6502_pair_keys_on_the_root_not_the_decoy(decoy):
    hvsc, work, specs = decoy
    right = f"{specs['bad']}:native"
    pd.build(hvsc, work / "tunes.csv", work, pd.CPU, right, jobs=1)
    an = pd.analyse(work, pd.CPU, right, progress=False)
    root = _where(specs["bad"], "0x1001")
    lure = _where(specs["bad"], "0x1002")
    assert all(p.relevant for p in an.parts)
    assert all(f"{root}>" in p.right for p in an.parts)
    (fix,) = an.fixes
    assert fix.n_tunes == 6 and fix.key[0] == "first"
    assert f"{root}>" in fix.key[2] and "0x1001" in fix.labels[0]
    assert any(t.startswith("vs: $") and "BCC" in t for t in fix.labels)
    plain = _plain(an)
    assert all(f"{lure}>" in p.right and not p.relevant for p in plain)
    assert {p.left for p in plain}.isdisjoint(p.left for p in an.parts)


def test_python_pair_keys_on_the_root_not_the_decoy(decoy):
    hvsc, work, specs = decoy
    left, right = f"{specs['good']}:native", f"{specs['bad']}:lift"
    pd.build(hvsc, work / "tunes.csv", work, left, right, mode="onset", jobs=1)
    an = pd.analyse(work, left, right, progress=False)
    (fix,) = an.fixes
    assert fix.n_tunes == 6 and 'st["trig"]' in fix.key[2]
    root = _where(specs["good"], "0x1001")
    assert all(p.relevant and f"{root}>" in p.left for p in an.parts)
    plain = _plain(an)
    assert all('st["fthr"]' in p.right for p in plain)
    lure = _where(specs["good"], "0x1002")
    assert all(f"{lure}>" in p.left for p in plain)


def test_cache_holds_relevance(decoy):
    hvsc, work, specs = decoy
    right = f"{specs['bad']}:native"
    rec = pd.record((hvsc / "D/t0.sid").read_bytes(), 1, pd.CPU, right, "lead", 12)
    assert rec["fail"] and len(rec["relevant"])
    names = [rec["names"][i] for i in rec["relevant"]]
    assert all(n.split(":", 1)[1].startswith("edge:") for n in names)
    assert {n.split(":", 1)[0] for n in names} == {"6502", "python"}
    pd.build(hvsc, work / "tunes.csv", work, pd.CPU, right, jobs=1, limit=1)
    (got,) = pd.analyse(work, pd.CPU, right, progress=False).records.values()
    assert np.array_equal(got["relevant"], rec["relevant"])


BULK = """
from pysidtracker.player import MemPlayer


class Bulk(MemPlayer):
    def _init(self, subtune):
        self.work = [0] * 25
        self.n = 0

    def _frame(self):
        self.n += 1
        if self.n > 1:
            self.work[0] = self.n
        if self.n > 2:
            self.work[1] = self.n * 2
        out = self.work
        self._mem[0xD400 : 0xD419] = bytes(v & 0xFF for v in out)
"""


def test_python_slice_follows_one_item_of_a_bulk_copy(tmp_path):
    path = tmp_path / "bulk.py"
    path.write_text(BULK)
    space = {}
    exec(compile(BULK, str(path), "exec"), space)  # pylint: disable=exec-used

    def make():
        return space["Bulk"](bytes(1), 0x1000)

    texts = []
    for register in (0, 1):
        side = rv.python_side(make, 3, register, roots=[str(path)])
        texts.append({side.events[i].text for i in side.keep})
    assert "if self.n > 1:" in texts[0] and "if self.n > 2:" not in texts[0]
    assert "if self.n > 2:" in texts[1] and "if self.n > 1:" not in texts[1]
