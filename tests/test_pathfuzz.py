"""Tests for pathfuzz: patches, coverage gaps, cmin, directed mutants and ddmin."""

import inspect
import itertools
import json

import numpy as np
import pytest

from pysidtracker import pathfuzz as pf
from pysidtracker import paths
from pysidtracker.asm6502 import assemble
from pysidtracker.image import SidImage
from pysidtracker.player import MemPlayer

from .helpers import build_prg, build_psid, build_psid_embedded_load

ORG = 0x1000
TUNE = (
    "init:\n  RTS\n"
    "play:\n  LDA tbl\n  CMP #$10\n  BCS hi\n  LDA #$11\n  STA $D400\n  RTS\n"
    "hi:\n  LDA #$22\n  STA $D401\n  RTS\n"
    "pad:\n  .byte $01, $02, $03\ntbl:\n  .byte $05\n"
)
CODE, LAB = assemble(TUNE, ORG)
SID = build_psid(CODE, ORG, init=LAB["init"], play=LAB["play"])
TBL = pf.addr_to_offset(SID, LAB["tbl"])
PAD = [pf.addr_to_offset(SID, LAB["pad"] + i) for i in range(3)]


class Toy(MemPlayer):
    """A transcription of :data:`TUNE` writing $23, not $22, on the high outcome."""

    def _init(self, subtune):
        pass

    def _frame(self):
        if self._rd(LAB["tbl"]) >= 0x10:
            self._wr(0xD401, 0x23)
        else:
            self._wr(0xD400, 0x11)


def make_right(data, song):  # pylint: disable=unused-argument
    """The ``--right`` hook: :class:`Toy` over ``data``'s image."""
    image = SidImage.from_bytes(data)
    return Toy(image.image, image.load)


def _high(value=0x20):
    return pf.apply_patch(SID, [[TBL, value]])


def test_patch_round_trip_and_offsets():
    data = build_psid_embedded_load(CODE, ORG, init=LAB["init"], play=LAB["play"])
    off = pf.addr_to_offset(data, LAB["tbl"])
    assert data[off] == 0x05 and off == 0x7C + 2 + LAB["tbl"] - ORG
    assert pf.addr_to_offset(data, ORG) == 0x7E
    assert pf.addr_to_offset(data, ORG - 1) is None
    assert pf.addr_to_offset(data, ORG + len(CODE)) is None
    assert pf.addr_to_offset(build_prg(CODE, ORG), ORG + 3) == 5
    mutated = pf.apply_patch(data, [[off, 0x1FF], [3, 7]])
    patch = pf.make_patch(data, mutated)
    assert patch == [[3, 7], [off, 0xFF]]
    assert pf.apply_patch(data, patch) == mutated
    assert pf.make_patch(data, data) == []
    with pytest.raises(ValueError):
        pf.make_patch(data, data[:-1])


def test_gaps_and_covered():
    trace = paths.cpu_paths(SID, 3)
    universe = paths.cpu_universe(SID)
    missing = pf.gaps(trace, universe)
    assert missing and set(missing) == set(universe) - set(trace.names)
    high = paths.cpu_paths(_high(), 3)
    both = pf.gaps([trace, high], universe)
    assert set(missing) - set(both) == (pf.covered(high) - set(trace.names)) & set(
        universe
    )
    assert len(both) == len(missing) - 2
    assert pf.covered(["a", ["b", "c"]]) == {"a", "b", "c"}
    assert pf.covered("a") == {"a"}


def _brute_cover(sets):
    feats = set().union(*sets.values())
    for k in range(len(sets) + 1):
        for combo in itertools.combinations(sorted(sets), k):
            if set().union(*(sets[t] for t in combo)) == feats:
                return k
    raise AssertionError


@pytest.mark.parametrize("seed", range(6))
def test_distil_is_a_minimum_cover(seed):
    rng = np.random.default_rng(seed)
    sets = {
        f"t{i}": {f"f{j}" for j in np.flatnonzero(rng.random(12) < 0.25)}
        for i in range(7)
    }
    got = pf.distil(sets)
    assert set().union(*(sets[t] for t in got)) == set().union(*sets.values())
    assert len(got) == _brute_cover(sets)


def test_py_arcs_key_like_the_universe():
    trace = paths.py_paths(lambda: make_right(SID, 1), 2)
    arcs = pf.py_arcs(trace.roots, trace.sources)
    assert sorted(arcs) == paths.py_universe(trace.roots, trace.sources)
    head = inspect.getsourcelines(Toy._frame)[1] + 1
    where = arcs[f"edge:tests/test_pathfuzz.py:Toy._frame:{head}>{head + 1}"]
    assert where == f"tests/test_pathfuzz.py:Toy._frame:{head}"
    generated = {"<gen>": "def f(a):\n    if a:\n        return 1\n    return 2\n"}
    assert pf.py_arcs(["<gen>"], generated) == {
        "edge:<gen>:f:if a:>return #": "<gen>:f:if a:",
        "edge:<gen>:f:if a:>return #@1": "<gen>:f:if a:",
    }


def test_values_and_branch_slice():
    assert pf.values(5, [("CMP", "imm", 0x10)])[:3] == [0x0F, 0x10, 0x11]
    assert pf.values(5, [("AND", "imm", 0x0A)])[:3] == [0x0F, 0x07, 0x0D]
    assert pf.values(0, [("BIT", "abs", None)])[:2] == [0x80, 0x40]
    plain = pf.values(0x80)
    assert 0x80 not in plain and len(plain) == len(set(plain)) and 0 in plain
    bcs = LAB["play"] + 5
    live, ops = pf.branch_slice(SID, 0, 0, bcs)
    assert live == {LAB["tbl"]: 5} and ops == [("CMP", "imm", 0x10)]


def test_baseline_targets_and_data_bytes():
    base = pf.baseline(SID, 1, make_right, 4)
    (cpu,) = base.targets("6502")
    assert cpu.pc == LAB["play"] + 5 and cpu.frame == 0 and cpu.member.all()
    assert cpu.label == f"${cpu.pc:04X} BCS ${LAB['hi']:04x} -> ${LAB['hi']:04X}"
    (py,) = base.targets("python")
    assert py.name.startswith("edge:tests/test_pathfuzz.py:Toy._frame:")
    assert np.flatnonzero(base.data_mask).tolist() == [LAB["pad"] + i for i in range(4)]
    assert base.run.verdict["ok"] and base.universe("6502") == paths.cpu_universe(SID)


@pytest.mark.parametrize("side", pf.SIDES)
def test_exercise_reaches_the_other_outcome_and_flags_python(side):
    mutants = pf.exercise(SID, 1, make_right, side=side, budget=40, frames=4)
    assert len(mutants) == 1
    m = mutants[0]
    assert m.hit and m.diverges and m.side == side and m.error is None
    assert m.verdict["first_reg"] == 1 and m.verdict["first"] == 0
    ((off, value),) = m.patch
    assert off == TBL and value >= 0x10
    assert any("BCS" in label for label in m.labels["6502"])
    assert any("0xD401, 0x23" in label for label in m.labels["python"])


def test_exercise_6502_tries_the_compare_boundary_first():
    mutants = pf.exercise(SID, 1, make_right, budget=2, frames=4)
    assert [m.patch for m in mutants] == [[[TBL, 0x10]]]


def test_exercise_target_filter_budget_and_side():
    assert pf.exercise(SID, 1, make_right, targets=[], frames=4) == []
    assert pf.exercise(SID, 1, make_right, budget=1, frames=4) == []
    with pytest.raises(ValueError):
        pf.exercise(SID, 1, make_right, side="z80")


def _fragile(data, song):
    if data[TBL] >= 0x10:
        raise ValueError("unsupported")
    return make_right(data, song)


def test_exercise_records_a_failing_right_side():
    mutants = pf.exercise(SID, 1, _fragile, budget=40, frames=4, width=1)
    assert len(mutants) == 1 and mutants[0].hit and mutants[0].diverges
    assert mutants[0].new["python"] == [] and "ValueError" in mutants[0].error
    base = pf.baseline(_high(), 1, _fragile, 4)
    assert base.run.py is None and base.run.verdict is None
    assert base.targets("python") == [] and base.universe("python") == []
    assert not base.departs(base.run, 4)


def test_candidates_cover_every_byte_once_after_the_derived_values():
    base = pf.baseline(SID, 1, make_right, 4)
    (target,) = base.targets("6502")
    got = list(pf.candidates(base, target, np.random.default_rng(3), width=1))
    derived = pf.values(5, [("CMP", "imm", 0x10)])
    assert got[: len(derived)] == [(TBL, v) for v in derived]
    rest = [v for _, v in got[len(derived) :]]
    assert sorted(rest) == [v for v in range(256) if v != 5]


def test_rank_reads_orders_by_slice_contrast_frame_and_specificity():
    reads = [np.array(r) for r in ([1, 2, 3], [1, 3, 4], [1, 2], [5])]
    took = np.array([True, False, True, False])
    other = np.array([False, True, False, False])
    target = pf.Target("t", "python", "t", 1, took | other, [took, other])
    mask = np.zeros(0x10000, bool)
    mask[[1, 2, 3, 4]] = True
    assert pf.rank_reads(reads, target, mask).tolist() == [4, 2, 3, 1]
    assert pf.rank_reads(reads, target, mask, {1: 0}).tolist() == [1, 4, 2, 3]
    v = pf._verdict(np.zeros((3, 25)), np.zeros((2, 25)), 8)
    assert not v["ok"] and "frames" in v["error"]


def test_ddmin_reduces_a_planted_trigger_exactly():
    rng = np.random.default_rng(1)
    offs = rng.choice(np.arange(0x7C, len(SID)), 12, replace=False)
    patch = [[int(o), int(v)] for o, v in zip(offs, rng.integers(0, 256, 12))]
    need = {tuple(patch[i]) for i in (2, 7, 9)}
    calls = []

    def predicate(data, frames):
        calls.append(frames)
        return frames >= 7 and all(data[o] == v for o, v in need)

    got, frames = pf.minimise(SID, 1, predicate, patch=patch, frames=40)
    assert {tuple(p) for p in got} == need and frames == 7
    assert pf.ddmin(list(range(5)), lambda c: True) == []
    assert pf.ddmin([1, 2, 3, 4], lambda c: 3 in c) == [3]
    assert pf.first_true(lambda k: k >= 1, 9) == 1
    with pytest.raises(ValueError):
        pf.minimise(SID, 1, lambda d, f: False, patch=patch, frames=4)


def test_same_onset_predicate_and_neutralising_minimise():
    pred = pf.same_onset_predicate(make_right, 1)
    assert pred(_high(), 1) and not pred(SID, 4)
    assert not pf.same_onset_predicate(make_right, 0)(_high(), 4)
    assert not pf.same_onset_predicate(_fragile, 1)(_high(), 4)
    data = _high()
    addrs = [LAB["pad"] + i for i in range(4)]
    patch, frames = pf.minimise(data, 1, pred, addrs=addrs, frames=30)
    assert patch == [[o, 0] for o in PAD] and frames == 1
    assert pf.minimise(data, 1, pred, frames=8) == ([], 1)


class Regs:
    """A generic player whose registers live in a list and that ends after 2 ticks."""

    def __init__(self):
        self.regs, self.tick = [0] * 25, 0

    def play_frame(self):
        self.tick += 1
        if self.tick > 2:
            return None
        self.regs[0] = self.tick
        return self.regs


def test_right_grid_of_a_generic_subject():
    subject = paths.PySubject(Regs, cell=lambda p, r: (p.regs, r))
    grid = pf.right_grid(subject, 5)
    assert grid.shape == (2, 25) and grid[:, 0].tolist() == [1, 2]
    assert pf.as_subject(subject) is subject
    assert pf.right_grid(make_right(SID, 1), 2)[:, 0].tolist() == [0x11, 0x11]


def test_cli_gaps_exercise_minimise(tmp_path, capsys):
    tune = tmp_path / "toy.sid"
    tune.write_bytes(SID)
    right = ["--right", "tests.test_pathfuzz:make_right", "--frames", "4"]
    gaps = tmp_path / "gaps.json"
    assert pf.main(["gaps", str(tune), *right, "--out", str(gaps)]) == 0
    doc = json.loads(gaps.read_text())
    assert len(doc["6502"]["targets"]) == len(doc["python"]["targets"]) == 1
    assert doc["6502"]["covered"] + len(doc["6502"]["gaps"]) == doc["6502"]["universe"]
    out = tmp_path / "mutants.json"
    pf.main(["exercise", str(tune), *right, "--budget", "5", "--out", str(out)])
    doc = json.loads(out.read_text())
    assert doc["mutants"][0]["patch"] == [[TBL, 0x10]] and doc["gain"]["python"] == 2
    assert "#0 [[" in capsys.readouterr().out
    small = tmp_path / "min.json"
    pf.main(["minimise", str(tune), *right, "--patch", f"{out}#0", "--out", str(small)])
    doc = json.loads(small.read_text())
    assert doc["patch"] == [[TBL, 0x10]] and doc["register"] == 1 and doc["frames"] == 1
    assert pf.load_patch(str(small)) == doc["patch"]
    (tmp_path / "bare.json").write_text(json.dumps(doc["patch"]))
    assert pf.load_patch(str(tmp_path / "bare.json")) == doc["patch"]
    high = tmp_path / "high.sid"
    high.write_bytes(_high())
    pf.main(["minimise", str(high), *right, "--register", "1"])
    assert "0 bytes, 1 frames" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        pf.main(["minimise", str(tune), *right])
