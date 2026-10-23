"""pathdiff: corpus build, fault ranking and strict xfail cases on a toy corpus."""

import csv
import functools
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pysidtracker import pathdiff as pd
from pysidtracker import survey
from pysidtracker.asm6502 import assemble
from pysidtracker.gridcmp import LEAD
from pysidtracker.oracle import register_grid
from pysidtracker.paths import PySubject

from .helpers import DATA_OFFSET, build_psid

TUNE = (
    "mode:\n  .byte {mode}\ntrig:\n  .byte {trig}\ncnt:\n  .byte 0\n"
    "init:\n  LDA #0\n  STA cnt\n  LDA #$0F\n  STA $D418\n  RTS\n"
    "play:\n  LDA cnt\n  CMP trig\n  BCC early\n"
    "  LDA cnt\n  ASL A\n  CLC\n  ADC mode\n  STA $D400\n  JMP gate\n"
    "early:\n  LDA cnt\n  STA $D400\n"
    "gate:\n  LDA cnt\n  AND #1\n  BNE odd\n  LDA #$41\n  STA $D404\n  JMP fin\n"
    "odd:\n  LDA #$40\n  STA $D404\n"
    "fin:\n  INC cnt\n  RTS\n"
)

BUG = "self._wr(0xD400, cnt * 2)"
FIXED = "self._wr(0xD400, cnt * 2 + self._rd(0x1000))"

PLAYERS = '''
from pysidtracker.paths import PySubject
from pysidtracker.player import MemPlayer

LIFT_NAME = "<playroutine toy>"
LIFT = """
def tick(regs, st):
    if st["cnt"] >= st["trig"]:
        regs[0] = (st["cnt"] * 2 + st["mode"]) & 0xFF
    else:
        regs[0] = st["cnt"]
    if st["cnt"] & 1:
        regs[4] = 0x40
    else:
        regs[4] = 0x41
    st["cnt"] += 1
"""


class Toy(MemPlayer):
    def _init(self, subtune):
        self._wr(0x1002, 0)
        self._wr(0xD418, 0x0F)

    def _frame(self):
        cnt = self._rd(0x1002)
        if cnt >= self._rd(0x1001):
            self._late(cnt)
        else:
            self._wr(0xD400, cnt)
        if cnt & 1:
            self._wr(0xD404, 0x40)
        else:
            self._wr(0xD404, 0x41)
        self._wr(0x1002, cnt + 1)

    def _late(self, cnt):
        {late}


class Lift:
    def __init__(self, data):
        space = {{}}
        exec(compile(LIFT, LIFT_NAME, "exec"), space)
        self.tick = space["tick"]
        self.regs = [0] * 25
        self.regs[24] = 0x0F
        self.st = dict(mode=data[{off}], trig=data[{off} + 1], cnt=0)

    def play_frame(self):
        self.tick(self.regs, self.st)
        return list(self.regs)

    def render_grid(self, n):
        return [self.play_frame() for _ in range(n)]


def native(data, song):
    return Toy(data[{off}:], 0x1000, song - 1)


def lift(data, song):
    return PySubject(
        lambda: Lift(data),
        cell=lambda p, r: (p.regs, r),
        sources={{LIFT_NAME: LIFT}},
        label="lift",
        starts=lambda p: (0, None, None),
    )
'''

FAILING = [(16, t) for t in (3, 4, 5, 6, 7, 9)]
EXACT = [(0, t) for t in (2, 3, 4, 5)]
FRAMES = 30


def _tune(mode, trig):
    code, lab = assemble(TUNE.format(mode=mode, trig=trig), 0x1000)
    return build_psid(code, 0x1000, init=lab["init"], play=lab["play"])


def _players(root: Path, late: str) -> str:
    root.mkdir(parents=True, exist_ok=True)
    (root / "players.py").write_text(PLAYERS.format(late=late, off=DATA_OFFSET))
    survey.load_render.cache_clear()
    return f"{root / 'players.py'}:native"


@pytest.fixture(name="corpus")
def _corpus(tmp_path):
    """A fake HVSC tree, its ``tunes.csv`` and a survey ``results.csv``."""
    hvsc, work = tmp_path / "hvsc", tmp_path / "work"
    (hvsc / "TOY").mkdir(parents=True)
    work.mkdir()
    tunes, results = [], []
    for i, (mode, trig) in enumerate(FAILING + EXACT):
        data = _tune(mode, trig)
        rel = f"TOY/t{i}_{mode}_{trig}.sid"
        (hvsc / rel).write_bytes(data)
        md5 = hashlib.md5(data).hexdigest()
        tunes.append(dict(path=rel, md5=md5, player="toy", songs=1, clock="PAL"))
        tunes[-1]["lengths"] = "1"
        results.append(
            dict(
                key=f"{md5}_1",
                path=rel,
                song=1,
                frames=FRAMES,
                verdict="differs" if mode else "exact",
                info=json.dumps({"build": i % 2}),
            )
        )
    results.append(dict(key="x_1", path="TOY/none.sid", song=1, verdict="exact"))
    for name, rows in (("tunes.csv", tunes), ("results.csv", results)):
        with open(work / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
    spec = _players(tmp_path / "players", BUG)
    return hvsc, work, spec


def _jennings(path, song, frames):
    return register_grid(Path(path).read_bytes(), frames, subtune=song - 1)


def _check(hvsc, **kw):
    return functools.partial(
        pd.check_case,
        resolve=lambda rel, cache_dir: hvsc / rel,
        reference=_jennings,
        **kw,
    )


def test_6502_pair_end_to_end(corpus, tmp_path, capsys):
    hvsc, work, spec = corpus
    tally = pd.build(hvsc, work / "tunes.csv", work, pd.CPU, spec, jobs=1)
    assert tally == {"differs": 6, "agrees": 4, "error": 1}
    assert pd.build(hvsc, work / "tunes.csv", work, pd.CPU, spec, jobs=1) == {}
    pair = pd.pair_dir(work, pd.CPU, spec)
    assert pair.name == "6502-players.native"
    meta = json.loads((pair / "meta.json").read_text())
    assert meta["prefixes"] == ["6502:", "python:"]

    an = pd.analyse(work, pd.CPU, spec, progress=False)
    assert int(an.spec.fail.sum()) == len(FAILING)
    top = an.fixes[0]
    assert top.n_tunes == len(FAILING) and len(an.fixes) == 1
    assert top.key[0] == "value"
    assert top.key[1].startswith("6502:store:0:")
    assert top.key[2].startswith("python:store:0:players.py:Toy._late:")
    assert top.labels[0].endswith("STA $d400 -> v1_freq_lo")
    assert top.labels[1].endswith(f"{BUG} -> v1_freq_lo")
    rep = an.tune(top.representative)
    assert rep["path"].endswith("_16_3.sid") and int(rep["onset"]) == 3
    assert pd.explained(an.fixes) == {1: 6, 5: 6, 10: 6}

    out = tmp_path / "fixes.csv"
    assert (
        pd.main(
            [
                "rank",
                "--work",
                str(work),
                "--right",
                spec,
                "--out",
                str(out),
                "--explain",
                "1",
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "6 differing tunes; top 1: 6" in printed and BUG in printed
    row = next(csv.DictReader(open(out, encoding="utf-8")))
    assert row["id"] == pd.fix_id(top.key) and row["n_tunes"] == "6"
    explain = (pair / "explain" / "1.txt").read_text()
    assert "divergence:" in explain and "v1_freq_lo" in explain

    manifest = tmp_path / "xfails.json"
    check = _check(hvsc)
    cases = pd.xfails(an, manifest, check=check)
    assert len(cases) == 1
    case = cases[0]
    assert case["id"] == pd.fix_id(top.key) and case["key"] == repr(top.key)
    assert case["frames"] == 3 + 1 + LEAD and case["left"] == pd.CPU
    assert case["right"] == spec and case["compare"] == "lead"
    assert BUG in case["reason"] and "v1_freq_lo" in case["reason"]
    first = manifest.read_text()
    pd.xfails(an, manifest, check=check)
    assert manifest.read_text() == first

    with pytest.raises(AssertionError, match="divergence"):
        check(case)
    params = pd.xfail_params(manifest)
    (mark,) = params[0].marks
    assert params[0].id == case["id"] and mark.kwargs["strict"] is True
    assert mark.kwargs["raises"] is AssertionError
    assert pd.xfail_params(manifest, kind="path") == []

    _players(tmp_path / "players", FIXED)
    assert pd.xfails(an, manifest, prune=True, check=check, top=0) == []
    assert json.loads(manifest.read_text()) == []


def test_python_pair_onset_end_to_end(corpus, tmp_path):
    hvsc, work, spec = corpus
    lift = spec.replace(":native", ":lift")
    args = [str(hvsc), str(work / "tunes.csv"), "--work", str(work)]
    args += ["--left", spec, "--right", lift, "--compare", "onset", "--jobs", "2"]
    assert pd.main(["build", *args]) == 0
    an = pd.analyse(work, spec, lift, progress=False)
    assert an.meta["prefixes"] == ["python:", "lift:"]
    (top,) = an.fixes
    assert top.n_tunes == len(FAILING) and top.key[0] == "value"
    assert top.labels[0].endswith(f"{BUG} -> v1_freq_lo")
    assert top.key[2].startswith("lift:store:0:<playroutine toy>:tick:regs[#] = ")
    manifest = tmp_path / "lift.json"
    base = ["xfails", "--work", str(work), "--left", spec, "--right", lift]
    assert pd.main([*base, "--manifest", str(manifest), "--no-verify"]) == 0
    (case,) = pd.load_manifest(manifest)
    assert case["frames"] == 4 and case["compare"] == "onset"
    check = _check(hvsc)
    with pytest.raises(AssertionError, match="LIFT|lift"):
        check(case)
    rec = pd.record(_tune(16, 3), 1, spec, lift, "onset", 10)
    assert repr(top.key) in {repr(k) for k in an.keys_of(rec, "k")}
    assert an.keys_of(pd.record(_tune(0, 3), 1, spec, lift, "onset", 10)) == set()
    _players(tmp_path / "players", FIXED)
    check(case)


def test_patch_turns_an_exact_tune_into_a_case(corpus):
    hvsc, _, spec = corpus
    rel = "TOY/t6_0_2.sid"
    raw = (hvsc / rel).read_bytes()
    entry = dict(
        id="patched",
        relpath=rel,
        md5=hashlib.md5(raw).hexdigest(),
        song=1,
        frames=12,
        left=pd.CPU,
        right=spec,
        compare="lead",
        reason="r",
    )
    check = _check(hvsc)
    check(entry)
    with pytest.raises(AssertionError, match="v1_freq_lo"):
        check(dict(entry, patch=[[DATA_OFFSET, 16]]))
    with pytest.raises(pd.ChecksumMismatch):
        check(dict(entry, md5="0" * 32))
    with pytest.raises(pd.TuneFetchError):
        pd.check_case(entry, resolve=lambda rel, cache_dir: None)
    assert pd.apply_patch(b"abc", [[1, 0x5A]]) == b"aZc"
    assert pd.apply_patch(b"abc") == b"abc"
    assert pd.apply_patch(b"abc", [[1, 1], [2, 0x163], [1, 0x15A]]) == b"aZc"


def test_host_mismatch_drops_rows_from_the_oracle_split(corpus, tmp_path, monkeypatch):
    hvsc, _, spec = corpus
    data = (hvsc / "TOY/t0_16_3.sid").read_bytes()
    ocsv = tmp_path / "oracle.csv.zst"
    ocsv.touch()

    def split(_path, nframes):
        grid = np.asarray(register_grid(data, nframes), np.int16)
        grid[1:, 24] = 0
        return nframes, grid

    monkeypatch.setattr(pd, "oracle_frames", split)
    rec = pd.record(data, 1, pd.CPU, spec, "lead", 20, str(ocsv))
    assert rec["host"] and not rec["fail"] and rec["onset"] == -1
    assert int(np.sum(rec["count"])) == 1
    clean = pd.record(data, 1, pd.CPU, spec, "lead", 20, str(tmp_path / "none"))
    assert clean["fail"] and clean["onset"] == 3 and not clean["host"]
    assert clean["left"].startswith("6502:store:0:")
    assert sum(clean["count"]) == 4


def test_tunes_csv_without_results_and_meta_guard(corpus, tmp_path):
    hvsc, work, spec = corpus
    bare = tmp_path / "bare"
    bare.mkdir()
    rows = pd.select(bare, work / "tunes.csv", limit=3)
    assert len(rows) == 3 and rows[0]["verdict"] == ""
    assert pd.frames_for(rows[0], 50) == 50
    assert pd.frames_for(dict(frames="20"), 50) == 20
    tally = pd.build(
        hvsc, work / "tunes.csv", bare, pd.CPU, spec, jobs=1, limit=2, max_frames=12
    )
    assert tally == {"differs": 1, "agrees": 1} or tally == {"differs": 2}
    with pytest.raises(ValueError, match="compare"):
        pd.build(hvsc, work / "tunes.csv", bare, pd.CPU, spec, mode="onset")
    again = pd.build(
        hvsc, work / "tunes.csv", bare, pd.CPU, spec, jobs=1, limit=1, force=True
    )
    assert sum(again.values()) == 1
    assert pd.info_extras(bare) == {}


def test_sides_and_comparisons(corpus, monkeypatch):
    hvsc, _, spec = corpus
    data = (hvsc / "TOY/t0_16_3.sid").read_bytes()
    native = pd.subject(spec, data, 1)
    assert native.label == "python" and native.make() is not native.make()
    assert pd.subject(pd.CPU, data, 1) is None
    lift = pd.subject(spec.replace(":native", ":lift"), data, 1)
    assert isinstance(lift, PySubject) and lift.label == "lift"
    g, starts = pd.grid(spec.replace(":native", ":lift"), data, 1, 6)
    assert g.shape == (6, 25) and starts == (0, None, None)
    jg, none = pd.grid(pd.CPU, data, 1, 6)
    assert none is None and np.array_equal(jg, g)

    class Fmt:  # pylint: disable=too-few-public-methods
        name = "toyfmt"

        @staticmethod
        def play(d, subtune):
            return survey.load_render(spec)(d, subtune + 1)

    monkeypatch.setattr(pd, "find_format", lambda d, name: Fmt)
    fmt = pd.subject("format:toyfmt", data, 1)
    assert fmt.label == "toyfmt"
    assert fmt.make().render_grid(4) == native.make().render_grid(4)

    assert pd.prefixes("python", "python") == ("python-left:", "python-right:")
    assert pd.slug("format:goattracker2") == "format.goattracker2"
    assert pd.slug("pkg.mod:fn") == "pkg.mod.fn"
    gated = np.zeros((5, 25), np.int16)
    gated[2:, 4] = 0x41
    gated[4, 18] = 0x11
    assert pd.gate_starts(gated) == (2, None, 4)
    other = gated.copy()
    other[1, 0] = 9
    other[3, 0] = 9
    cut = pd.compare(gated, other, "onset")
    assert cut == pd.Cut(0, 5, 3, 0)
    assert pd.compare(gated, other, "onset", (None, None, None)).frame is None
    long, _ = pd.grid(pd.CPU, data, 1, 20)
    lead = pd.compare(long[:10], np.vstack([np.zeros((2, 25), np.int16), long]))
    assert lead.lead == 2 and lead.frame is None and lead.size == 10
    assert lead.at(True, 1) == 3 and lead.at(False, 1) == 1


def test_compare_masks_pulse_width_highs():
    base = np.zeros((12, 25), np.int16)
    high = base.copy()
    high[3:, 3] = 0x40
    assert pd.compare(base, high, "lead").frame is None
    assert pd.compare(base, high, "onset", (0, 0, 0)).frame is None


def test_branch_bug_keys_on_the_first_parting(corpus, tmp_path):
    hvsc, work, _ = corpus
    root = tmp_path / "gate"
    root.mkdir()
    text = PLAYERS.format(late=FIXED, off=DATA_OFFSET)
    (root / "players.py").write_text(
        text.replace("if cnt & 1:", "if cnt & 1 or cnt == 6:")
    )
    survey.load_render.cache_clear()
    spec = f"{root / 'players.py'}:native"
    assert pd.build(hvsc, work / "tunes.csv", work, pd.CPU, spec, jobs=1) == {
        "differs": len(FAILING + EXACT),
        "error": 1,
    }
    an = pd.analyse(work, pd.CPU, spec, progress=False)
    (top,) = an.fixes
    assert top.key[0] == "first" and top.n_tunes == len(FAILING + EXACT)
    assert "BNE" in top.labels[0] and "or cnt == 6" in top.labels[1]
    assert top.labels[2].startswith("after: $") and "BCC" in top.labels[2]
    assert top.labels[3].startswith("after: ") and "if cnt >= " in top.labels[3]
    rec = an.records[top.representative]
    assert int(rec["onset"]) == 6 and (np.asarray(rec["rank"]) >= 0).any()
    fresh = pd.record(_tune(0, 3), 1, pd.CPU, spec, "lead", 10)
    assert an.keys_of(fresh, "k") == {top.key}

    pair = pd.pair_dir(work, pd.CPU, spec)
    meta = json.loads((pair / "meta.json").read_text())
    (pair / "meta.json").write_text(json.dumps(dict(meta, format=1)))
    with pytest.raises(ValueError, match="--force"):
        pd.analyse(work, pd.CPU, spec, progress=False)
    with pytest.raises(ValueError, match="--force"):
        pd.build(hvsc, work / "tunes.csv", work, pd.CPU, spec, jobs=1)
    assert pd.build(hvsc, work / "tunes.csv", work, pd.CPU, spec, limit=1, force=True)
    assert json.loads((pair / "meta.json").read_text())["format"] == pd.FORMAT
