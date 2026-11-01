"""Tests for paths: per-frame 6502 edges and Python arcs, store sites, reads and universes."""

import inspect
import os
import sys

import numpy as np
import pytest

from pysidtracker import oracle, paths
from pysidtracker.asm6502 import assemble
from pysidtracker.errors import SidParseError
from pysidtracker.player import MemPlayer
from pysidtracker.testing import song_frames, whole_song

from .helpers import build_psid, build_prg

TUNE = (
    "init:\n  LDA #0\n  STA cnt\n{pad}  RTS\n"
    "play:\n  LDA cnt\n  AND #1\n  BNE odd\n  LDA #$11\n  STA $D400\n  JMP done\n"
    "odd:\n  LDA #$22\n  STA $D401\n"
    "done:\n  JSR sub\n  INC cnt\n  RTS\n"
    "sub:\n  LDA tbl\n  STA $D418\n  RTS\n"
    "tbl:\n  .byte $0F\ncnt:\n  .byte $00\n"
)


def _tune(org=0x1000, pad=""):
    code, labels = assemble(TUNE.format(pad=pad), org)
    return build_psid(code, org, init=labels["init"], play=labels["play"]), labels


def _edges(trace, frame):
    return {trace.labels[i] for i in trace.frames[frame] if trace.names[i][0] == "e"}


def test_cpu_repeats_rank_and_out_of_span_calls():
    # Two identical voice copies, and a call out of the span to a stub at $0400.
    voice = "v{n}:\n  LDA #1\n  BNE w{n}\n  RTS\nw{n}:\n  STA $D400,X\n  RTS\n"
    src = (
        "init:\n  LDA #$60\n  STA $0400\n  RTS\n"
        "play:\n  LDX #0\n  JSR v1\n  LDX #7\n  JSR v2\n  JSR $0400\n  RTS\n"
        + voice.format(n=1)
        + voice.format(n=2)
    )
    code, lab = assemble(src, 0x1000)
    data = build_psid(code, 0x1000, init=lab["init"], play=lab["play"])
    trace = paths.cpu_paths(data, 1)
    ranked = {n.split("#")[0]: n for n in trace.names if n.endswith("#1")}
    assert len(ranked) == 2 and set(ranked) <= set(trace.names)
    out = f"${lab['play'] + 10:04X} JSR $0400 -> out"
    back = f"out -> ${lab['play'] + 13:04X}"
    assert {out, back} <= set(trace.labels)
    assert {n for n in trace.names if n.startswith("edge:")} <= set(
        paths.cpu_universe(data)
    )
    seven, zero = (trace.names[trace.sites[0, r]] for r in (7, 0))
    assert seven.replace("store:7:", "store:0:") == zero


def test_cpu_paths_needs_a_header():
    code, _ = assemble("  RTS\n", 0x1000)
    with pytest.raises(SidParseError):
        paths.cpu_paths(build_prg(code, 0x1000), 1)


# --- Python -----------------------------------------------------------------


class ToyMem(MemPlayer):
    """Alternates two SID stores by frame, reading a table byte for the volume."""

    def _init(self, subtune):
        self.cnt = 0

    def _frame(self):
        if self.cnt & 1:
            self._wr(0xD401, 0x22)
        else:
            self._wr(0xD400, 0x11)
        self._volume()
        self.cnt += 1

    def _volume(self):
        self._wr(0xD418, self._rd(0x1000))


def _line(func, text):
    lines, start = inspect.getsourcelines(func)
    return start + next(i for i, s in enumerate(lines) if text in s)


@pytest.mark.parametrize("backend", sorted(paths._BACKENDS))
def test_py_memplayer_arcs_sites_and_grid(backend):
    if backend == "monitoring" and not hasattr(sys, "monitoring"):
        pytest.skip("sys.monitoring needs Python 3.12")
    trace = paths.py_paths(lambda: ToyMem(b"\x0f", 0x1000), 4, backend=backend)
    src = paths.source_of(__file__)
    head = _line(ToyMem._frame, "if self.cnt")
    odd = src.arc_key(head, head + 1)
    even = src.arc_key(head, head + 3)
    names = [{trace.names[i] for i in f} for f in trace.frames]
    assert even in names[0] and odd not in names[0]
    assert odd in names[1] and even not in names[1]
    assert np.array_equal(
        trace.grid, np.asarray(ToyMem(b"\x0f", 0x1000).render_grid(whole_song(4)))
    )
    assert trace.names[trace.sites[0, 0]] == f"store:0:{src.where(head + 3)}"
    vol = _line(ToyMem._volume, "self._wr")
    assert trace.names[trace.sites[1, 24]] == f"store:24:{src.where(vol)}"
    assert trace.sites[0, 1] == -1 and trace.sites[1, 0] == -1
    assert trace.reads[0].tolist() == [0x1000]
    assert [trace.names[i] for i in trace.sequence(0)] == [
        even,
        f"store:0:{src.where(head + 3)}",
        f"store:24:{src.where(vol)}",
    ]
    assert trace.side == "python"
    assert not any("pysidtracker/player.py" in n for n in trace.names)
    assert {n for n in trace.names if n.startswith("edge:")} <= set(
        paths.py_universe(trace.roots, trace.sources)
    )


class Aliased:
    """A generic player writing its ``regs`` list through a local alias."""

    def __init__(self):
        self.regs = [0] * 25
        self.tick = 0

    def play_frame(self):
        regs = self.regs
        if self.tick % 2 and any(r >= 0 for r in regs):
            regs[1] = 0x22
        else:
            regs[0], regs[2:4] = 0x11, [5, 6]
        self.tick += 1
        return list(regs)

    def render_grid(self, n):
        return [list(self.play_frame()) for _ in range(n)]


def _regs(player, register):
    return player.regs, register


GENERATED = (
    "def _x_{k}(v):\n"
    "    return v + {c}\n"
    "\n"
    "def f_tick(out, t):\n"
    "    if t % 2:\n"
    "        out[4] = _x_{k}({c}) & 0xFF\n"
    "    else:\n"
    "        out[7 * 1 + 4] = {d}\n"
    "    if t > {d}:\n"
    "        out[4] = 1\n"
)


class Generated:
    """A VM-like player running per-tune generated code that writes a fresh list."""

    def __init__(self, name, text):
        namespace = {}
        exec(compile(text, name, "exec"), namespace)  # pylint: disable=exec-used
        self.tick_fn = namespace["f_tick"]
        self.regs = [0] * 25
        self.ticks = 0

    def play_frame(self):
        out = list(self.regs)
        self.tick_fn(out, self.ticks)
        self.regs = out
        self.ticks += 1
        return out

    def render_grid(self, n):
        return [list(self.play_frame()) for _ in range(n)]


def _generated(k, c, d):
    name, text = f"<playroutine toy{k}>", GENERATED.format(k=k, c=c, d=d)
    subject = paths.PySubject(
        lambda: Generated(name, text), cell=_regs, sources={name: text}, label="lift"
    )
    return subject, name, text


class Ending(Aliased):
    """A document-like player whose song ends after three ticks."""

    def play_frame(self):
        if self.tick >= 3:
            return None
        return super().play_frame()

    def render_grid(self, n):
        return [r for r in (self.play_frame() for _ in range(n)) if r is not None]


def test_py_ending_player_stops_its_grid():
    trace = paths.py_paths(paths.PySubject(Ending, cell=_regs), 6)
    assert trace.grid.shape == (3, 25) and len(trace.frames) == len(trace.reads) == 3
    assert trace.starts is None
    started = paths.PySubject(Ending, cell=_regs, starts=lambda p: (p.tick, None))
    assert paths.py_paths(started, 6).starts == (3, None)
    assert trace.sites.shape == (3, 25)
    assert np.array_equal(trace.grid, np.asarray(Ending().render_grid(whole_song(6))))


def test_py_bare_callable_and_explicit_roots(tmp_path):
    trace = paths.py_paths(lambda: ToyMem(b"\x0f", 0x1000), 2)
    assert trace.roots and trace.sources == {}
    none = paths.py_paths(
        paths.PySubject(lambda: ToyMem(b"\x0f", 0x1000), roots=[str(tmp_path)]), 2
    )
    assert none.names == [] and (none.sites == -1).all() and none.grid.shape == (2, 25)
    bad = tmp_path / "bad.py"
    bad.write_text("def (:\n")
    (tmp_path / "ok.py").write_text(
        "def f(a):\n    if a:\n        return 1\n    return 2\n"
    )
    src = paths.source_of(str(tmp_path / "ok.py"))
    assert paths.py_universe([str(tmp_path)]) == sorted(
        [src.arc_key(2, 3), src.arc_key(2, 4)]
    )


HVSC_TUNE = "/scratch/datasets/hvsc/C64Music/DEMOS/0-9/2hero.sid"


@pytest.mark.skipif(not os.path.exists(HVSC_TUNE), reason="no local HVSC")
def test_real_goattracker_tune():
    sid = pytest.importorskip("pygoattracker.sid")
    with open(HVSC_TUNE, "rb") as fh:
        data = fh.read()
    frames = song_frames(data, hvsc=HVSC_TUNE.split("/DEMOS/")[0], local_env="")
    cpu = paths.cpu_paths(data, frames)
    assert np.array_equal(cpu.grid, np.asarray(oracle.register_grid(data, frames)))
    assert (cpu.sites >= 0).any()
    py = paths.py_paths(lambda: sid.decompile_sid(data, subtune=0).player(), frames)
    fresh = sid.decompile_sid(data, subtune=0).player()
    assert np.array_equal(py.grid, np.asarray(fresh.render_grid(frames)))
    assert (py.sites >= 0).all()
    assert {n for n in py.names if n.startswith("edge:")} <= set(
        paths.py_universe(py.roots)
    )


SMC = (
    "init:\n  LDA #{v}\n  STA ${patch:04X}\n  RTS\n"
    "play:\npatch:\n  LDA #0\n  BEQ off\n  STA $D400\noff:\n"
    "  INC ${bump:04X}\nbump:\n  LDA #{v}\n  STA $D401\n  RTS\n"
)


class Masked(ToyMem):
    """A MemPlayer whose register file masks pulse-width highs it stores whole."""

    def _volume(self):
        self._wr(0xD403, 0x47)

    def snapshot(self):
        regs = super().snapshot()
        regs[3] &= 0x0F
        return regs
