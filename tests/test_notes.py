"""Standard note words, declared note tables and the standard-notes SpecPlayer."""

from dataclasses import replace

import pytest

from pysidtracker import SidFormat
from pysidtracker.errors import SidFormatError, SidParseError
from pysidtracker.formats import SpecBinding
from pysidtracker.notes import frequency
from pysidtracker.playspec import SpecError, SpecPlayer, elaborate, parse
from pysidtracker.playspec.pyexec import source
from pysidtracker.playspec.smt import Executor, State, step

from .helpers import build_psid

PAL, NTSC = 985248, 1022727


def test_frequency_matches_the_grammar_package():
    notes = pytest.importorskip("unitracker_grammar.notes")
    for clock in (PAL, NTSC):
        for note in range(256):
            assert frequency(note, clock) == notes.frequency(note, clock)


LOAD = 0x1000
LO, HI, IDX, SID = 0x1100, 0x1200, 0x1300, 0xD400
SPEC = """
spec notes;
layout { lo, hi, idx, sid: addr; }
cell sid: u8[32] @ L.sid out;
cell idx: u8[4] @ L.idx;
cell flo: u8[8] @ L.lo ro notes(lo, 0);
cell fhi: u8[8] @ L.hi ro notes(hi, 0);
cell glo: u8[8] @ L.lo ro notes(lo, -1);
cell ghi: u8[8] @ L.hi ro notes(hi, -1);
cell raw: u8[8] @ L.lo ro;
cell word: u16[8] @ (L.lo, L.hi) ro notes(word, 3);
machine m {
  step {
    sid[0] := flo[idx[0]];
    sid[1] := fhi[idx[0]];
    sid[7] := glo[idx[1]];
    sid[8] := ghi[idx[1]];
    sid[14] := lo(word[idx[2]]);
    sid[15] := hi(word[idx[2]]);
    sid[2] := raw[idx[0]];
  }
}
schedule { m; }
"""
BINDING = {"lo": LO, "hi": HI, "idx": IDX, "sid": SID}
NATIVE = {n: (0x1111 * (n + 1)) & 0xFFFF for n in range(256)}


def tune(indexes):
    """A one-page tune (rts for init and play) with a native table over all indexes."""
    image = bytearray(0x400)
    image[0] = 0x60
    for n, word in NATIVE.items():
        image[LO - LOAD + n], image[HI - LOAD + n] = word & 255, word >> 8
    image[IDX - LOAD : IDX - LOAD + len(indexes)] = bytes(indexes)
    return bytes(image)


def player(spec, indexes, notes=None, bind=None):
    return SpecPlayer(
        spec,
        tune(indexes),
        LOAD,
        LOAD,
        LOAD,
        bind=bind or (lambda view, entry: BINDING),
        notes=notes,
    )


def regs_of(spec, indexes, notes=None, bind=None):
    p = player(spec, indexes, notes, bind)
    p.play_frame()
    r = p.regs
    return {
        "f": r[1] << 8 | r[0],
        "g": r[8] << 8 | r[7],
        "w": r[15] << 8 | r[14],
        "raw": r[2],
    }


@pytest.mark.parametrize(
    "decl",
    [
        "cell t: u8[4] @ L.lo notes(lo, 0);",
        "cell t: u8 @ L.lo ro notes(lo, 0);",
        "cell t: u8[4] @ L.lo ro notes(word, 0);",
        "cell t: u16[4] @ (L.lo, L.hi) ro notes(lo, 0);",
        "cell t: u8[4] @ L.lo ro notes(lo, L.idx + zz);",
    ],
)
def test_malformed_declarations_are_rejected(decl):
    with pytest.raises(SpecError):
        elaborate(parse(SPEC.replace("cell raw: u8[8] @ L.lo ro;", decl)), BINDING)


def test_the_solver_rejects_standard_mode():
    program = elaborate(parse(SPEC), BINDING, standard=PAL)
    assert "T0[" in source(program)
    with pytest.raises(Exception, match="standard note mode"):
        step(program, None)
    with pytest.raises(Exception, match="standard note mode"):
        Executor().expr(next(iter(program.procs.values()))[0][2], State(None, {}))


class Model:
    name, author, released = "n", "a", "r"
    load_addr, init_addr, play_addr = LOAD, LOAD, LOAD


def test_sid_format_without_a_spec_hook_says_so():
    with pytest.raises(SidFormatError, match="no playspec"):
        SidFormat("bare", None, None).spec_player(Model())


def test_spec_binding_entries_and_subtune_reach_the_player():
    spec = parse(SPEC)
    binding = SpecBinding(
        spec, lambda v, e: BINDING, tune([1]), entries=lambda view: set(), subtune=0
    )
    fmt = SidFormat("synthetic", None, None, spec=lambda model: binding)
    with pytest.raises(SidParseError, match="no modelled entry"):
        fmt.spec_player(Model())


def _sid_format(flags=None, image=None):
    spec = parse(SPEC)
    sid = None if flags is None else build_psid(image or tune([7]), LOAD, flags=flags)
    return SidFormat(
        "synthetic",
        None,
        None,
        spec=lambda model: SpecBinding(
            spec, lambda view, entry: BINDING, tune([7, 7, 7]), sid=sid
        ),
    )


def _word(player):
    player.play_frame()
    return player.regs[1] << 8 | player.regs[0]


@pytest.mark.parametrize(
    "flags,clock", [(0b0100, PAL), (0b1000, NTSC), (0b0000, PAL), (0b1100, PAL)]
)
def test_notes_true_is_the_tune_clock(flags, clock):
    player = _sid_format(flags).spec_player(Model(), notes=True)
    assert _word(player) == frequency(7, clock)


def test_binding_accepts_a_loaded_image():
    from pysidtracker import SidImage  # pylint: disable=import-outside-toplevel

    data = build_psid(tune([7]), LOAD, flags=0b1000)
    fmt = _sid_format()
    binding = fmt.spec(Model())
    image = SidImage.from_bytes(data)
    fmt = SidFormat("x", None, None, spec=lambda m: replace(binding, sid=image))
    assert _word(fmt.spec_player(Model(), notes=True)) == frequency(7, NTSC)
    prg = SidImage.from_prg(bytes([0, 0x10]) + tune([7]))
    fmt = SidFormat("x", None, None, spec=lambda m: replace(binding, sid=prg))
    assert _word(fmt.spec_player(Model(), notes=True)) == frequency(7, PAL)
