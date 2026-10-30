"""Standard note words, declared note tables and the standard-notes SpecPlayer."""

from dataclasses import replace

import pytest

from pysidtracker import SidFormat
from pysidtracker.errors import SidFormatError, SidParseError
from pysidtracker.formats import SpecBinding
from pysidtracker.notes import frequency, table
from pysidtracker.playspec import SpecError, SpecPlayer, elaborate, parse, run
from pysidtracker.playspec.pyexec import source
from pysidtracker.playspec.smt import Executor, State, step
from pysidtracker.playspec.syntax import Spec

from .helpers import build_psid

PAL, NTSC = 985248, 1022727


@pytest.mark.parametrize(
    "clock,expected",
    [
        (PAL, {0: 278, 1: 295, 24: 1114, 57: 7493, 60: 8910, 94: 63504, 95: 0}),
        (NTSC, {0: 268, 1: 284, 24: 1073, 57: 7218, 60: 8584, 94: 61177, 95: 64815}),
    ],
)
def test_frequency_values(clock, expected):
    assert {n: frequency(n, clock) for n in expected} == expected


def test_frequency_matches_the_grammar_package():
    notes = pytest.importorskip("unitracker_grammar.notes")
    for clock in (PAL, NTSC):
        for note in range(256):
            assert frequency(note, clock) == notes.frequency(note, clock)


def test_table_parts_and_notes_below_c0():
    words = [frequency(-1, PAL), frequency(0, PAL), frequency(1, PAL)]
    assert 0 < words[0] < words[1]
    assert table(PAL, 2, -1, 3) == tuple(words)
    assert table(PAL, 0, -1, 3) == tuple(w & 255 for w in words)
    assert table(PAL, 1, -1, 3) == tuple(w >> 8 for w in words)


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


@pytest.mark.parametrize("x", [0, 3, 94, 200])
def test_native_mode_reads_memory(x):
    got = regs_of(parse(SPEC), [x, x, x])
    assert got["f"] == NATIVE[x]
    assert got["g"] == NATIVE[x]
    assert got["w"] == NATIVE[x]
    assert got["raw"] == NATIVE[x] & 255


@pytest.mark.parametrize("clock", [PAL, NTSC])
@pytest.mark.parametrize("x", [0, 3, 94, 95, 200, 255])
def test_standard_mode_answers_the_standard_word(clock, x):
    got = regs_of(parse(SPEC), [x, x, x], notes=clock)
    assert got["f"] == frequency(x, clock)
    assert got["g"] == frequency(x - 1, clock)
    assert got["w"] == frequency(x + 3, clock)
    assert got["raw"] == NATIVE[x] & 255


def test_standard_mode_leaves_memory_untouched():
    p = player(parse(SPEC), [200, 0, 0], notes=PAL)
    before = p.memory[LO : LO + 256] + p.memory[HI : HI + 256]
    p.play_frame()
    assert p.memory[LO : LO + 256] + p.memory[HI : HI + 256] == before


def test_aliasing_cells_with_different_offsets_differ():
    got = regs_of(parse(SPEC), [10, 10, 0], notes=PAL)
    assert got["f"] == frequency(10, PAL) and got["g"] == frequency(9, PAL)


def test_variants_carry_or_replace_the_declaration():
    base = parse(SPEC)
    shifted = parse(
        "spec shifted extends notes;\n"
        "override cell flo: u8[8] @ L.lo ro notes(lo, 10);\n"
        "override cell fhi: u8[8] @ L.hi ro notes(hi, 10);\n"
        "override cell glo: u8[8] @ L.lo ro;\n"
        "override cell ghi: u8[8] @ L.hi ro;\n",
        {"notes": base},
    )
    plain = parse("spec plain extends notes;", {"notes": base})
    got = regs_of(base, [5, 5, 0], PAL, lambda v, e: (shifted, BINDING))
    assert got["f"] == frequency(15, PAL) and got["g"] == NATIVE[5]
    got = regs_of(base, [5, 5, 0], PAL, lambda v, e: (plain, BINDING))
    assert got["f"] == frequency(5, PAL) and got["g"] == frequency(4, PAL)


def test_native_programs_are_unchanged_by_the_declaration():
    declared = elaborate(parse(SPEC), BINDING)
    stripped = elaborate(parse(SPEC.replace("notes(", "ro; //")), BINDING)
    assert source(declared) == source(stripped)
    assert declared.standard is None and "T0" not in source(declared)
    mem = bytearray(0x10000)
    mem[LO : LO + 256] = bytes(range(256))
    mem[IDX] = 9
    assert run(declared, mem)[SID] == 9


def test_declared_cells_are_recorded():
    cells = {c.name: c.notes for c in elaborate(parse(SPEC), BINDING).cells}
    assert cells["flo"] == ("lo", 0) and cells["word"] == ("word", 3)
    assert cells["raw"] is None and cells["idx"] is None


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


def test_unindexed_read_of_a_declared_cell_is_still_an_error():
    text = SPEC.replace("sid[2] := raw[idx[0]];", "sid[2] := flo;")
    with pytest.raises(SpecError, match="needs an index"):
        elaborate(parse(text), BINDING, standard=PAL)


class Model:
    name, author, released = "n", "a", "r"
    load_addr, init_addr, play_addr = LOAD, LOAD, LOAD


def test_sid_format_builds_the_spec_player():
    spec = parse(SPEC)
    fmt = SidFormat(
        "synthetic",
        None,
        None,
        spec=lambda model: SpecBinding(
            spec, lambda view, entry: BINDING, tune([7, 7, 7])
        ),
    )
    plain = fmt.spec_player(Model())
    standard = fmt.spec_player(Model(), notes=NTSC)
    plain.play_frame()
    standard.play_frame()
    assert plain.regs[1] << 8 | plain.regs[0] == NATIVE[7]
    assert standard.regs[1] << 8 | standard.regs[0] == frequency(7, NTSC)
    assert isinstance(standard.spec, Spec)


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


def test_notes_true_without_a_container_is_pal():
    assert _word(_sid_format().spec_player(Model(), notes=True)) == frequency(7, PAL)


def test_explicit_clock_and_kwargs_override_the_header():
    fmt = _sid_format(0b1000)
    assert _word(fmt.spec_player(Model(), notes=PAL)) == frequency(7, PAL)
    assert _word(fmt.spec_player(Model(), notes=True, ntsc=False)) == frequency(7, PAL)
    assert _word(fmt.spec_player(Model())) == NATIVE[7]


def test_header_sets_the_chip_and_machine_of_a_called_player():
    player = _sid_format(0b101000).spec_player(Model())
    assert player._sid_model == "8580"  # pylint: disable=protected-access
    assert player._pal_flag == 0  # pylint: disable=protected-access
    other = _sid_format(0b010100).spec_player(Model())
    assert other._sid_model == "6581"  # pylint: disable=protected-access
    assert other._pal_flag == 1  # pylint: disable=protected-access
    assert (
        _sid_format(0b101000).spec_player(Model(), sid_model="6581")._sid_model
        == "6581"
    )  # pylint: disable=protected-access


def test_binding_without_a_container_is_unchanged():
    player = _sid_format().spec_player(Model())
    assert (
        player._sid is None and player._clock is None
    )  # pylint: disable=protected-access
    assert player._sid_model == "6581"  # pylint: disable=protected-access


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
