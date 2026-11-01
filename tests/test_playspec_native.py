"""The native call machine against the Python executor: random programs, tunes whose play
routine reaches the spec through 6502 code, memory banking and the paths back to Python.
"""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pysidtracker import SidParseError
from pysidtracker.asm6502 import assemble
from pysidtracker.playspec import SpecPlayer, SpecRuntimeError, native, parse
from pysidtracker.playspec.ir import K, Op, Program, Rd, V
from pysidtracker.playspec.pyexec import compile_program, controls
from pysidtracker.testing import whole_song

from .helpers import build_psid

pytestmark = pytest.mark.skipif(
    not native.available(), reason="the native core is not built"
)


@pytest.fixture(autouse=True)
def engine(backend, request):
    """The tests drive jennings' native core; the pure-Python engine is checked once."""
    if (backend == "python") != ("python_engine" in request.node.name):
        pytest.skip("drives the other engine")
    return backend


ENTRY = 0x1000
BANK = 4
SCRATCH = range(0x2000, 0x2010)
SID = 0xD400
NAMES = ("a", "b", "c", "d")
BINARY = ("add", "sub", "mul", "shl")
PLAIN = ("and", "or", "xor", "shr", "eq", "ne", "ult", "ule", "land", "lor", "word")
SIGNED = ("slt", "sle")
UNARY = ("not", "neg", "zext", "lo", "hi", "lnot", "sext")
TRUTH = ("eq", "ne", "ult", "ule", "lnot", "land", "lor", "slt", "sle")


def masked(expr, mask):
    """``expr`` kept to ``mask``."""
    return Op("and", 16, (expr, K(mask, 16)))


def above_the_port(expr):
    """``expr`` as an address in ``$2000-$3FFF``."""
    return Op("or", 16, (masked(expr, 0x1FFF), K(0x2000, 16)))


@st.composite
def expressions(draw, depth=3):
    """A random expression over the variables, constants and the scratch bytes."""
    if depth == 0 or draw(st.integers(0, 5)) == 0:
        kind = draw(st.integers(0, 3))
        if kind == 0:
            return K(draw(st.integers(-300, 70000)), draw(st.sampled_from((8, 16))))
        if kind == 1:
            return V(draw(st.sampled_from(NAMES)), draw(st.sampled_from((8, 16))))
        address = draw(st.sampled_from(SCRATCH))
        return Rd(K(address, 16)) if kind == 2 else Rd(K(address ^ 0x1, 16))

    def sub():
        return draw(expressions(depth - 1))

    kind = draw(st.integers(0, 6))
    width = draw(st.sampled_from((8, 16)))
    if kind == 0:
        return Op(draw(st.sampled_from(BINARY)), width, (sub(), sub()))
    if kind == 1:
        return Op(draw(st.sampled_from(PLAIN)), width, (sub(), sub()))
    if kind == 2:
        return Op(draw(st.sampled_from(SIGNED)), 1, (sub(), sub()))
    if kind == 3:
        return Op(draw(st.sampled_from(UNARY)), width, (sub(),))
    if kind == 4:
        return Op("ite", width, (sub(), sub(), sub()))
    if kind == 5:
        return Op(
            draw(st.sampled_from(BINARY)),
            width,
            (sub(), K(draw(st.integers(-40, 90)), 0)),
        )
    return Rd(above_the_port(sub()))


@st.composite
def statements(draw, depth=2, procs=0):
    """A random list of statements over the variables and the scratch bytes."""
    out = []
    for _ in range(draw(st.integers(1, 4))):
        kind = draw(st.integers(0, 7 if depth else 2))
        if kind == 0:
            out.append(("set", draw(st.sampled_from(NAMES)), draw(expressions())))
        elif kind == 1:
            address = draw(st.sampled_from([*SCRATCH, *range(SID, SID + 4)]))
            out.append(("wr", K(address, 16), masked(draw(expressions()), 0xFF)))
        elif kind == 2:
            out.append(
                (
                    "wr",
                    above_the_port(draw(expressions())),
                    masked(draw(expressions()), 0xFF),
                )
            )
        elif kind == 3:
            then, other = draw(statements(depth - 1, procs)), draw(
                statements(depth - 1, procs)
            )
            cond = Op(
                draw(st.sampled_from(TRUTH)),
                1,
                (draw(expressions(2)), draw(expressions(2))),
            )
            out.append(("if", cond, then, other if draw(st.booleans()) else []))
        elif kind == 4:
            cond = Op("ult", 1, (V("a", 8), K(draw(st.integers(0, 20)), 8)))
            body = [("set", "a", Op("add", 8, (V("a", 8), K(1, 8))))]
            out.append(
                ("loop", cond, body + draw(statements(0)), draw(st.integers(1, 30)), 7)
            )
        elif kind == 5:
            out.append(
                (
                    "assert",
                    Op("ne", 1, (draw(expressions(2)), K(draw(st.integers(0, 3)), 8))),
                    "x",
                )
            )
        elif kind == 6 and procs:
            out.append(("call", f"p{draw(st.integers(0, procs - 1))}#"))
        elif kind == 7:
            out.append(("finish",))
    return out


@st.composite
def programs(draw):
    """A random program: initial variables, a body and two procedures."""
    inner = draw(statements(1))
    outer = draw(statements(1, procs=1))
    init = [("set", name, K(draw(st.integers(0, 255)), 8)) for name in NAMES]
    body = init + draw(statements(2, procs=2))
    return Program("random", body, procs={"p0#": inner, "p1#": outer})


def memory_of(seed: bytes) -> bytearray:
    """A 64 KiB memory with ``seed`` repeated over the scratch bytes."""
    memory = bytearray(0x10000)
    for i, address in enumerate(range(0x2000, 0x2040)):
        memory[address] = seed[i % len(seed)]
    return memory


def sid_writes_of(memory, writes):
    """A store function writing ``memory`` and recording SID stores to ``writes``."""

    def store(address, value):
        if SID <= address < SID + 25:
            writes.append((address - SID, value & 0xFF))
        memory[address] = value

    return store


def native_core(program, memory):
    """A native core running ``program`` at ``ENTRY`` over ``memory`` with no banking."""
    core = native.Core(memory, ENTRY, ENTRY, 1_000_000)
    core.add_bank(BANK, False, [], [ENTRY])
    core.select(BANK)
    core.add(BANK, ENTRY, *native.encode(program))
    core.set_cpu((ENTRY, 0xFF, 0, 0, 0, 0x24, 0))
    return core


@pytest.mark.strain_exempt(
    reason="asserts equality of two executors, which coverage cannot see"
)
@settings(
    max_examples=400,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(programs(), st.binary(min_size=1, max_size=16))
def test_random_programs_agree_with_the_python_executor(program, seed):
    expected, got = memory_of(seed), memory_of(seed)
    frame = compile_program(program)
    state = controls(program)
    core = native_core(program, got)
    out = bytearray(25)
    for _ in range(3):
        before = bytes(got)
        writes = []
        try:
            frame(
                expected.__getitem__, sid_writes_of(expected, writes), state, None, {}
            )
            python_error = None
        except Exception as exc:  # pylint: disable=broad-exception-caught
            python_error = exc
        played, status, _info = core.run(1, out, 0)
        if python_error is not None:
            assert (played, status) == (0, 2), python_error
            assert bytes(got) == before
            return
        assert (played, status) == (1, 0)
        assert bytes(got) == bytes(expected)
        assert core.writes() == writes
        assert list(out) == [
            got[SID + i] & (0x0F if i in (3, 10, 17) else 0xFF) for i in range(25)
        ]


def note_program():
    """A program reading the note table and storing the words."""
    index = Op("and", 8, (Rd(K(0x2000, 16)), K(0x3F, 8)))
    lo = Op("note", 8, (K(985248, 0), K(0, 0), K(0, 0), index))
    hi = Op("note", 8, (K(985248, 0), K(1, 0), K(0, 0), index))
    body = [("wr", K(SID, 16), lo), ("wr", K(SID + 1, 16), hi)]
    return Program("notes", body)


@pytest.mark.parametrize("index", [0, 33, 63, 200])
def test_note_tables_and_their_bounds(index):
    program = note_program()
    expected, got = bytearray(0x10000), bytearray(0x10000)
    expected[0x2000] = got[0x2000] = index
    compile_program(program)(expected.__getitem__, expected.__setitem__, [], None, {})
    played, status, _ = native_core(program, got).run(1, bytearray(25), 0)
    assert (played, status) == (1, 0) and got == expected


def test_unrepresentable_programs_are_unsupported():
    with pytest.raises(native.Unsupported, match="binding"):
        native.encode(Program("x", [("set", "a", K(None, 8))]))
    with pytest.raises(native.Unsupported, match="no native form for teleport"):
        native.encode(Program("x", [("teleport",)]))
    with pytest.raises(native.Unsupported, match="no native form for cube"):
        native.encode(Program("x", [("set", "a", Op("cube", 8, (K(1, 8),)))]))


def test_the_core_refuses_what_it_cannot_run():
    with pytest.raises(ValueError, match="writable"):
        native.Core(bytes(0x10000), ENTRY, ENTRY, 1000)
    core = native.Core(bytearray(0x10000), ENTRY, ENTRY, 1000)
    with pytest.raises(ValueError, match="port setting"):
        core.add_bank(9, False, [], [])
    with pytest.raises(ValueError, match="no such bank"):
        core.select(3)
    core.add_bank(3, False, [], [])
    with pytest.raises(ValueError, match="whole instructions"):
        core.add(3, ENTRY, b"\0" * 15, [], [], 0, 0)
    with pytest.raises(ValueError, match="no opcode"):
        core.add(3, ENTRY, bytes([255, 255]) + bytes(14), [], [], 0, 0)
    with pytest.raises(ValueError, match="out of range"):
        core.add(
            3,
            ENTRY,
            bytes([native.OPS["Jmp"]]) + bytes(7) + bytes([9]) + bytes(7),
            [],
            [],
            0,
            0,
        )
    with pytest.raises(ValueError, match="no such program"):
        core.controls(3, ENTRY)
    with pytest.raises(RuntimeError, match="too small"):
        core.run(2, bytearray(25), 0)


PLAY = """
spec counter;
layout { freq, sid, rom: addr; }
cell sid: u8[32] @ L.sid out;
cell rom: u8 @ L.rom;
machine m {
  cell freq: u8 @ L.freq;
  step { freq += 3; sid[0] := freq; sid[1] := rom; sid[4] := 0x41; }
}
schedule { m; }
"""

TUNES = {
    "direct": ("", "", "", ""),
    "wrapped": ("", "", "", ""),
    "banked": ("  lda #$34\n  sta $01\n", "  lda #$35\n  sta $01\n", "", ""),
    "device": ("", "", "  lda $d41b\n  sta $1301\n", ""),
}


def tune(kind):
    """``(image, labels)`` of a tune whose play routine calls the modelled routine."""
    before, after, probe, _ = TUNES[kind]
    code, labels = assemble(
        f"""
init:
  lda #$10
  sta $1300
  lda #$5a
  sta $a000
  rts
play:
  lda $1300
  clc
  adc #$03
  sta $1300
  sta $d400
  lda $a000
  sta $d401
  lda #$41
  sta $d404
  rts
wrap:
{probe}{before}  jsr play
{after}  rts
""",
        0x1000,
    )
    return (
        build_psid(code, 0x1000, init=labels["init"], play=labels["wrap"])[0x7C:],
        labels,
    )


def player(kind, use_native, spec_text=PLAY, **kwargs):
    """A spec player of the tune ``kind``."""
    image, labels = tune(kind)
    play = labels["play"] if kind == "direct" else labels["wrap"]
    return SpecPlayer(
        parse(spec_text),
        image,
        0x1000,
        labels["init"],
        play,
        bind=lambda view, entry: {"freq": 0x1300, "sid": 0xD400, "rom": 0xA000},
        entries=lambda view: {labels["play"]},
        native=use_native,
        **kwargs,
    )


def cpu_state(play):
    """The registers and cycle count of ``play``'s CPU."""
    mpu = play._mpu  # pylint: disable=protected-access
    return mpu.pc, mpu.sp, mpu.a, mpu.x, mpu.y, mpu.p, mpu.processorCycles


@pytest.mark.parametrize("kind", ["direct", "wrapped"])
def test_tunes_play_the_same_natively(kind):
    python, fast = player(kind, False), player(kind, True)
    assert fast.render_grid(whole_song(300)) == python.render_grid(whole_song(300))
    assert (
        fast.writes
        == python.writes
        == [(0, fast.regs[0]), (1, fast.regs[1]), (4, 0x41)]
    )
    assert cpu_state(fast) == cpu_state(python)
    assert fast._played >= 299 and fast._bails == 0  # pylint: disable=protected-access
    assert (
        fast.regs == python.regs and fast._last_regs == python._last_regs
    )  # pylint: disable=protected-access


def test_frame_by_frame_agrees_with_the_grid():
    stepped, grid = player("wrapped", True), player("wrapped", True)
    frames = whole_song(40)
    rows = []
    for _ in range(frames):
        stepped.play_frame()
        rows.append(stepped.regs)
    assert rows == grid.render_grid(frames)
    assert not grid.render_grid(whole_song(0))


def test_the_cpu_port_selects_memory_maps_the_machine_adds_as_it_meets_them():
    python, fast = player("banked", False), player("banked", True)
    assert fast.render_grid(whole_song(100)) == python.render_grid(whole_song(100))
    assert fast._bus.bits == python._bus.bits  # pylint: disable=protected-access
    assert (
        fast._core is not None and fast._bails == 0
    )  # pylint: disable=protected-access
    assert cpu_state(fast) == cpu_state(python)


def test_a_device_the_cpu_reads_hands_every_frame_to_python_until_it_gives_up():
    python, fast = player("device", False), player("device", True)
    assert fast.render_grid(whole_song(200)) == python.render_grid(whole_song(200))
    assert not fast._fast and fast._bails > 8  # pylint: disable=protected-access
    assert cpu_state(fast) == cpu_state(python)


def test_the_environment_and_the_argument_disable_the_machine(monkeypatch):
    assert player("direct", None)._fast  # pylint: disable=protected-access
    monkeypatch.setenv(native.ENV, "0")
    assert not player("direct", None)._fast  # pylint: disable=protected-access
    assert player("direct", True)._fast  # pylint: disable=protected-access
    assert not native.available()


STORE_PORT = PLAY.replace(
    "sid[4] := 0x41;", "sid[4] := 0x41; port[0] := 0x34;"
).replace("cell rom:", "cell port: u8[2] @ 1;\ncell rom:")


def test_a_spec_storing_to_the_cpu_port_is_played_by_python():
    python, fast = player("direct", False, STORE_PORT), player(
        "direct", True, STORE_PORT
    )
    assert fast.render_grid(whole_song(30)) == python.render_grid(whole_song(30))
    assert fast._bus.bits == python._bus.bits  # pylint: disable=protected-access


READ_PORT = PLAY.replace(
    "sid[4] := 0x41;", "sid[4] := 0x41; sid[2] := port[1]; sid[3] := port[0];"
).replace("cell rom:", "cell port: u8[2] @ 0;\ncell rom:")


def test_a_spec_reading_the_cpu_port_sees_what_python_sees():
    python, fast = player("direct", False, READ_PORT), player("direct", True, READ_PORT)
    assert fast.render_grid(whole_song(20)) == python.render_grid(whole_song(20))
    assert fast._played >= 19  # pylint: disable=protected-access


HIDDEN = """
spec hidden;
layout { sid, freq: addr; }
cell sid: u8[32] @ L.sid out;
machine m {
  control phase { low = 0, high = 1 }
  cell v: u8 @ L.freq;
  step { in phase { low => { v += 1; goto high; } high => { v += 2; goto low; } } sid[0] := v; }
}
schedule { m; }
"""


def test_hidden_control_states_persist_between_native_and_python_frames():
    python, fast = player("direct", False, HIDDEN), player("direct", True, HIDDEN)
    python.render_grid(whole_song(5))
    fast.render_grid(whole_song(5))
    state = fast._compiled[next(iter(fast._compiled))][
        2
    ]  # pylint: disable=protected-access
    assert (
        state == python._compiled[next(iter(python._compiled))][2]
    )  # pylint: disable=protected-access
    assert fast.render_grid(whole_song(11)) == python.render_grid(whole_song(11))


LOOPING = PLAY.replace("freq += 3;", "while freq != 99 bound 2 { freq += 3; }")


def test_a_failing_spec_raises_what_python_raises():
    for use_native in (False, True):
        with pytest.raises(SpecRuntimeError, match="ran past"):
            player("direct", use_native, LOOPING).render_grid(whole_song(5))


NATIVE_SPEC = """
spec hooked;
layout { freq, sid, hook, tail: addr; }
cell sid: u8[32] @ L.sid out;
machine m {
  cell freq: u8 @ L.freq;
  step {
    let v = freq;
    native L.hook(a = v) resumes L.tail;
    freq += 3; sid[0] := freq; sid[4] := 0x41;
  }
}
schedule { m; }
"""

HOOKS = {
    "plain": "",
    "banked": "  ldx #$34\n  stx $01\n  ldx #$35\n  stx $01\n",
}


def hooked(kind, use_native, wrapped):
    """A spec player whose modelled routine hands control to the tune's code."""
    code, labels = assemble(
        f"""
init:
  lda #$10
  sta $1300
  rts
play:
  lda $1300
  jmp hook
tail:
  clc
  adc #$03
  sta $1300
  sta $d400
  lda #$41
  sta $d404
  rts
hook:
{HOOKS[kind]}  cmp #$40
  beq away
  jmp tail
away:
  lda #$00
  sta $1300
  sta $d400
  rts
wrap:
  jsr play
  rts
""",
        0x1000,
    )
    return SpecPlayer(
        parse(NATIVE_SPEC),
        code,
        0x1000,
        labels["init"],
        labels["wrap" if wrapped else "play"],
        bind=lambda view, entry: {
            "freq": 0x1300,
            "sid": 0xD400,
            "hook": labels["hook"],
            "tail": labels["tail"],
        },
        entries=lambda view: {labels["play"]},
        native=use_native,
    )


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("kind", ["plain", "banked"])
def test_native_blocks_run_the_tunes_code_and_may_end_the_call(kind, wrapped):
    python, fast = hooked(kind, False, wrapped), hooked(kind, True, wrapped)
    grid = fast.render_grid(whole_song(80))
    assert grid == python.render_grid(whole_song(80))
    assert 0 in {row[0] for row in grid[1:]}
    assert cpu_state(fast) == cpu_state(python) and fast.memory == python.memory
    assert (
        fast._played > 40 if kind == "plain" else fast._bails
    )  # pylint: disable=protected-access


def test_native_code_that_never_reaches_its_resume_address_is_refused():
    code, labels = assemble(
        """
init:
  rts
play:
spin:
  jmp spin
""",
        0x1000,
    )
    for use_native in (False, True):
        fast = SpecPlayer(
            parse(NATIVE_SPEC),
            code,
            0x1000,
            labels["init"],
            labels["play"],
            bind=lambda view, entry: {
                "freq": 0x1300,
                "sid": 0xD400,
                "hook": labels["spin"],
                "tail": 0x2000,
            },
            entries=lambda view: {labels["play"]},
            native=use_native,
        )
        with pytest.raises(SidParseError, match="never reaches"):
            fast.render_grid(whole_song(2))


class Doubling(SpecPlayer):
    """A player whose stores to the first SID register are doubled."""

    def _store(self, addr, value):
        super()._store(addr, value * 2 & 0xFF if addr == SID else value)


def test_a_subclass_replacing_a_hook_is_played_in_python():
    image, labels = tune("wrapped")
    made = []
    for use_native in (False, True):
        made.append(
            Doubling(
                parse(PLAY),
                image,
                0x1000,
                labels["init"],
                labels["wrap"],
                bind=lambda view, entry: {"freq": 0x1300, "sid": 0xD400, "rom": 0xA000},
                entries=lambda view: {labels["play"]},
                native=use_native,
            )
        )
    python, fast = made
    assert fast.render_grid(whole_song(30)) == python.render_grid(whole_song(30))
    assert (
        fast.regs[0] == (python.regs[0]) and fast._played == 0
    )  # pylint: disable=protected-access


def test_an_instance_replacing_a_hook_sees_every_call():
    fast = player("wrapped", True)
    entered = []
    run = fast._run  # pylint: disable=protected-access

    def trace(entry):
        entered.append(entry)
        return run(entry)

    fast._run = trace  # pylint: disable=protected-access
    fast.render_grid(whole_song(12))
    assert len(entered) == 12 and fast._played == 0  # pylint: disable=protected-access


def test_the_play_address_may_change_between_frames():
    python, fast = player("wrapped", False), player("wrapped", True)
    rows = []
    for each in (python, fast):
        grid = each.render_grid(whole_song(10))
        each._play_addr = tune("wrapped")[1]["play"]  # pylint: disable=protected-access
        rows.append(grid + each.render_grid(whole_song(10)))
    assert rows[0] == rows[1]
    assert cpu_state(fast) == cpu_state(python)
