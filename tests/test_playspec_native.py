"""The native call machine against the Python executor: random programs, tunes whose play
routine reaches the spec through 6502 code, memory banking and the paths back to Python.
"""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pysidtracker import SidParseError
from pysidtracker.asm6502 import assemble
from pysidtracker.playspec import SpecPlayer, SpecRuntimeError, native, parse
from pysidtracker.playspec import elaborate
from pysidtracker.playspec.ir import K, Obj, Op, Program, Rd, V
from pysidtracker.playspec.pyexec import compile_program, controls
from pysidtracker.testing import whole_song

from .helpers import build_psid
from .test_playspec_events import BINDING, FULL

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


def doubling(use_native):
    """A :class:`Doubling` player of the wrapped tune."""
    image, labels = tune("wrapped")
    return Doubling(
        parse(PLAY),
        image,
        0x1000,
        labels["init"],
        labels["wrap"],
        bind=lambda view, entry: {"freq": 0x1300, "sid": 0xD400, "rom": 0xA000},
        entries=lambda view: {labels["play"]},
        native=use_native,
    )


def test_a_subclass_replacing_a_hook_is_played_in_python():
    python, fast = doubling(False), doubling(True)
    assert fast.render_grid(whole_song(30)) == python.render_grid(whole_song(30))
    assert fast._played == 0  # pylint: disable=protected-access


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


def test_the_entries_of_the_last_frames_calls_are_recorded():
    labels = tune("wrapped")[1]
    python, fast = player("wrapped", False), player("wrapped", True)
    fast.render_grid(whole_song(5))
    python.render_grid(whole_song(5))
    assert fast.entered == python.entered == [labels["play"]]
    fast.play_frame()
    assert fast.entered == [labels["play"]]
    assert fast._played >= 5  # pylint: disable=protected-access


ROUTINGS = {
    "routes": dict(
        routes={0xD400: (0xD410, 0x7F, 0x80), 0xD401: None}, muted={0xD404: True}
    ),
    "unmuted": dict(routes={0xD400: (0xD402, 0xFF, 0)}, muted={0xD404: False}),
    "ghost": dict(ghost=0x3000, routes={0xD401: (0xD400, 0x0F, 0x20)}),
}


@pytest.mark.parametrize("routing", sorted(ROUTINGS))
def test_stores_routed_to_another_register_land_there_natively(routing):
    python, fast = player("wrapped", False), player("wrapped", True)
    for each in (python, fast):
        each.reroute(**ROUTINGS[routing])
    assert fast.render_grid(whole_song(40)) == python.render_grid(whole_song(40))
    assert fast.memory == python.memory and fast.writes == python.writes
    assert fast._played >= 39  # pylint: disable=protected-access
    fast.reroute()
    python.reroute()
    assert fast.render_grid(whole_song(5)) == python.render_grid(whole_song(5))


def test_the_play_address_switches_after_the_frames_it_names():
    python, fast = player("wrapped", False), player("wrapped", True)
    play = tune("wrapped")[1]["play"]
    for each in (python, fast):
        each.play_after = (7, play)
    assert fast.render_grid(whole_song(20)) == python.render_grid(whole_song(20))
    assert (
        fast.play_after is None and fast._play_addr == play
    )  # pylint: disable=protected-access
    assert cpu_state(fast) == cpu_state(python)
    assert fast._played >= 19  # pylint: disable=protected-access


def test_a_bus_that_is_not_the_machines_is_played_in_python():
    fast = player("wrapped", True)

    class Proxy:
        """The bus, seen through another object."""

        def __init__(self, bus):
            self._bus = bus

        def __getitem__(self, addr):
            return self._bus[addr]

        def __getattr__(self, name):
            return getattr(self._bus, name)

    fast._bus = Proxy(fast._bus)  # pylint: disable=protected-access
    fast.render_grid(whole_song(5))
    assert fast._played == 0  # pylint: disable=protected-access


def run_both(program, memory, frames=3):
    """Run ``program`` on both executors from ``memory``: the Python error or None, and
    the native outcome, checking that they agree on the memory and the SID writes."""
    expected, got = bytearray(memory), bytearray(memory)
    frame, state = compile_program(program), controls(program)
    core, out = native_core(program, got), bytearray(25)
    for _ in range(frames):
        before, writes = bytes(got), []
        try:
            frame(
                expected.__getitem__, sid_writes_of(expected, writes), state, None, {}
            )
        except SpecRuntimeError as exc:
            assert core.run(1, out, 0)[:2] == (0, 2) and bytes(got) == before
            return exc
        assert core.run(1, out, 0)[:2] == (1, 0)
        assert bytes(got) == bytes(expected) and core.writes() == writes
    return None


def table_memory(seed):
    """Memory whose scratch and table pages are ``seed`` over an alphabet of tags."""
    memory = bytearray(0x10000)
    tags = (0, 1, 2, 3, 0x7F, 0x80, 0x81, 0xF0, 0xF2, 0xFE, 0xFF)
    for i, address in enumerate(range(0x2000, 0x2300)):
        byte = seed[i % len(seed)]
        memory[address] = tags[byte % len(tags)] if byte & 0x80 else byte
    return memory


@pytest.mark.strain_exempt(reason="asserts equality of two executors")
@pytest.mark.parametrize("seed", range(6))
def test_the_spec_of_every_object_kind_plays_the_same_natively(seed):
    program = elaborate(parse(FULL), BINDING)
    memory = bytearray(0x10000)
    for i in range(0x2000, 0x2100):
        memory[i] = (i * (seed + 3) >> 2) & 0xFF if seed % 2 else (i * 7 + seed) % 5
    memory[0x12] = 3 + seed
    assert run_both(program, memory, 40) is None


def narrow(draw, width, source):
    """An operand of ``width`` bits: a constant or bytes of the scratch page."""
    pick = draw(source)
    if width == 1:
        return K(pick & 1, 1)
    if pick & 1:
        return K(pick if width == 8 else pick * 257 & 0xFFFF, width)
    if width == 8:
        return Rd(K(0x2000 + pick % 0x40, 16))
    return Op(
        "word", 16, (Rd(K(0x2000 + pick % 0x40, 16)), Rd(K(0x2001 + pick % 0x40, 16)))
    )


@st.composite
def transforms(draw, columns):
    """An ``(column, mask, shl, shr)`` operand over ``columns`` row bytes."""
    col = draw(st.integers(-1, columns - 1))
    mask = draw(st.sampled_from((0xFF, 0x7F, 0x0F, 0xF0, 0x03)))
    shift = draw(st.sampled_from(((0, 0), (1, 0), (4, 0), (0, 1), (0, 4), (0, 0))))
    return (col, mask, *shift) if col >= 0 else (-1, draw(st.integers(0, 255)), 0, 0)


@st.composite
def walkers(draw, name):
    """A random walker over cells below ``$2040`` and a table in ``$2100``."""
    columns, width = draw(st.integers(1, 3)), draw(st.integers(1, 3))
    wide = [draw(st.booleans()) for _ in range(draw(st.integers(1, 3)))]
    base = 0x2000 + 3 * draw(st.integers(0, 8))
    locs = {"pos": (base, base + 1)[: 1 + draw(st.booleans())], "cnt": (base + 2,)}
    for i, w in enumerate(wide):
        locs[f"v{i}"] = (base + 3 + 3 * i, base + 4 + 3 * i)[: 1 + w]

    def assigns():
        out = []
        for _ in range(draw(st.integers(0, 2))):
            i = draw(st.integers(0, len(wide)))
            part = 0 if i == len(wide) else draw(st.integers(0, 2 if wide[i] else 1))
            guard = None
            if draw(st.booleans()):
                guard = (
                    draw(st.integers(0, columns - 1)),
                    draw(st.sampled_from((0x80, 0xFF, 0xF0))),
                    draw(st.sampled_from((0, 0x80, 0x10))),
                    draw(st.booleans()),
                )
            out.append((i, part, draw(transforms(columns)), guard))
        return tuple(out)

    edges = sorted(
        draw(st.lists(st.integers(0, 255), min_size=1, max_size=4, unique=True))
    )
    rows = []
    for lo, hi in zip([0, *[e + 1 for e in edges[:-1]]], edges):
        kind = draw(st.sampled_from(("stop", "jump", "sweep", "delay", "set")))
        into = draw(st.integers(0, len(wide) - 1))
        frac = draw(st.integers(1, 6)) if wide[into] and draw(st.booleans()) else 0
        if kind == "jump":
            hold = draw(st.sampled_from((None, (0, 0x7F), (0x80, 0xFF), (3, 3))))
            via = draw(st.sampled_from((None, 0x80, 0x10, 0)))
            action = ("jump", draw(transforms(columns)), hold, via)
        elif kind == "sweep":
            count, step = draw(transforms(columns)), draw(transforms(columns))
            action = ("sweep", into, count, step, frac, assigns())
        elif kind == "delay":
            action = ("delay", draw(transforms(columns)), assigns())
        elif kind == "set":
            action = ("set", assigns(), assigns())
        else:
            action = ("stop",)
        rows.append((lo, hi, action))
    events = draw(st.lists(st.sampled_from(("frame", "note")), min_size=1, unique=True))
    reactions = {e: (draw(st.sampled_from(("step", "reset"))),) for e in events}
    return Obj(name, "walker", "", locs, (width, columns, tuple(rows)), reactions)


@st.composite
def cursors(draw, name):
    """A random cursor over a stream in ``$2100``."""
    base = 0x2000 + 3 * draw(st.integers(0, 8))
    locs = {"pos": (base,)}
    run = None
    if draw(st.booleans()):
        lo = draw(st.integers(0, 250))
        run = (lo, lo + draw(st.integers(0, 5)), draw(st.integers(0, lo)))
        locs["run"] = (base + 1,)
    fields = []
    for k in range(draw(st.integers(1, 4))):
        cond = (
            draw(st.sampled_from((0x80, 0xFF, 0x0F))),
            draw(st.integers(0, 3)),
            draw(st.booleans()),
        )
        stores = []
        for j in range(draw(st.integers(0, 2))):
            locs[f"o{k}_{j}"] = (base + 3 + 6 * k + 3 * j, base + 4 + 6 * k + 3 * j)[
                : 1 + draw(st.booleans())
            ]
            keep, shl, shr = draw(
                st.sampled_from(
                    ((0xFF, 0, 0), (0x7F, 0, 0), (0x0F, 4, 0), (0xFF, 0, 3))
                )
            )
            test = draw(
                st.sampled_from(((0, 0, True), (0x80, 0, True), (0x80, 0, False)))
            )
            stores.append((keep, shl, shr, test))
        fields.append((cond, tuple(stores)))
    events = draw(st.lists(st.sampled_from(("frame", "row")), min_size=1, unique=True))
    reactions = {
        e: (draw(st.sampled_from(("next", "reset", "settle"))),) for e in events
    }
    last = draw(st.booleans())
    return Obj(name, "cursor", "", locs, (run, tuple(fields), last), reactions)


@st.composite
def small_objects(draw, name):
    """A random counter or owner."""
    base = 0x2000 + 3 * draw(st.integers(0, 8))
    if draw(st.booleans()):
        action = draw(
            st.one_of(
                st.just(("reset",)),
                st.tuples(st.sampled_from(("up", "down")), st.integers(1, 9)),
                st.tuples(
                    st.sampled_from(("up", "down")), st.integers(1, 9), st.just(True)
                ),
            )
        )
        loc = (base, base + 1)[: 1 + draw(st.booleans())]
        fires = draw(st.sampled_from((None, "tick")))
        return Obj(name, "counter", "", {"cell": loc}, (), {"frame": action}, fires)
    action = (draw(st.sampled_from(("claim", "release"))),)
    return Obj(name, "owner", "", {"cell": (base,)}, (), {"frame": action})


@st.composite
def object_programs(draw):
    """A random program reacting random objects, and the flag of each wrap."""
    objects = [
        draw(st.one_of(walkers("o0"), cursors("o1"), small_objects("o2"))),
        draw(st.one_of(walkers("o3"), cursors("o4"), small_objects("o5"))),
    ]
    program = Program("random", [], objects={o.name: o for o in objects})
    source = st.integers(0, 255)
    for index in range(draw(st.integers(1, 6))):
        obj = objects[draw(st.integers(0, 1))]
        event = draw(st.sampled_from(sorted(obj.reactions)))
        ops = tuple(narrow(draw, w, source) for w in obj.widths())
        if obj.kind in ("walker", "cursor"):
            ops = (
                tuple(
                    K(0x2100 + draw(st.integers(0, 0x3F)), 16)
                    for _ in ops[: obj.arity() - 1]
                )
                + ops[-1:]
            )
        flag = f"w{index}" if obj.fires and draw(st.booleans()) else None
        program.body.append(("obj", obj.name, event, flag, ops, (), (), ()))
        if flag:
            program.body.append(
                ("if", V(flag, 1), [("wr", K(0x2200 + index, 16), K(1, 8))], [])
            )
    return program


@pytest.mark.strain_exempt(reason="asserts equality of two executors")
@settings(
    max_examples=400,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(object_programs(), st.binary(min_size=4, max_size=64))
def test_random_object_programs_agree_with_the_python_executor(program, seed):
    run_both(program, table_memory(seed), 4)


@pytest.mark.strain_exempt(reason="asserts equality of two executors")
def test_a_native_walker_that_jumps_forever_bails_to_the_error_python_raises():
    rows = ((0, 255, ("jump", (0, 0xFF, 0, 0), None, None)),)
    walker = Obj(
        "w",
        "walker",
        "",
        {"pos": (0x2000,), "cnt": (0x2001,), "v0": (0x2002,)},
        (1, 1, rows),
        {"frame": ("step",)},
    )
    program = Program(
        "loop",
        [("obj", "w", "frame", None, (K(0x2100, 16), K(0, 8)), (), (), ())],
        objects={"w": walker},
    )
    memory = bytearray(0x10000)
    memory[0x2100:0x2200] = bytes((i + 1) & 0xFF for i in range(256))
    assert "forever" in str(run_both(program, memory, 1))


def instructions(*ins):
    """The encoded ``(opcode name, argument)`` instructions."""
    return b"".join(
        native.OPS[op].to_bytes(8, "little") + arg.to_bytes(8, "little", signed=True)
        for op, arg in ins
    )


@pytest.mark.strain_exempt(reason="asserts equality of two executors")
@pytest.mark.parametrize(
    "description, operands, message",
    [
        ([0, -1, 2, 0x10, -1, 1, 1], 1, "out of range"),
        ([7, -1, 0], 0, "object kind"),
        ([0, -1, 2, 0x10, -1], 2, "ends early"),
        ([0, -1, 2, 0x10, -1, 1, 1, 0], 2, "too long"),
        ([0, 9, 2, 0x10, -1, 1, 1], 2, "flag variable"),
    ],
)
def test_malformed_object_descriptions_are_refused(description, operands, message):
    core = native.Core(bytearray(0x10000), ENTRY, ENTRY, 1000)
    core.add_bank(BANK, False, [], [ENTRY])
    code = instructions(
        *[("Konst", 0)] * operands, ("Obj", operands << 24), ("Halt", 0)
    )
    with pytest.raises(ValueError, match=message):
        core.add(BANK, ENTRY, code, [], [], 1, 0, [description])


@pytest.mark.strain_exempt(reason="asserts equality of two executors")
@pytest.mark.parametrize(
    "action", [("up", 1), ("up", 3), ("down", 1), ("down", 3), ("reset",)]
)
@pytest.mark.parametrize("wide", [False, True])
def test_counters_wrap_at_every_boundary_natively(action, wide):
    cell = (0x2000, 0x2001) if wide else (0x2000,)
    counter = Obj("c", "counter", "", {"cell": cell}, (), {"frame": action}, "tick")

    def operand(addr):
        low = Rd(K(addr, 16))
        return Op("word", 16, (low, Rd(K(addr + 1, 16)))) if wide else low

    program = Program(
        "count",
        [("obj", "c", "frame", "w", (operand(0x2002), operand(0x2004)), (), (), ())],
        objects={"c": counter},
    )
    for value in [0, 1, 2, 3, 4, 5, 0xFE, 0xFF, 0x100, 0x1FF, 0xFFFF]:
        for limit in (0, 2, 5, 0xFF, 0x100):
            memory = bytearray(0x10000)
            memory[0x2000:0x2002] = (value & 0xFFFF).to_bytes(2, "little")
            memory[0x2002:0x2004] = (limit & 0xFFFF).to_bytes(2, "little")
            memory[0x2004:0x2006] = (0x1234 if wide else 0x34).to_bytes(2, "little")
            assert run_both(program, memory, 1) is None
