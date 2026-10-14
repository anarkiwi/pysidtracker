"""The interrupt host against recorded sidtrace traces and documented C64 timing."""

import collections

import pytest

from pysidtracker import irqhost
from pysidtracker import registers as reg
from pysidtracker.asm6502 import assemble
from pysidtracker.errors import SidParseError
from pysidtracker.irqhost import (
    BR,
    F,
    X2,
    InterruptHost,
    InterruptPlayer,
    boot_cycle,
    host_cadence,
    kernal_stub,
    opcode_layout,
    ram_pattern,
)
from pysidtracker.oracle import SidtraceRow, read_sidtrace, sidtrace_cadence
from pysidtracker.oracle import sidtrace_grid

from . import irqhost_tunes as tunes
from .helpers import build_psid, build_psid_embedded_load

PAL_LINE, PAL_FRAME = reg.PAL_CYCLES_PER_LINE, reg.PAL_CYCLES_PER_FRAME
QUIET = """
init: SEI
  LDA #$7F
  STA $DC0D
  LDA $DC0D
"""


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(irqhost, "poweron_writes", lambda: None)


def _rsid(src, **kwargs):
    code, labels = assemble(src, tunes.LOAD)
    kwargs.setdefault("magic", b"RSID")
    play = labels[kwargs.pop("play")] if "play" in kwargs else 0
    return build_psid_embedded_load(
        code, tunes.LOAD, init=tunes.LOAD, play=play, **kwargs
    )


def _run(src, seconds=0.25, **kwargs):
    host = InterruptHost(_rsid(src, **kwargs))
    host.run(int(seconds * host.clock_hz))
    return host, host.rows()


def _events(rows, value):
    return [r.cycle for r in rows if r.chip is None and r.value == value]


def _writes(rows, register, chip=0):
    return [(r.cycle, r.value) for r in rows if r.chip == chip and r.reg == register]


def _line(cycle):
    """The PAL raster line the beam is on at ``cycle``, and the line cycle."""
    return (cycle // PAL_LINE - 1) % 312, cycle % PAL_LINE


@pytest.mark.parametrize("name", sorted(tunes.TUNES))
def test_trace_matches_the_recorded_oracle(name):
    want = read_sidtrace(tunes.fixture(name))
    host = InterruptHost(tunes.build(name), poweron=irqhost.KERNAL_VECTORS)
    end = host.start + tunes.FIXTURE_CYCLES
    host.run(end + 1 - host.cycle)
    assert [r for r in host.rows() if r.cycle <= end] == want


def test_every_opcode_takes_its_documented_cycles():
    from deity_informant import CYCLETIME, EXTRACYCLES
    from jennings.opcodes import OPCODES

    for op in range(256):
        ents, exec_at, kind = opcode_layout(op)
        if OPCODES[op][0] == "JAM":
            continue
        fetch = next(i for i, k in enumerate(ents) if k in irqhost._FETCHES or k == BR)
        base = 1 + fetch - (X2 in ents)
        if kind == "branch":
            assert (base, EXTRACYCLES[op]) == (2, 2)
        else:
            assert base == CYCLETIME[op], hex(op)
            assert (X2 in ents) == bool(EXTRACYCLES[op]), hex(op)
        assert 0 <= exec_at < len(ents) or OPCODES[op][0] in ("BRK", "ANE")


def test_layouts_put_writes_last_and_reads_before():
    assert opcode_layout(0x8D)[0] == (irqhost.R, irqhost.R, irqhost.W, F)
    assert opcode_layout(0xEE)[0][-3:] == (irqhost.W, irqhost.W, F)
    assert opcode_layout(0xEE)[2] == "rmw"
    assert opcode_layout(0x20)[0] == (
        irqhost.R,
        irqhost.R,
        irqhost.W,
        irqhost.W,
        irqhost.R,
        F,
    )
    assert opcode_layout(0x00)[0] == irqhost.BRK_ENTRIES
    assert opcode_layout(0x02)[0] == (irqhost.HALT,)


def test_kernal_stub_is_sidplayfps_rom_less_kernal():
    rom = kernal_stub()
    word = lambda a: rom[a - 0xE000] | rom[a - 0xDFFF] << 8  # noqa: E731
    assert (word(0xFFFA), word(0xFFFC), word(0xFFFE)) == (0xFE43, 0xFCE2, 0xFF48)
    assert rom[0xEA31 - 0xE000 : 0xEA34 - 0xE000] == b"\x4c\x7e\xea"
    assert rom[0xEA7E - 0xE000 : 0xEA81 - 0xE000] == b"\x0c\x0d\xdc"
    assert rom[0xFF48 - 0xE000 : 0xFF50 - 0xE000] == bytes.fromhex("488a4898486c1403")
    assert rom[0xFFD2 - 0xE000] == rom[0xF1CA - 0xE000] == 0x60
    assert rom[0xE100 - 0xE000] == 0xEA


def test_ram_powers_up_in_four_byte_runs():
    ram = ram_pattern()
    assert ram[:16] == bytes.fromhex("0000ffffffff00000000ffffffff0000")
    assert ram[0x4000:0x4008] == bytes.fromhex("ffff00000000ffff")
    assert ram[0xC000:0xC008] == ram[0x4000:0x4008]


def test_the_driver_writes_volume_where_the_oracle_logs_it():
    assert boot_cycle(False) == 339711
    host = InterruptHost(tunes.build("init_writes"))
    host.run(100)
    assert host.rows()[0] == SidtraceRow(
        boot_cycle(False) + 40, None, None, None, 0, 24, 15
    )


def test_cia_timer_irq_raises_every_latch_plus_one_cycles():
    src = QUIET + """
  LDA #<irq
  STA $0314
  LDA #>irq
  STA $0315
  LDA #$34
  STA $DC04
  LDA #$12
  STA $DC05
  LDA #$81
  STA $DC0D
  LDA #$11
  STA $DC0E
  CLI
loop: JMP loop
irq: INC $D400
  JMP $EA31
"""
    _, rows = _run(src)
    raises = _events(rows, "cia_irq")
    assert len(raises) > 10
    assert {b - a for a, b in zip(raises, raises[1:])} == {0x1235}
    assert len(_events(rows, "cia_irq_entry")) == len(raises)


def test_one_shot_timer_raises_once():
    src = QUIET + """
  LDA #$81
  STA $DC0D
  LDA #$19
  STA $DC0E
  CLI
loop: JMP loop
"""
    _, rows = _run(src)
    assert len(_events(rows, "cia_irq")) == 1


def test_nmi_is_edge_triggered_until_icr_read():
    src = QUIET + """
  LDA #<nmi
  STA $0318
  LDA #>nmi
  STA $0319
  LDA #$00
  STA $DD04
  STA $DD05
  LDA #$81
  STA $DD0D
  LDA #$01
  STA $DD0E
loop: JMP loop
nmi: INC $D400
  RTI
"""
    _, rows = _run(src)
    assert len(_events(rows, "nmi")) == len(_events(rows, "nmi_entry")) == 1


def test_an_irq_waits_for_cli_then_one_more_instruction():
    src = QUIET + """
  LDA #<irq
  STA $0314
  LDA #>irq
  STA $0315
  LDA #$00
  STA $DC04
  LDA #$01
  STA $DC05
  LDA #$81
  STA $DC0D
  LDA #$19
  STA $DC0E
  LDX #$00
  LDA #$11
  LDY #$22
wait: DEX
  BNE wait
  CLI
  STA $D401
  STY $D402
loop: JMP loop
irq: LDA #$33
  STA $D400
  JMP $EA31
"""
    _, rows = _run(src)
    raise_at = _events(rows, "cia_irq")[0]
    entry = _events(rows, "cia_irq_entry")[0]
    assert entry - raise_at > 1000
    assert [r.reg for r in rows if r.chip == 0 and r.reg < 3] == [1, 0, 2]


@pytest.mark.parametrize("stall", [False, True])
def test_irq_latency_is_the_finishing_instruction_then_seven_cycles(stall):
    """An IRQ asserted before an instruction's last cycle waits for it, else for
    the next one too (NOP, NOP or JMP here), then takes 7 cycles to its entry."""
    src = QUIET + ("" if stall else "  LDA #$0B\n  STA $D011\n") + """
  LDA #<irq
  STA $0314
  LDA #>irq
  STA $0315
  LDA #$83
  STA $DC04
  LDA #$01
  STA $DC05
  LDA #$81
  STA $DC0D
  LDA #$11
  STA $DC0E
  CLI
loop: NOP
  NOP
  NOP
  NOP
  JMP loop
irq: LDA $DC0D
  JMP $EA81
"""
    _, rows = _run(src)
    pairs = zip(_events(rows, "cia_irq")[2:], _events(rows, "cia_irq_entry")[2:])
    latency = collections.Counter(entry - at for at, entry in pairs)
    if stall:
        assert max(latency) > 7
    else:
        assert set(latency) == {6, 7, 8}


def _loop_gaps(src):
    _, rows = _run(src, seconds=0.1)
    cycles = [c for c, _ in _writes(rows, 0x18)][1:]
    return cycles, [b - a for a, b in zip(cycles, cycles[1:])]


LOOP = """
  LDX #$00
loop: INX
  STX $D418
  JMP loop
"""


def test_badlines_steal_forty_to_forty_three_cycles_twenty_five_times_a_frame():
    cycles, gaps = _loop_gaps(QUIET + LOOP)
    stolen = collections.Counter(g - 9 for g in gaps if g != 9)
    assert set(stolen) <= {40, 41, 42, 43}
    stalled = [c for c, g in zip(cycles[1:], gaps) if g != 9]
    lines = [_line(c - 1)[0] for c in stalled]
    per_frame = collections.Counter((c - 1) // PAL_FRAME for c in stalled)
    assert set(list(per_frame.values())[1:-1]) == {25}
    assert all(0x30 <= line <= 0xF7 and line & 7 == 3 for line in lines)


def test_a_blank_screen_steals_nothing():
    _, gaps = _loop_gaps(QUIET + "  LDA #$0B\n  STA $D011\n" + LOOP)
    assert set(gaps) == {9}


@pytest.mark.parametrize("expand, height", [(0, 21), (1, 42)])
def test_a_sprite_steals_cycles_on_each_of_its_lines(expand, height):
    src = QUIET + f"""
  LDA #$0B
  STA $D011
  LDA #$01
  STA $D015
  LDA #$80
  STA $D001
  LDA #${expand:02X}
  STA $D017
""" + LOOP
    cycles, gaps = _loop_gaps(src)
    stalled = [c for c, g in zip(cycles[1:], gaps) if g != 9]
    frame = collections.defaultdict(set)
    for c in stalled:
        frame[(c - 1) // PAL_FRAME].add(_line(c - 1)[0])
    full = [lines for lines in frame.values() if len(lines) == height]
    assert full and all(min(lines) in (0x80, 0x81) for lines in full)


@pytest.mark.parametrize("line", [0x00, 0x37, 0x100])
def test_raster_irq_raises_at_the_start_of_its_line(line):
    src = QUIET + f"""
  LDA #<irq
  STA $0314
  LDA #>irq
  STA $0315
  LDA #${0x1B | (line >> 1 & 0x80):02X}
  STA $D011
  LDA #${line & 0xFF:02X}
  STA $D012
  LDA #$01
  STA $D01A
  CLI
loop: JMP loop
irq: INC $D019
  JMP $EA81
"""
    _, rows = _run(src)
    raises = _events(rows, "video_irq")[1:]
    assert len(raises) >= 10
    assert {_line(c) for c in raises} == {(line, 1 if line == 0 else 0)}
    assert {b - a for a, b in zip(raises, raises[1:])} == {PAL_FRAME}


@pytest.mark.parametrize("ack", ["INC", "DEC", "ASL", "LSR"])
def test_read_modify_write_acknowledges_the_raster_irq(ack):
    src = QUIET + f"""
  LDA #<irq
  STA $0314
  LDA #>irq
  STA $0315
  LDA #$01
  STA $D01A
  CLI
loop: JMP loop
irq: {ack} $D019
  JMP $EA81
"""
    _, rows = _run(src)
    assert len(_events(rows, "video_irq")) >= 10


def test_an_unacknowledged_raster_irq_is_raised_once():
    src = QUIET + """
  LDA #<irq
  STA $0314
  LDA #>irq
  STA $0315
  LDA #$01
  STA $D01A
loop: JMP loop
irq: JMP $EA81
"""
    _, rows = _run(src)
    assert len(_events(rows, "video_irq")) == 1


def test_the_kernal_exit_acknowledges_cia1():
    src = """
init: LDA #<irq
  STA $0314
  LDA #>irq
  STA $0315
  RTS
irq: INC count
  LDA count
  STA $D400
  JMP $EA31
count: .byte 0
"""
    _, rows = _run(src, seconds=0.2)
    raises = _events(rows, "cia_irq")
    assert {b - a for a, b in zip(raises, raises[1:])} == {0x4026}
    assert len(_writes(rows, 0)) == len(raises)


def test_brk_runs_through_the_irq_vector():
    src = """
init: LDA #<brk
  STA $0314
  LDA #>brk
  STA $0315
  BRK
  NOP
  LDA #$22
  STA $D401
  RTS
brk: LDA #$11
  STA $D400
  JMP $EA81
"""
    _, rows = _run(src, seconds=0.05)
    assert [r.value for r in rows if r.value == "brk_entry"] == ["brk_entry"]
    assert [(r.reg, r.value) for r in rows if r.chip == 0 and r.reg < 2] == [
        (0, 0x11),
        (1, 0x22),
    ]


def test_a_jam_stops_the_cpu():
    host, rows = _run("init: LDA #$01\n  STA $D400\n  .byte $02\n  STA $D401\n")
    assert host.jammed and [r.reg for r in rows if r.chip == 0] == [24, 0]


@pytest.mark.parametrize(
    "data, match",
    [
        (bytes([0x00, 0x10, 0x60]), "no SID header"),
        (
            build_psid_embedded_load(b"\x60", 0x1000, play=0, magic=b"RSID", flags=2),
            "BASIC",
        ),
    ],
)
def test_unplayable_images_are_rejected(data, match):
    with pytest.raises(SidParseError, match=match):
        InterruptHost(data)


def test_clock_follows_the_header_or_the_override():
    data = tunes.build("ntsc_raster")
    assert InterruptHost(data).clock_hz == reg.NTSC_CLOCK_HZ
    assert InterruptHost(data, clock="pal").clock_hz == reg.PAL_CLOCK_HZ
    with pytest.raises(ValueError):
        InterruptHost(data, clock="secam")


def test_grid_frames_the_rows_as_the_oracle_grid():
    host = InterruptHost(tunes.build("raster_kernal"))
    grid = host.grid(30)
    rows = host.rows()
    want = sidtrace_grid(rows, cycles_per_frame=sidtrace_cadence(rows))
    assert grid == want[:30] and len({r[0] for r in grid}) == 30
    assert host.framed == ("video_irq", PAL_FRAME)


def test_grid_forward_fills_past_the_last_frame():
    host = InterruptHost(tunes.build("psid_vbi"))
    host.run(PAL_FRAME * 3)
    grid = host.grid(5)
    assert len(grid) == 5


def test_grid_without_interrupts_is_unplayable():
    host = InterruptHost(_rsid(QUIET + "loop: JMP loop\n"))
    with pytest.raises(SidParseError, match="no interrupt"):
        host.grid(3)


def test_frames_repeat_a_row_for_each_missed_frame():
    src = QUIET + """
  LDA #<irq
  STA $0314
  LDA #>irq
  STA $0315
  LDA #$01
  STA $D01A
  CLI
loop: JMP loop
irq: INC $D019
  INC count
  LDA count
  STA $D400
  AND #$03
  BNE done
  LDA #$00
  STA $D01A
  LDA #$19
  STA $DC0E
  LDA #$81
  STA $DC0D
done: JMP $EA81
count: .byte 0
"""
    host = InterruptHost(_rsid(src))
    rows = list(
        zip(range(12), host.frames(source="video_irq", cycles_per_frame=PAL_FRAME))
    )
    assert [row[0] for _, row in rows][:3] == [1, 2, 3]


def test_frames_keep_advancing_once_the_source_stops():
    src = QUIET + """
  LDA #<irq
  STA $0314
  LDA #>irq
  STA $0315
  LDA #$01
  STA $D01A
  CLI
loop: JMP loop
irq: INC $D019
  INC count
  LDA count
  STA $D400
  CMP #$03
  BNE done
  LDA #$00
  STA $D01A
done: JMP $EA81
count: .byte 0
"""
    host = InterruptHost(_rsid(src))
    frames = host.frames(source="video_irq", cycles_per_frame=PAL_FRAME)
    got = [next(frames)[0] for _ in range(60)]
    assert got[:2] == [1, 2] and set(got[2:]) == {3}


def test_frames_without_interrupts_are_unplayable():
    host = InterruptHost(_rsid(QUIET + "loop: JMP loop\n"))
    with pytest.raises(SidParseError):
        host.frames(probe_seconds=0.1)


def test_player_renders_the_grid_then_continues_frame_by_frame():
    data = tunes.build("raster_kernal")
    player = InterruptPlayer(data)
    grid = player.render_grid(5)
    more = player.render_grid(3)
    assert [r[0] for r in grid + more] == [r[0] for r in InterruptHost(data).grid(8)]
    fresh = InterruptPlayer(data)
    assert fresh.play_frame()[0] == (0, grid[0][0])
    assert fresh.snapshot() == grid[0]


@pytest.mark.parametrize(
    "name, source, cpc",
    [
        ("raster_kernal", "pal_video", PAL_FRAME),
        ("psid_cia", "cia_timer", 0x3001),
        ("ntsc_raster", "ntsc_video", reg.NTSC_CYCLES_PER_FRAME),
    ],
)
def test_host_cadence_measures_the_play_source(name, source, cpc):
    cadence = host_cadence(tunes.build(name), seconds=1)
    assert (cadence.source.value, cadence.cycles_per_call) == (source, cpc)
    assert not cadence.dynamic


def test_host_cadence_without_a_source_is_one_frame():
    cadence = host_cadence(_rsid(QUIET + "loop: JMP loop\n"), seconds=0.2)
    assert (cadence.source, cadence.cycles_per_call) == (None, PAL_FRAME)


def test_rows_carry_the_cycles_since_each_source():
    rows = irqhost.to_rows(
        [(10, None, None, "nmi"), (15, 0, 1, 2), (20, None, None, "rti")]
    )
    assert rows[1] == SidtraceRow(15, 5, None, None, 0, 1, 2)
    assert rows[2].since_nmi == 10


def test_psid_drivers_play_at_their_speed_bit():
    vbi = InterruptHost(tunes.build("psid_vbi"))
    vbi.run(PAL_FRAME * 10)
    cia = InterruptHost(tunes.build("psid_cia"))
    cia.run(PAL_FRAME * 10)
    assert _events(vbi.rows(), "video_irq") and not _events(vbi.rows(), "cia_irq")
    assert _events(cia.rows(), "cia_irq") and not _events(cia.rows(), "video_irq")


def test_a_psid_without_a_free_page_header_still_boots():
    data = build_psid(b"\x60\xee\x00\xd4\x60", 0x1000, init=0x1000, play=0x1001)
    host = InterruptHost(data)
    host.run(PAL_FRAME * 4)
    assert len(_writes(host.rows(), 0)) >= 3
