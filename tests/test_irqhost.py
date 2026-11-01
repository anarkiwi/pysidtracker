"""The interrupt host against recorded sidtrace traces and documented C64 timing."""

import pytest

from vagrant_moses import memmap as mm

from pysidtracker import irqhost
from pysidtracker import registers as reg
from pysidtracker.asm6502 import assemble
from pysidtracker.errors import SidParseError
from pysidtracker.irqhost import (
    InterruptHost,
    InterruptPlayer,
    TunePlayer,
    host_cadence,
)
from pysidtracker.oracle import read_sidtrace
from pysidtracker.oracle import register_grid
from pysidtracker.testing import whole_song

from . import irqhost_tunes as tunes
from .helpers import build_psid, build_psid_embedded_load

PAL_LINE, PAL_FRAME = mm.PAL.cycles_per_line, reg.PAL_CYCLES_PER_FRAME
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


CIA_MASKED = QUIET + """
  LDA #$C7
  STA $DC04
  LDA #$4C
  STA $DC05
  LDA #$81
  STA $DC0D
  LDA #$11
  STA $DC0E
loop: JMP loop
"""


class _TuneError(SidParseError):
    pass


class _Tune(TunePlayer):
    error_class = _TuneError


def test_tune_player_runs_a_play_address_on_jennings():
    data = build_psid(
        bytes.fromhex("a90a8d00d460a9058d01d460"), load=0x1000, play=0x1006
    )
    player = TunePlayer(data)
    assert player.render_grid(whole_song(3)) == register_grid(data, whole_song(3))
    assert player.cycles_per_frame == player.cadence.cycles_per_call == PAL_FRAME
    writes = list(TunePlayer(data).register_writes(2))
    assert writes[0].clock == 0 and writes[-1].clock >= PAL_FRAME


@pytest.mark.parametrize("name", ["raster_kernal", "nmi_digi_oneshot"])
def test_tune_player_hosts_play_address_zero(name):
    data = tunes.build(name)
    player = TunePlayer(data)
    grid = player.render_grid(whole_song(5)) + player.render_grid(whole_song(3))
    assert grid == InterruptPlayer(data).render_grid(whole_song(8))
    fresh = TunePlayer(data)
    assert fresh.play_frame()[0] == (0, grid[0][0])
    assert fresh.snapshot() == grid[0]


@pytest.mark.parametrize("src", [CIA_MASKED, QUIET + "loop: JMP loop\n"])
def test_tune_player_without_an_entered_interrupt_is_unplayable(src):
    with pytest.raises(_TuneError, match="no interrupt"):
        _Tune(_rsid(src)).render_grid(whole_song(3))


def test_tune_player_rejects_a_bare_prg():
    with pytest.raises(_TuneError, match="bare .prg"):
        _Tune(b"\x00\x10\x60")
