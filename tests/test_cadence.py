"""Tests for the interrupt-programmed play-routine cadence."""

import pytest

from pysidtracker import Cadence, SidImage, TriggerSource, playroutine_cadence
from pysidtracker import registers as reg
from pysidtracker.cadence import _driver_base, _Machine
from pysidtracker.errors import SidParseError
from pysidtracker.testing import resolve_tune

from .helpers import build_prg, build_psid, build_psid_embedded_load

LOAD = 0x1000
PAL = reg.PAL_CYCLES_PER_FRAME
RTS, RTI, SEI, CLI = b"\x60", b"\x40", b"\x78", b"\x58"
PLAY_SID = bytes((0xEE, 0x00, 0xD4)) + RTS  # INC $D400 ; RTS
ACK_VIC = bytes((0x0E, 0x19, 0xD0))  # ASL $D019
ACK_CIA2 = bytes((0xAD, 0x0D, 0xDD))  # LDA $DD0D


def sta(value, addr):
    """LDA #value ; STA addr."""
    return bytes((0xA9, value, 0x8D, addr & 0xFF, addr >> 8))


def vector(addr_lo, target):
    """Store ``target`` to the little-endian vector at ``addr_lo``."""
    return sta(target & 0xFF, addr_lo) + sta(target >> 8, addr_lo + 1)


def latch(value, base=reg.CIA1_BASE):
    """Program Timer A's latch."""
    return sta(value & 0xFF, base + 4) + sta(value >> 8, base + 5)


def psid(init, play=PLAY_SID, **kwargs):
    """A PSID whose init (ending RTS) is followed by its play routine."""
    init += RTS
    return build_psid(init + play, LOAD, LOAD, LOAD + len(init), **kwargs)


def image(**parts):
    """A 256-byte image at ``LOAD`` with each ``name_OFF=bytes`` part at offset OFF."""
    body = bytearray(0x100)
    for name, code in parts.items():
        off = int(name.rsplit("_", 1)[1], 16)
        body[off : off + len(code)] = code
    return bytes(body)


def rsid(body):
    """An RSID whose init is at ``LOAD`` (the header load address is embedded)."""
    return build_psid_embedded_load(body, LOAD, init=LOAD, play=0, magic=b"RSID")


def _cad(data, **kwargs):
    return playroutine_cadence(data, **kwargs)


def test_psid_vbi_bit_with_no_init_programming_is_one_raster_per_frame():
    cad = _cad(psid(b""))
    assert (cad.source, cad.cycles_per_call) == (TriggerSource.PAL_VIDEO, PAL)
    assert (cad.calls_per_frame, cad.raster_lines) == (1.0, (0,))
    assert cad.clock_hz == reg.PAL_CLOCK_HZ and cad.latch is None


@pytest.mark.parametrize(
    "clock,latch_value",
    [("PAL", reg.KERNAL_CIA1_LATCH_PAL), ("NTSC", reg.KERNAL_CIA1_LATCH_NTSC)],
)
def test_psid_cia_bit_with_no_init_programming_is_the_driver_timer(clock, latch_value):
    cad = _cad(psid(b"", speed=1), clock=clock)
    assert (cad.source, cad.latch) == (TriggerSource.CIA_TIMER, latch_value)
    assert cad.cycles_per_call == latch_value + 1
    assert cad.calls_per_second == pytest.approx(cad.clock_hz / (latch_value + 1))


def test_speed_bit_is_per_subtune():
    data = psid(b"", speed=0b10, songs=2)
    assert _cad(data).source is TriggerSource.PAL_VIDEO
    assert _cad(data, subtune=1).source is TriggerSource.CIA_TIMER


def test_init_overrides_vbi_bit_with_an_armed_cia_timer():
    init = sta(0x00, reg.VIC_IRQ_ENABLE) + latch(0x5BF9)
    init += sta(0x81, reg.CIA1_ICR) + sta(0x11, reg.CIA1_CRA)
    cad = _cad(psid(init))
    assert (cad.source, cad.cycles_per_call) == (TriggerSource.CIA_TIMER, 0x5BFA)
    assert not cad.dynamic


def test_init_overrides_cia_bit_with_a_raster_irq():
    init = sta(0x7F, reg.CIA1_ICR) + sta(0x01, reg.VIC_IRQ_ENABLE)
    init += sta(0x80, reg.VIC_RASTER)
    cad = _cad(psid(init, speed=1))
    assert (cad.source, cad.raster_lines) == (TriggerSource.PAL_VIDEO, (0x80,))


def test_a_latch_alone_does_not_start_the_vbi_drivers_stopped_timer():
    cad = _cad(psid(latch(0x5BF9)))
    assert cad.source is TriggerSource.PAL_VIDEO


def test_cia_multispeed_latch():
    cad = _cad(psid(latch(PAL // 4 - 1), speed=1))
    assert (cad.source, cad.cycles_per_call) == (TriggerSource.CIA_TIMER, PAL // 4)
    assert cad.calls_per_frame == 4.0


def test_multispeed_raster_chain_counts_calls_per_frame():
    lines = (0x10, 0x30, 0x90, 0xF0)
    play = (
        bytes((0xAE, 0x7F, LOAD >> 8))  # LDX LOAD+$7F
        + bytes((0xBD, 0x80, LOAD >> 8))  # LDA LOAD+$80,X
        + bytes((0x8D, 0x12, 0xD0))  # STA $D012
        + bytes((0xE8, 0x8A, 0x29, 0x03, 0x8D, 0x7F, LOAD >> 8))  # idx = idx+1 & 3
        + PLAY_SID
    )
    body = image(init_00=sta(lines[0], reg.VIC_RASTER) + RTS, play_20=play)
    body = body[:0x80] + bytes(lines) + body[0x84:]
    cad = _cad(build_psid(body, LOAD, LOAD, LOAD + 0x20))
    assert cad.source is TriggerSource.PAL_VIDEO
    assert (cad.calls_per_frame, cad.raster_lines) == (4.0, lines)
    assert cad.cycles_per_call == PAL // 4


def test_disarmed_timer_with_no_raster_plays_nothing():
    for disarm in (sta(0x01, reg.CIA1_ICR), sta(0x00, reg.CIA1_CRA)):
        cad = _cad(psid(disarm, speed=1))
        assert cad.source is None
        assert cad.cycles_per_call == PAL


def test_one_shot_timer_is_not_periodic():
    cad = _cad(psid(sta(0x19, reg.CIA1_CRA), speed=1))
    assert cad.source is None


def test_one_shot_timer_restarted_by_play_is_measured():
    play = sta(0x19, reg.CIA1_CRA) + PLAY_SID
    cad = _cad(psid(latch(4999), play, speed=1))
    assert cad.source is TriggerSource.CIA_TIMER
    assert 5000 < cad.cycles_per_call < 5100


def test_play_that_never_writes_the_sid_is_unknown():
    assert _cad(psid(b"", RTS)).source is None


def test_dynamic_when_play_alternates_the_latch():
    play = (
        bytes((0xAD, 0xF0, 0x10, 0x49, 0x01, 0x8D, 0xF0, 0x10))  # flip a RAM bit
        + bytes((0x09, 0x20, 0x8D, 0x05, 0xDC))  # latch hi = $20 | bit
        + PLAY_SID
    )
    cad = _cad(psid(latch(0x2000), play, speed=1))
    assert cad.source is TriggerSource.CIA_TIMER and cad.dynamic
    assert abs(cad.cycles_per_call - (0x2001 + 0x2101) // 2) < 0x80


def _raster_handler(exit_code):
    return bytes((0xEE, 0x00, 0xD4)) + ACK_VIC + exit_code


IDLE = bytes((0x4C, 0x40, LOAD >> 8))  # JMP LOAD+$40 (at LOAD+$40)


def test_rsid_raster_through_the_kernal_vector():
    init = SEI + sta(0x7F, reg.CIA1_ICR) + sta(0x01, reg.VIC_IRQ_ENABLE)
    init += sta(0x40, reg.VIC_RASTER) + sta(0x1B, reg.VIC_CONTROL_1)
    init += vector(reg.RAM_IRQ_VECTOR_LO, LOAD + 0x80) + CLI + RTS
    handler = _raster_handler(bytes((0x4C, 0x31, 0xEA)))  # JMP $EA31
    cad = _cad(rsid(image(init_00=init, irq_80=handler)))
    assert (cad.source, cad.raster_lines) == (TriggerSource.PAL_VIDEO, (0x40,))


def test_rsid_raster_through_the_hardware_vector_with_kernal_banked_out():
    init = SEI + sta(0x35, 0x01) + sta(0x7F, reg.CIA1_ICR)
    init += sta(0x01, reg.VIC_IRQ_ENABLE)
    init += vector(reg.CPU_IRQ_VECTOR, LOAD + 0x80) + CLI + bytes((0x4C, 0x40, 0x10))
    cad = _cad(rsid(image(init_00=init, idle_40=IDLE, irq_80=_raster_handler(RTI))))
    assert (cad.source, cad.raster_lines) == (TriggerSource.PAL_VIDEO, (311,))


def test_rsid_nmi_timer():
    init = sta(0x7F, reg.CIA1_ICR) + latch(PAL // 2 - 1, reg.CIA2_BASE)
    init += vector(reg.RAM_NMI_VECTOR_LO, LOAD + 0x80)
    init += sta(0x81, reg.CIA2_ICR) + sta(0x11, reg.CIA2_CRA) + RTS
    handler = bytes((0xEE, 0x00, 0xD4)) + ACK_CIA2 + RTI
    cad = _cad(rsid(image(init_00=init, nmi_80=handler)))
    assert (cad.source, cad.cycles_per_call) == (TriggerSource.NMI_TIMER, PAL // 2)
    assert cad.calls_per_frame == 2.0


def test_a_raster_player_beats_a_faster_nmi_sample_player():
    init = sta(0x7F, reg.CIA1_ICR) + sta(0x01, reg.VIC_IRQ_ENABLE)
    init += vector(reg.RAM_IRQ_VECTOR_LO, LOAD + 0x80)
    init += latch(PAL // 8 - 1, reg.CIA2_BASE)
    init += vector(reg.RAM_NMI_VECTOR_LO, LOAD + 0xC0)
    init += sta(0x81, reg.CIA2_ICR) + sta(0x11, reg.CIA2_CRA) + CLI + RTS
    music = bytes((0xEE, 0x00, 0xD4, 0xEE, 0x01, 0xD4)) + ACK_VIC
    digi = bytes((0xEE, 0x18, 0xD4)) + ACK_CIA2 + RTI
    body = image(init_00=init, irq_80=music + bytes((0x4C, 0x81, 0xEA)), nmi_C0=digi)
    cad = _cad(rsid(body))
    assert (cad.source, cad.calls_per_frame) == (TriggerSource.PAL_VIDEO, 1.0)


def test_an_unacknowledged_nmi_is_raised_once():
    init = sta(0x7F, reg.CIA1_ICR) + latch(PAL // 2 - 1, reg.CIA2_BASE)
    init += vector(reg.RAM_NMI_VECTOR_LO, LOAD + 0x80)
    init += sta(0x81, reg.CIA2_ICR) + sta(0x11, reg.CIA2_CRA) + RTS
    cad = _cad(rsid(image(init_00=init, nmi_80=bytes((0xEE, 0x00, 0xD4)) + RTI)))
    assert cad.source is None


def test_ntsc_header_and_explicit_clock():
    cad = _cad(psid(b"", flags=0b10 << 2))
    assert (cad.source, cad.cycles_per_call) == (
        TriggerSource.NTSC_VIDEO,
        reg.NTSC_CYCLES_PER_FRAME,
    )
    assert _cad(psid(b""), clock="ntsc").clock_hz == reg.NTSC_CLOCK_HZ
    with pytest.raises(ValueError):
        _cad(psid(b""), clock="SECAM")


def test_accepts_a_sidimage_and_rejects_a_bare_prg():
    assert _cad(SidImage.from_bytes(psid(b""))).source is TriggerSource.PAL_VIDEO
    with pytest.raises(SidParseError):
        _cad(build_prg(RTS, LOAD))


def test_cadence_defaults_calls_per_frame():
    cad = Cadence(PAL // 2, TriggerSource.CIA_TIMER, reg.PAL_CLOCK_HZ)
    assert cad.calls_per_frame == 2.0
    assert cad.calls_per_second == pytest.approx(reg.PAL_CLOCK_HZ / (PAL // 2))


# Real HVSC tunes whose cadence sidplayfp's interrupt rows confirm.
_REAL = {
    "MUSICIANS/H/Honey/8_Bit-Maerchenland_V2.sid": (TriggerSource.CIA_TIMER, 16641),
    "MUSICIANS/F/Fern_Eric/Goldberg_Variations_parts_1-7.sid": (
        TriggerSource.CIA_TIMER,
        17431,
    ),
    "MUSICIANS/E/Eeben_Aleksi/Stella_2600_by_Starlight.sid": (
        TriggerSource.CIA_TIMER,
        19760,
    ),
    "MUSICIANS/G/Goto80/Automatas.sid": (TriggerSource.CIA_TIMER, 2457),
}


@pytest.mark.parametrize("relpath,expected", list(_REAL.items()))
def test_real_tune_cadence(relpath, expected, tmp_path):
    path = resolve_tune(relpath, cache_dir=tmp_path)
    if path is None:
        pytest.skip("HVSC mirror unreachable")
    cad = _cad(path.read_bytes())
    assert (cad.source, cad.cycles_per_call) == expected


def test_writes_with_io_banked_out_reach_ram_not_the_chips():
    init = sta(0x34, 0x01) + sta(0x01, reg.VIC_IRQ_ENABLE) + sta(0x7F, reg.CIA1_ICR)
    init += sta(0x37, 0x01)
    cad = _cad(psid(init, speed=1))
    assert cad.source is TriggerSource.CIA_TIMER


def test_driver_page_is_the_headers_free_page_else_the_first_free_page():
    data = bytearray(psid(b""))
    assert _driver_base(SidImage.from_bytes(bytes(data))) == 0x0400
    data[0x78] = 0x20
    assert _driver_base(SidImage.from_bytes(bytes(data))) == 0x2000
    low = build_psid(RTS, 0x0400, 0x0400, 0x0400)
    assert _driver_base(SidImage.from_bytes(low)) == 0x0500


def test_nmis_nested_in_a_timer_irq_do_not_take_its_writes():
    init = sta(0x7F, reg.CIA1_ICR) + latch(PAL - 1)
    init += sta(0x81, reg.CIA1_ICR) + sta(0x11, reg.CIA1_CRA)
    init += vector(reg.RAM_IRQ_VECTOR_LO, LOAD + 0x80)
    init += latch(144, reg.CIA2_BASE) + vector(reg.RAM_NMI_VECTOR_LO, LOAD + 0xF0)
    init += sta(0x81, reg.CIA2_ICR) + sta(0x11, reg.CIA2_CRA) + CLI + RTS
    delay = bytes((0xA2, 0x30, 0xCA, 0xD0, 0xFD))  # LDX #$30 ; DEX ; BNE *-1
    music = b"".join(bytes((0xEE, r, 0xD4)) + delay for r in range(7))
    music += bytes((0xAD, 0x0D, 0xDC, 0x4C, 0x81, 0xEA))  # LDA $DC0D ; JMP $EA81
    digi = bytes((0xEE, 0x18, 0xD4)) + ACK_CIA2 + RTI
    cad = _cad(rsid(image(init_00=init, irq_80=music, nmi_F0=digi)))
    assert (cad.source, cad.cycles_per_call) == (TriggerSource.CIA_TIMER, PAL)


def test_a_brk_inside_a_handler_runs_on_its_behalf():
    init = SEI + sta(0x35, 0x01) + sta(0x7F, reg.CIA1_ICR)
    init += sta(0x01, reg.VIC_IRQ_ENABLE)
    init += vector(reg.CPU_IRQ_VECTOR, LOAD + 0x80) + CLI + bytes((0x4C, 0x40, 0x10))
    pull = bytes((0x68, 0xAA, 0x68)) + RTI  # PLA ; TAX ; PLA ; RTI
    handler = bytes((0x48, 0x8A, 0x48, 0xBA, 0xBD, 0x03, 0x01, 0x29, 0x10))
    handler += bytes((0xD0, 0x0C, 0x00, 0xEA, 0xEE, 0x01, 0xD4)) + ACK_VIC + pull
    brk = bytes((0xEE, 0x00, 0xD4)) + pull  # B set: PHA/TXA/PHA/TSX/LDA $0103,X
    body = image(init_00=init, idle_40=IDLE, irq_80=handler, brk_97=brk)
    machine = _Machine(SidImage.from_bytes(rsid(body)), False, 2_000_000)
    machine.driver(SidImage.from_bytes(rsid(body)).header, 0, False)
    machine.execute(4)
    assert {wrote for *_, wrote, _ in machine.events[1:]} == {0b11}
    assert machine.handlers == [None]
