"""Tests for the interrupt-programmed play-routine cadence."""

import pytest

from vagrant_moses import memmap as mm

from pysidtracker import SidImage, TriggerSource, playroutine_cadence
from pysidtracker import registers as reg
from pysidtracker.cadence import _cadence, _driver_base, _Machine
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


def latch(value, base=mm.CIA1_BASE):
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


def test_disarmed_timer_with_no_raster_plays_nothing():
    for disarm in (sta(0x01, mm.CIA1_ICR), sta(0x00, mm.CIA1_CRA)):
        cad = _cad(psid(disarm, speed=1))
        assert cad.source is None
        assert cad.cycles_per_call == PAL


def test_one_shot_timer_restarted_by_play_is_measured():
    play = sta(0x19, mm.CIA1_CRA) + PLAY_SID
    cad = _cad(psid(latch(4999), play, speed=1))
    assert cad.source is TriggerSource.CIA_TIMER
    assert 5000 < cad.cycles_per_call < 5100


def _raster_handler(exit_code):
    return bytes((0xEE, 0x00, 0xD4)) + ACK_VIC + exit_code


IDLE = bytes((0x4C, 0x40, LOAD >> 8))  # JMP LOAD+$40 (at LOAD+$40)


def test_rsid_raster_through_the_kernal_vector():
    init = SEI + sta(0x7F, mm.CIA1_ICR) + sta(0x01, mm.VIC_IRQ_ENABLE)
    init += sta(0x40, mm.VIC_RASTER) + sta(0x1B, mm.VIC_CONTROL_1)
    init += vector(mm.RAM_IRQ_VECTOR_LO, LOAD + 0x80) + CLI + RTS
    handler = _raster_handler(bytes((0x4C, 0x31, 0xEA)))  # JMP $EA31
    cad = _cad(rsid(image(init_00=init, irq_80=handler)))
    assert (cad.source, cad.raster_lines) == (TriggerSource.PAL_VIDEO, (0x40,))


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
    init = sta(0x34, 0x01) + sta(0x01, mm.VIC_IRQ_ENABLE) + sta(0x7F, mm.CIA1_ICR)
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


def _nmi_tune(target, nmi_exit=RTI):
    """An RSID NMI timer vectored to ``target``; the handler at ``LOAD + 0x80``
    bumps ``$D400`` and leaves by ``nmi_exit`` (``$DD0C`` holds an RTI)."""
    init = sta(0x7F, mm.CIA1_ICR) + latch(PAL // 2 - 1, mm.CIA2_BASE)
    init += sta(0x40, mm.CIA2_BASE + 0xC) + vector(mm.RAM_NMI_VECTOR_LO, target)
    init += sta(0x81, mm.CIA2_ICR) + sta(0x11, mm.CIA2_CRA) + RTS
    handler = bytes((0xEE, 0x00, 0xD4)) + ACK_CIA2 + nmi_exit
    return SidImage.from_bytes(rsid(image(init_00=init, nmi_80=handler)))


def _run(tune):
    machine = _Machine(tune, False, 2_000_000)
    machine.driver(tune.header, 0, False)
    machine.execute(4)
    return machine


def test_a_handler_returning_through_io_unwinds_its_frame():
    tune = _nmi_tune(LOAD + 0x80, bytes((0x4C, 0x0C, 0xDD)))  # JMP $DD0C
    assert len(_run(tune).handlers) <= 2
    cad = _cad(tune)
    assert (cad.source, cad.cycles_per_call) == (TriggerSource.NMI_TIMER, PAL // 2)


def test_a_recurring_handler_vectored_into_io_makes_the_source_unknown():
    assert _cad(_nmi_tune(LOAD + 0x80)).source == TriggerSource.NMI_TIMER
    machine = _run(_nmi_tune(mm.CIA1_BASE + 4))
    assert machine.opaque == {(1, 0)} and _cadence(machine, 0, 4).source is None
