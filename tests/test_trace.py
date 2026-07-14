"""Tests for the py65 init/timer-vector tracer."""

import pytest

from pysidtracker import InitTrace, SidImage, SidParseError, trace_init

from .helpers import build_prg, build_psid

py65 = pytest.importorskip("py65")


def _lda_sta(value, addr):
    """LDA #value ; STA addr (absolute)."""
    return bytes([0xA9, value, 0x8D, addr & 0xFF, addr >> 8])


def _init_code():
    """Init that latches the CIA timer, installs an IRQ vector, sets raster, RTS."""
    return (
        _lda_sta(0x24, 0xDC04)  # Timer-A lo latch
        + _lda_sta(0x40, 0xDC05)  # Timer-A hi latch
        + _lda_sta(0x00, 0x0314)  # IRQ vector lo
        + _lda_sta(0x20, 0x0315)  # IRQ vector hi
        + _lda_sta(0x37, 0xD012)  # raster compare
        + bytes([0x60])  # RTS
    )


def test_trace_captures_latch_vector_and_raster():
    data = build_psid(_init_code(), load=0x1000, init=0x1000, play=0x1000)
    trace = trace_init(SidImage.from_bytes(data))
    assert isinstance(trace, InitTrace)
    assert trace.cia1_timer_latch == 0x4024
    assert trace.cia2_timer_latch is None
    assert trace.irq_vector == 0x2000
    assert trace.nmi_vector is None
    assert trace.vic_raster == 0x37
    assert 0xDC04 in trace.registers_touched
    assert 0x0314 in trace.registers_touched


def test_trace_raster_folds_control1_high_bit():
    code = _lda_sta(0x37, 0xD012) + _lda_sta(0x80, 0xD011) + bytes([0x60])
    data = build_psid(code, load=0x1000, init=0x1000, play=0x1000)
    trace = trace_init(SidImage.from_bytes(data))
    # $D011 bit7 is raster compare bit 8 -> 0x37 | 0x100.
    assert trace.vic_raster == 0x137


def test_trace_play_calls_and_sid_writes():
    # init RTS immediately; play writes a SID register each call.
    init = bytes([0x60])
    play = _lda_sta(0x0F, 0xD418) + bytes([0x60])
    code = init + play
    data = build_psid(code, load=0x1000, init=0x1000, play=0x1001)
    trace = trace_init(SidImage.from_bytes(data), play_calls=2)
    assert trace.sid_writes.get(0xD418) == 0x0F
    assert trace.cia1_timer_latch is None


def test_trace_captures_cia_control_and_icr():
    code = (
        _lda_sta(0x11, 0xDC0E)  # CIA1 control A (START set, continuous)
        + _lda_sta(0x81, 0xDC0D)  # CIA1 ICR (set Timer-A enable)
        + _lda_sta(0x08, 0xDD0E)  # CIA2 control A (one-shot)
        + _lda_sta(0x01, 0xDD0D)  # CIA2 ICR (clear Timer-A enable)
        + bytes([0x60])
    )
    data = build_psid(code, load=0x1000, init=0x1000, play=0x1000)
    trace = trace_init(SidImage.from_bytes(data))
    assert trace.cia1_control == 0x11
    assert trace.cia1_icr == 0x81
    assert trace.cia2_control == 0x08
    assert trace.cia2_icr == 0x01


def test_trace_cia_control_and_icr_default_none():
    trace = trace_init(SidImage.from_bytes(build_psid(bytes([0x60]), load=0x1000)))
    assert trace.cia1_control is None
    assert trace.cia1_icr is None
    assert trace.cia2_control is None
    assert trace.cia2_icr is None


def test_trace_requires_header():
    img = SidImage.from_bytes(build_prg(bytes([0x60]), load=0x1000))
    with pytest.raises(SidParseError):
        trace_init(img)
