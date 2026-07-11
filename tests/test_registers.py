"""Tests for the hardware register map predicates and store scanner."""

from pysidtracker import RegisterStore, SidImage, find_register_stores
from pysidtracker import registers as reg


def _image(payload, load=0x1000):
    return SidImage.from_prg(bytes([load & 0xFF, load >> 8]) + bytes(payload))


def test_sid_predicates_and_mirror():
    assert reg.is_sid_reg(0xD400)
    assert reg.is_sid_reg(0xD418)
    assert not reg.is_sid_reg(0xD419)  # gap before the $20 mirror
    assert reg.is_sid_reg(0xD420)  # first mirror register
    assert reg.is_sid_reg(0xD7F8)
    assert not reg.is_sid_reg(0xD800)
    assert reg.sid_register(0xD432) == 0xD412
    assert reg.is_sid_control(0xD404)
    assert reg.is_sid_control(0xD40B)
    assert reg.is_sid_control(0xD412)
    assert not reg.is_sid_control(0xD415)


def test_cia_vic_and_vector_predicates():
    assert reg.is_cia1_reg(0xDC0E)
    assert reg.is_cia2_reg(0xDD0F)
    assert reg.is_cia_timer(0xDC04)
    assert reg.is_cia_timer(0xDD05)
    assert not reg.is_cia_timer(0xDC0D)
    assert reg.is_vic_reg(0xD012)
    assert not reg.is_vic_reg(0xD030)
    assert reg.is_irq_vector(0x0314)
    assert reg.is_nmi_vector(0x0319)
    assert reg.is_cpu_vector(0xFFFE)
    assert not reg.is_cpu_vector(0xFFF9)
    assert list(reg.sid_write_band(lo=0xD400, hi=0xD402)) == [0xD400, 0xD401, 0xD402]


def test_find_register_stores():
    # STA $DC04 ; STA $D412 ; STA $0314 ; LDA (irrelevant) ; STA $D400,X
    payload = [
        0x8D,
        0x04,
        0xDC,  # STA $DC04
        0x8D,
        0x12,
        0xD4,  # STA $D412
        0x8D,
        0x14,
        0x03,  # STA $0314
        0x9D,
        0x00,
        0xD4,  # STA $D400,X
    ]
    img = _image(payload)
    watched = [reg.CIA1_TIMER_A_LO, reg.SID_CTRL_V3, reg.RAM_IRQ_VECTOR_LO, 0xD400]
    stores = find_register_stores(img, watched)
    assert stores == [
        RegisterStore(0x1000, 0x8D, 0xDC04, "STA"),
        RegisterStore(0x1003, 0x8D, 0xD412, "STA"),
        RegisterStore(0x1006, 0x8D, 0x0314, "STA"),
        RegisterStore(0x1009, 0x9D, 0xD400, "STA"),
    ]
    # Only stores to the requested addresses are reported.
    assert find_register_stores(img, [0xD415]) == []


def test_hardware_timing_and_layout_constants():
    assert reg.PAL_CYCLES_PER_FRAME == 19656
    assert reg.NTSC_CYCLES_PER_FRAME == 17095
    assert reg.PAL_CLOCK_HZ == 985248
    assert reg.NTSC_CLOCK_HZ == 1022727
    assert reg.PW_HI_REGS == (0x03, 0x0A, 0x11)
    assert reg.SID_VOICE_OFFSET == (0, 7, 14)
    assert reg.SID_REG_COUNT == 25
    assert reg.SID_BASE == 0xD400
    # PW-hi registers are the pulse-width-high byte of each voice.
    assert all(
        off + 0x03 == pw for off, pw in zip(reg.SID_VOICE_OFFSET, reg.PW_HI_REGS)
    )


def test_cycles_per_frame_for_flags():
    # PSID clock bits (2-3): 1=PAL, 2=NTSC, 0=unknown, 3=both.
    assert reg.cycles_per_frame_for_flags(0b0100) == reg.PAL_CYCLES_PER_FRAME
    assert reg.cycles_per_frame_for_flags(0b1000) == reg.NTSC_CYCLES_PER_FRAME
    assert reg.cycles_per_frame_for_flags(0b0000) == reg.PAL_CYCLES_PER_FRAME
    assert reg.cycles_per_frame_for_flags(0b1100) == reg.PAL_CYCLES_PER_FRAME
    # unrelated bits are ignored
    assert (
        reg.cycles_per_frame_for_flags(0b100000 | 0b1000) == reg.NTSC_CYCLES_PER_FRAME
    )
