"""Tests for the shared MOS 6502 primitives."""

from pysidtracker import OP_LEN, SidWriteCapturingMemory, adc, s8, sbc, walk_until
from pysidtracker import mos6502


def test_op_len_table():
    assert len(OP_LEN) == 256
    assert set(OP_LEN) == {1, 2, 3}
    # Known lengths.
    assert OP_LEN[0xA9] == 2  # LDA #imm
    assert OP_LEN[0x8D] == 3  # STA abs
    assert OP_LEN[0x60] == 1  # RTS
    assert OP_LEN[0x4C] == 3  # JMP abs
    assert OP_LEN[0xEA] == 1  # NOP


def test_opcode_classes():
    assert mos6502.is_branch(0xD0)
    assert not mos6502.is_branch(0xEA)
    assert mos6502.is_inert(0xA9)
    assert mos6502.is_abs_store(0x8D)
    assert mos6502.is_zp_store(0x85)
    assert not mos6502.is_zp_store(0x8D)
    assert mos6502.disasm_len(0x8D) == 3
    assert mos6502.disasm_len(0x100) is None


def test_s8():
    assert s8(0x00) == 0
    assert s8(0x7F) == 127
    assert s8(0x80) == -128
    assert s8(0xFF) == -1


def test_adc_sbc():
    assert adc(0xFF, 0x01, 0) == (0x00, 1)
    assert adc(0x10, 0x20, 1) == (0x31, 0)
    assert sbc(0x00, 0x01, 1) == (0xFF, 0)
    assert sbc(0x05, 0x03, 1) == (0x02, 1)


def test_sid_write_capturing_memory():
    mem = SidWriteCapturingMemory(b"\xa9\x01", 0x1000)
    assert mem.r(0x1000) == 0xA9
    mem.w(0xD405, 0x11)
    mem.w(0x2000, 0x22)  # not a SID write
    mem.w(0xD418, 0x0F)
    assert mem.take() == [(5, 0x11), (0x18, 0x0F)]
    assert mem.take() == []  # cleared


def test_walk_until():
    code = bytes([0xA9, 0x01, 0x8D, 0x00, 0xD4, 0x60, 0xFF])
    assert list(walk_until(code, 0, {0x60})) == [0, 2, 5]
    # budget stops the walk.
    assert list(walk_until(bytes([0xEA] * 10), 0, set(), budget=3)) == [0, 1, 2]
