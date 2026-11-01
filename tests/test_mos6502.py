import pytest

"""Tests for the shared MOS 6502 primitives."""

from pysidtracker import OP_LEN, SidWriteCapturingMemory, adc, sbc, walk_until
from pysidtracker import mos6502


@pytest.mark.strain_exempt(reason="exercises code outside the measured package")
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


class _Img:
    def __init__(self, data):
        self.mem = bytearray(0x10000)
        for addr, blob in data.items():
            self.mem[addr : addr + len(blob)] = blob

    def peek(self, addr):
        return self.mem[addr] if 0 <= addr < 0x10000 else 0


def test_walk_calls():
    img = _Img(
        {
            0x1000: b"\xea\x20\x00\x20\x60",  # NOP; JSR $2000; RTS
            0x2000: b"\x4c\x00\x30",  # JMP $3000
            0x3000: b"\x60",
            0x4000: b"\x60\x4c\x00\x40",  # RTS ends the walk
        }
    )
    assert mos6502.walk_calls(img, 0x1000, {0x3000}) == 0x3000
    assert mos6502.walk_calls(img, 0x1000, {0x5000}) is None
    assert mos6502.walk_calls(img, 0x4000, {0x4001}) is None
    img = _Img({0x1000: b"\x4c\x00\x10"})
    assert mos6502.walk_calls(img, 0x1000, {0x9999}) is None


def test_track_x_and_called_x_values():
    img = _Img(
        {
            0x1000: b"\xa2\x01\xe8\x4c\x00\x20",  # LDX #1; INX; JMP out
            0x2000: b"\x60",
            0x3000: b"\x20\x00\x20\xe8\x20\x00\x20\x4c\x00\x20",
            0x4000: b"\xea",
        }
    )
    assert mos6502.track_x(img, 0x1000, 0, 0x2000) == (2, True)
    assert mos6502.track_x(img, 0x4000, 0, 0x2000) == (0, False)
    assert mos6502.called_x_values(img, 0x3000, 0x2000) == (0, 1, 1)
    assert mos6502.called_x_values(img, 0x4000, 0x2000) == ()
