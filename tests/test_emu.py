"""Tests for the shared jennings host: NMOS illegal opcodes and the C64 read model."""

import pytest

from pysidtracker import run_to_rts, wire_mpu
from pysidtracker.errors import EmulatorUnavailable

jennings = pytest.importorskip("jennings")

CODE = 0x1000
TARGET = 0x0080  # zero-page operand for the read-modify-write illegals


def _run(code, *, a=0, x=0, y=0, sp=0xFF, p=0x20, mem_at=None, illegals=True):
    """Assemble ``code`` at $1000, step it once, return ``(mpu, subject)``."""
    subject = bytearray(0x10000)
    subject[CODE : CODE + len(code)] = bytes(code)
    if mem_at:
        for addr, val in mem_at.items():
            subject[addr] = val
    mpu, _ = wire_mpu(subject, illegal_opcodes=illegals)
    mpu.a, mpu.x, mpu.y, mpu.sp, mpu.p, mpu.pc = a, x, y, sp, p, CODE
    mpu.step()
    return mpu, subject


def test_sbx_computes_x_and_carry():
    # SBX #$01: X = (A & X) - imm, CMP-style carry.
    mpu, _ = _run([0xCB, 0x01], a=0xFF, x=0x5C)
    assert mpu.x == 0x5B
    assert mpu.p & mpu.CARRY


def test_sbx_borrow_clears_carry():
    mpu, _ = _run([0xCB, 0x10], a=0xFF, x=0x01)
    assert mpu.x == 0xF1
    assert not mpu.p & mpu.CARRY


@pytest.mark.parametrize(
    "opcode,a_in,mem_in,mem_out,a_out",
    [
        (0x07, 0x01, 0x81, 0x02, 0x03),  # SLO: ASL mem, ORA
        (0x27, 0xFF, 0x40, 0x80, 0x80),  # RLA: ROL mem, AND
        (0x47, 0xF0, 0x03, 0x01, 0xF1),  # SRE: LSR mem, EOR
        (0xC7, 0x05, 0x06, 0x05, 0x05),  # DCP: DEC mem, CMP (A unchanged)
    ],
)
def test_rmw_illegals(opcode, a_in, mem_in, mem_out, a_out):
    mpu, subject = _run([opcode, TARGET], a=a_in, mem_at={TARGET: mem_in})
    assert subject[TARGET] == mem_out
    assert mpu.a == a_out


def test_rra_rotates_then_adds():
    # ROR $80 (carry clear): $02 -> $01; then ADC -> A = 1 + 1.
    mpu, subject = _run([0x67, TARGET], a=0x01, mem_at={TARGET: 0x02})
    assert subject[TARGET] == 0x01
    assert mpu.a == 0x02


def test_isc_increments_then_subtracts():
    # INC $80: $01 -> $02; then SBC (carry set) -> A = 5 - 2.
    mpu, subject = _run([0xE7, TARGET], a=0x05, p=0x21, mem_at={TARGET: 0x01})
    assert subject[TARGET] == 0x02
    assert mpu.a == 0x03


def test_sax_stores_a_and_x():
    _, subject = _run([0x87, TARGET], a=0xF0, x=0x3C)
    assert subject[TARGET] == 0x30


def test_lax_loads_a_and_x():
    mpu, _ = _run([0xA7, TARGET], mem_at={TARGET: 0x42})
    assert mpu.a == mpu.x == 0x42


def test_anc_sets_carry_from_bit7():
    mpu, _ = _run([0x0B, 0xF0], a=0x80)
    assert mpu.a == 0x80
    assert mpu.p & mpu.CARRY


def test_alr_ands_then_shifts():
    mpu, _ = _run([0x4B, 0x0F], a=0xFF)
    assert mpu.a == 0x07
    assert mpu.p & mpu.CARRY


def test_arr_rotates_in_carry():
    mpu, _ = _run([0x6B, 0xFF], a=0x80, p=0x21)
    assert mpu.a == 0xC0
    assert mpu.p & mpu.CARRY


def test_sbc_alias_and_lax_immediate_and_ane():
    # LXA ($AB) and ANE ($8B) are the unstable magic-constant group: jennings models (A | 0xEE) & ... (NMS).
    assert _run([0xEB, 0x01], a=0x05, p=0x21)[0].a == 0x04  # SBC alias: stable
    mpu, _ = _run([0xAB, 0x77])  # LXA: (A | 0xEE) & imm = 0xEE & 0x77
    assert mpu.a == mpu.x == 0x66
    assert _run([0x8B, 0x0F], x=0x33)[0].a == 0x02  # ANE: (A | 0xEE) & X & imm


def test_high_byte_stores_and_stack_illegals():
    _, subject = _run([0x9C, 0x00, 0x20], y=0xFF)  # SHY $2000,x
    assert subject[0x2000] == 0x21
    _, subject = _run([0x9E, 0x00, 0x20], x=0xFF)  # SHX $2000,y
    assert subject[0x2000] == 0x21
    _, subject = _run([0x9F, 0x00, 0x20], a=0xFF, x=0xFF)  # AHX $2000,y
    assert subject[0x2000] == 0x21
    mpu, subject = _run([0x9B, 0x00, 0x20], a=0xFF, x=0x0F)  # TAS $2000,y
    assert mpu.sp == 0x0F and subject[0x2000] == 0x01
    mpu, _ = _run([0xBB, 0x00, 0x20], sp=0xF0, mem_at={0x2000: 0x3F})  # LAS
    assert mpu.a == mpu.x == mpu.sp == 0x30


def test_ahx_indirect_y():
    # SHA/AHX ($93) ANDs with high(pointer $1FFF)+1 = $20, not the effective address $2000.
    _, subject = _run(
        [0x93, 0x10], a=0xFF, x=0xFF, y=0x01, mem_at={0x10: 0xFF, 0x11: 0x1F}
    )
    assert subject[0x2000] == 0x20


@pytest.mark.parametrize(
    "code,length",
    [
        ([0x1A], 1),  # implied NOP
        ([0x80, 0x00], 2),  # immediate NOP
        ([0x04, 0x00], 2),  # zero-page NOP
        ([0x14, 0x00], 2),  # zero-page,x NOP
        ([0x0C, 0x00, 0x20], 3),  # absolute NOP
        ([0x1C, 0x00, 0x20], 3),  # absolute,x NOP
    ],
)
def test_multi_byte_nops_advance_pc(code, length):
    mpu, _ = _run(code)
    assert mpu.pc == CODE + length


def test_illegals_off_still_decoded_natively_by_jennings():
    # jennings runs SBX natively even with the pysidtracker patch withheld: X = (A & X) - imm.
    mpu, _ = _run([0xCB, 0x01], a=0xFF, x=0x5C, illegals=False)
    assert mpu.x == 0x5B
    assert mpu.p & mpu.CARRY


def test_hardware_reads_are_synthesised_not_ram():
    subject = bytearray(0x10000)
    subject[CODE : CODE + 3] = bytes([0xAD, 0x12, 0xD0])  # LDA $D012
    mpu, mem = wire_mpu(subject)
    seen = set()
    for _ in range(8):
        mpu.pc = CODE
        mpu.step()
        seen.add(mpu.a)
        mpu.processorCycles += 63
    assert len(seen) > 1  # a raster spin loop can observe its compare value
    assert mem[0xD41B] != subject[0xD41B]  # osc3 reads the cycle counter, not RAM
    assert mem[0xD011] & 0x80 == 0x00


def test_run_to_rts_honours_cycle_cap():
    subject = bytearray(0x10000)
    subject[CODE : CODE + 2] = bytes([0x4C, 0x00])  # JMP $1000 (endless)
    subject[CODE + 2] = 0x10
    mpu, mem = wire_mpu(subject)
    run_to_rts(mpu, mem, CODE, 0, max_cycles=1000)
    assert mpu.processorCycles > 1000


def test_wire_mpu_reports_missing_jennings(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _no_jennings(name, *args, **kwargs):
        if name.startswith("jennings"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_jennings)
    with pytest.raises(EmulatorUnavailable):
        wire_mpu(bytearray(0x10000))
