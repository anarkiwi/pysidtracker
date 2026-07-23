"""The jennings 6502 host every init/replay path in this package runs on.

jennings' ``mpu6502`` is a py65-compatible 6510 that decodes every NMOS illegal
opcode natively (stock py65 stubbed them as NOPs, wrecking a defMON-style replay).
It is a bare CPU, so :func:`wire_mpu` synthesises the C64 hardware reads.
"""

from __future__ import annotations

from . import registers as reg
from .errors import EmulatorUnavailable

# addressing mode -> (MPU method returning the operand address, operand bytes)
_MODES = {
    "imm": ("ProgramCounter", 1),
    "zp": ("ZeroPageAddr", 1),
    "zpx": ("ZeroPageXAddr", 1),
    "zpy": ("ZeroPageYAddr", 1),
    "abs": ("AbsoluteAddr", 2),
    "abx": ("AbsoluteXAddr", 2),
    "aby": ("AbsoluteYAddr", 2),
    "inx": ("IndirectXAddr", 1),
    "iny": ("IndirectYAddr", 1),
}

# NOP illegals as (opcodes, operand bytes, cycles), per addressing column.
_NOPS = (
    ((0x1A, 0x3A, 0x5A, 0x7A, 0xDA, 0xFA), 0, 2),
    ((0x80, 0x82, 0x89, 0xC2, 0xE2), 1, 2),
    ((0x04, 0x44, 0x64), 1, 3),
    ((0x14, 0x34, 0x54, 0x74, 0xD4, 0xF4), 1, 4),
    ((0x0C,), 2, 4),
    ((0x1C, 0x3C, 0x5C, 0x7C, 0xDC, 0xFC), 2, 4),
)

# (opcode offset within its block, mode, cycles) of the read-modify-write column.
_RMW_MODES = (
    (0x03, "inx", 8),
    (0x07, "zp", 5),
    (0x0F, "abs", 6),
    (0x13, "iny", 8),
    (0x17, "zpx", 6),
    (0x1B, "aby", 7),
    (0x1F, "abx", 7),
)


def patch_illegals(mpu):
    """Install the stable NMOS illegal opcodes on an MPU *instance*.

    SLO/RLA/SRE/RRA/SAX/LAX/DCP/ISC, immediate ANC/ALR/ARR/SBX/SBC($EB)/LAX#/ANE,
    the unstable high-byte stores (SHY/SHX/AHX/TAS/LAS, ``reg & (hi+1)``) and the
    multi-byte NOPs. Copies the instance's tables, so only this MPU changes.
    """
    mpu.instruct = list(mpu.instruct)
    mpu.cycletime = list(mpu.cycletime)
    mpu.extracycles = list(mpu.extracycles)

    def _set(op, fn, cycles):
        mpu.instruct[op] = fn
        mpu.cycletime[op] = cycles
        mpu.extracycles[op] = 0

    def _mode(op, mode, factory, cycles):
        meth, pcadd = _MODES[mode]
        _set(op, factory(meth, pcadd), cycles)

    def slo(meth, pcadd):  # ASL mem; ORA
        def f(self):
            addr = getattr(self, meth)()
            val = self.ByteAt(addr)
            self.p = (self.p & ~self.CARRY) | (1 if val & 0x80 else 0)
            val = (val << 1) & 0xFF
            self.memory[addr] = val
            self.a |= val
            self.FlagsNZ(self.a)
            self.pc += pcadd

        return f

    def rla(meth, pcadd):  # ROL mem; AND
        def f(self):
            addr = getattr(self, meth)()
            val = self.ByteAt(addr)
            carry = 1 if self.p & self.CARRY else 0
            self.p = (self.p & ~self.CARRY) | (1 if val & 0x80 else 0)
            val = ((val << 1) | carry) & 0xFF
            self.memory[addr] = val
            self.a &= val
            self.FlagsNZ(self.a)
            self.pc += pcadd

        return f

    def sre(meth, pcadd):  # LSR mem; EOR
        def f(self):
            addr = getattr(self, meth)()
            val = self.ByteAt(addr)
            self.p = (self.p & ~self.CARRY) | (val & 1)
            val >>= 1
            self.memory[addr] = val
            self.a ^= val
            self.FlagsNZ(self.a)
            self.pc += pcadd

        return f

    def rra(meth, pcadd):  # ROR mem; ADC
        def f(self):
            addr = getattr(self, meth)()
            val = self.ByteAt(addr)
            carry = 1 if self.p & self.CARRY else 0
            newcarry = val & 1
            self.memory[addr] = (val >> 1) | (carry << 7)
            self.p = (self.p & ~self.CARRY) | newcarry
            self.opADC(lambda: addr)
            self.pc += pcadd

        return f

    def dcp(meth, pcadd):  # DEC mem; CMP
        def f(self):
            addr = getattr(self, meth)()
            self.memory[addr] = (self.ByteAt(addr) - 1) & 0xFF
            self.opCMPR(lambda: addr, self.a)
            self.pc += pcadd

        return f

    def isc(meth, pcadd):  # INC mem; SBC
        def f(self):
            addr = getattr(self, meth)()
            self.memory[addr] = (self.ByteAt(addr) + 1) & 0xFF
            self.opSBC(lambda: addr)
            self.pc += pcadd

        return f

    def sax(meth, pcadd):  # store A & X
        def f(self):
            self.memory[getattr(self, meth)()] = self.a & self.x
            self.pc += pcadd

        return f

    def lax(meth, pcadd):  # LDA + LDX
        def f(self):
            val = self.ByteAt(getattr(self, meth)())
            self.a = self.x = val
            self.FlagsNZ(val)
            self.pc += pcadd

        return f

    def hi_store(meth, pcadd, pick):  # SHY/SHX/AHX: store reg & (hi(addr)+1)
        def f(self):
            addr = getattr(self, meth)()
            self.memory[addr] = pick(self) & (((addr >> 8) + 1) & 0xFF)
            self.pc += pcadd

        return f

    def nop(pcadd):
        def f(self):
            self.pc += pcadd

        return f

    for base, factory in (
        (0x00, slo),
        (0x20, rla),
        (0x40, sre),
        (0x60, rra),
        (0xC0, dcp),
        (0xE0, isc),
    ):
        for offset, mode, cycles in _RMW_MODES:
            _mode(base + offset, mode, factory, cycles)
    for op, mode, cycles in (
        (0x83, "inx", 6),
        (0x87, "zp", 3),
        (0x8F, "abs", 4),
        (0x97, "zpy", 4),
    ):
        _mode(op, mode, sax, cycles)
    for op, mode, cycles in (
        (0xA3, "inx", 6),
        (0xA7, "zp", 3),
        (0xAF, "abs", 4),
        (0xB3, "iny", 5),
        (0xB7, "zpy", 4),
        (0xBF, "aby", 4),
    ):
        _mode(op, mode, lax, cycles)

    def i_anc(self):  # ANC #imm: A &= imm; C = bit7
        self.a &= self.ByteAt(self.ProgramCounter())
        self.FlagsNZ(self.a)
        self.p = (self.p & ~self.CARRY) | (1 if self.a & 0x80 else 0)
        self.pc += 1

    def i_alr(self):  # ALR #imm: A = (A & imm) >> 1
        self.a &= self.ByteAt(self.ProgramCounter())
        self.p = (self.p & ~self.CARRY) | (self.a & 1)
        self.a >>= 1
        self.FlagsNZ(self.a)
        self.pc += 1

    def i_arr(self):  # ARR #imm: (A & imm) ROR, with its own C/V rule
        self.a &= self.ByteAt(self.ProgramCounter())
        carry = 1 if self.p & self.CARRY else 0
        self.a = (self.a >> 1) | (carry << 7)
        self.FlagsNZ(self.a)
        self.p &= ~(self.CARRY | self.OVERFLOW)
        if self.a & 0x40:
            self.p |= self.CARRY
        if bool(self.a & 0x40) ^ bool(self.a & 0x20):
            self.p |= self.OVERFLOW
        self.pc += 1

    def i_sbx(self):  # SBX/AXS #imm: X = (A & X) - imm, CMP-style carry
        val = self.ByteAt(self.ProgramCounter())
        res = (self.a & self.x) - val
        self.x = res & 0xFF
        self.p &= ~(self.CARRY | self.ZERO | self.NEGATIVE)
        if res >= 0:
            self.p |= self.CARRY
        self.FlagsNZ(self.x)
        self.pc += 1

    def i_lax_imm(self):  # LAX #imm (unstable): stable approximation A = X = imm
        val = self.ByteAt(self.ProgramCounter())
        self.a = self.x = val
        self.FlagsNZ(val)
        self.pc += 1

    def i_ane(self):  # ANE/XAA #imm (unstable): A = X & imm
        self.a = self.x & self.ByteAt(self.ProgramCounter())
        self.FlagsNZ(self.a)
        self.pc += 1

    def i_sbc(self):  # SBC #imm alias ($EB)
        self.opSBC(self.ProgramCounter)
        self.pc += 1

    def i_tas(self):  # TAS: SP = A & X; store SP & (hi+1)
        addr = self.AbsoluteYAddr()
        self.sp = self.a & self.x & 0xFF
        self.memory[addr] = self.sp & (((addr >> 8) + 1) & 0xFF)
        self.pc += 2

    def i_las(self):  # LAS: A = X = SP = mem & SP
        val = self.ByteAt(self.AbsoluteYAddr()) & self.sp
        self.a = self.x = self.sp = val
        self.FlagsNZ(val)
        self.pc += 2

    for op, fn in (
        (0x0B, i_anc),
        (0x2B, i_anc),
        (0x4B, i_alr),
        (0x6B, i_arr),
        (0xCB, i_sbx),
        (0xAB, i_lax_imm),
        (0x8B, i_ane),
        (0xEB, i_sbc),
    ):
        _set(op, fn, 2)
    _set(0x9B, i_tas, 5)
    _set(0xBB, i_las, 4)
    _set(0x9C, hi_store("AbsoluteXAddr", 2, lambda s: s.y), 5)  # SHY
    _set(0x9E, hi_store("AbsoluteYAddr", 2, lambda s: s.x), 5)  # SHX
    _set(0x9F, hi_store("AbsoluteYAddr", 2, lambda s: s.a & s.x), 5)  # AHX abs,y
    _set(0x93, hi_store("IndirectYAddr", 1, lambda s: s.a & s.x), 5)  # AHX ind,y

    for ops, pcadd, cycles in _NOPS:
        for op in ops:
            _set(op, nop(pcadd), cycles)
    return mpu


def wire_mpu(subject, illegal_opcodes: bool = True):
    """Build a jennings MPU over the 64 KiB buffer ``subject``; return ``(mpu, mem)``.

    VIC raster ($D011/$D012) and SID osc3/env3 ($D41B/$D41C) reads are synthesised
    from the cycle counter -- plausible and monotonic, so sync spin loops exit.
    ``illegal_opcodes`` (default on, as on real NMOS) applies :func:`patch_illegals`;
    jennings decodes the illegals natively either way, so the flag is a no-op on it.

    Raises:
      EmulatorUnavailable: jennings (a core dependency) is not installed.
    """
    try:
        from jennings.devices.mpu6502 import MPU
        from jennings.memory import ObservableMemory
    except ImportError as exc:  # pragma: no cover - jennings is a core dependency
        raise EmulatorUnavailable(
            "jennings is required to run a tune: pip install pysidtracker"
        ) from exc

    mem = ObservableMemory(subject=subject)
    holder: dict = {}

    def _on_raster(addr):
        line = (holder["mpu"].processorCycles // 63) % 312
        if addr == reg.VIC_RASTER:
            return line & 0xFF
        return (subject[reg.VIC_CONTROL_1] & 0x7F) | (((line >> 8) & 1) << 7)

    def _on_sidread(addr):  # pylint: disable=unused-argument
        return (holder["mpu"].processorCycles >> 3) & 0xFF

    mem.subscribe_to_read([reg.VIC_CONTROL_1, reg.VIC_RASTER], _on_raster)
    mem.subscribe_to_read([0xD41B, 0xD41C], _on_sidread)
    mpu = MPU(memory=mem)
    holder["mpu"] = mpu
    if illegal_opcodes:
        patch_illegals(mpu)
    return mpu, mem


def run_to_rts(mpu, mem, pc: int, acc: int, max_cycles: int) -> None:
    """Run the subroutine at ``pc`` until its final RTS (or the cycle cap).

    A two-byte return address is pushed so the routine's final RTS climbs the
    stack pointer back above its start, terminating the step loop. ``acc`` is the
    accumulator on entry (the SID subtune calling convention).
    """
    start_sp = mpu.sp
    mem[0x0100 + mpu.sp] = 0x00
    mpu.sp = (mpu.sp - 1) & 0xFF
    mem[0x0100 + mpu.sp] = 0x01
    mpu.sp = (mpu.sp - 1) & 0xFF
    mpu.a = acc & 0xFF
    mpu.pc = pc
    start_cycles = mpu.processorCycles
    while mpu.sp < start_sp:
        mpu.step()
        if mpu.processorCycles - start_cycles > max_cycles:
            break


__all__ = ["patch_illegals", "wire_mpu", "run_to_rts"]
