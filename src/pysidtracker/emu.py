"""The jennings 6502 host every init/replay path in this package runs on.

jennings' ``mpu6502`` is a py65-compatible 6510 that decodes every NMOS illegal
opcode natively, byte/cycle-exact. It is a bare CPU, so :func:`wire_mpu`
synthesises the C64 hardware reads.
"""

from __future__ import annotations

from . import registers as reg
from .errors import EmulatorUnavailable


def wire_mpu(subject, illegal_opcodes: bool = True):  # pylint: disable=unused-argument
    """Build a jennings MPU over the 64 KiB buffer ``subject``; return ``(mpu, mem)``.

    VIC raster ($D011/$D012) and SID osc3/env3 ($D41B/$D41C) reads are synthesised
    from the cycle counter, so sync spin loops exit. ``illegal_opcodes`` is kept
    for API compatibility; jennings decodes the NMOS illegals natively regardless.

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


__all__ = ["wire_mpu", "run_to_rts"]
