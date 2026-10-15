"""The jennings 6502 host every init/replay path in this package runs on.

jennings' ``mpu6502`` is a py65-compatible 6510 that decodes every NMOS illegal
opcode natively, byte/cycle-exact. It is a bare CPU, so :func:`wire_mpu`
synthesises the C64 hardware reads and the ``$01`` banking the C64's PLA applies
to the ROM space (sidplayfp's ROM-less KERNAL/BASIC/CHAR; see
:class:`_BankedMemory`).
"""

from __future__ import annotations

from array import array
from typing import Optional

from jennings.memory import ObservableMemory  # core dependency

from . import registers as reg
from .errors import EmulatorUnavailable

# osc3/env3 depend only on voice 3 and its sync/ring source, voice 2.
_CHIP_INPUTS = range(reg.SID_BASE + 0x07, reg.SID_BASE + 0x15)
_CHIP_OUTPUTS = (reg.SID_BASE + 0x1B, reg.SID_BASE + 0x1C)


class SidReads:
    """A reSIDfp chip answering osc3/env3 reads at the cycle the CPU makes them.

    Register writes are queued and the chip is clocked through them only when a
    read needs it, so a tune that never reads the chip costs nothing.
    """

    def __init__(self, model: str = "6581"):
        from pyresidfp._pyresidfp import SID, ChipModel, SamplingMethod

        chip = {"6581": ChipModel.MOS6581, "8580": ChipModel.MOS8580}[model]
        self._sid = SID(chip, SamplingMethod.DECIMATE, reg.PAL_CLOCK_HZ, 8000.0)
        self._now = 0
        self._queue = array("Q")

    def _clock(self, cycle: int) -> None:
        if cycle > self._now:
            self._sid.clock(cycle - self._now)
            self._now = cycle

    def write(self, cycle: int, register: int, value: int) -> None:
        """Queue a write to SID ``register`` (offset) at CPU ``cycle``."""
        self._queue.append(cycle << 16 | register << 8 | value)

    def flush(self, cycle: int) -> None:
        """Clock the chip through every queued write up to CPU ``cycle``."""
        for word in self._queue:
            self._clock(word >> 16)
            self._sid.write((word >> 8) & 0xFF, word & 0xFF)
        del self._queue[:]
        self._clock(cycle)

    def pending(self) -> int:
        """Writes queued since the last read or flush."""
        return len(self._queue)

    def read(self, cycle: int, register: int) -> int:
        """The chip's value of ``register`` (offset) at CPU ``cycle``."""
        self.flush(cycle)
        return self._sid.read(register)


def access_cycle(mpu) -> int:
    """The cycle the executing instruction's data access lands on: its last.

    SID registers share one page, so no indexed access to them adds a
    page-crossing cycle.
    """
    memory = getattr(mpu.memory, "_subject", mpu.memory)
    return mpu.processorCycles + mpu.cycletime[memory[mpu.pc]] - 1


class _BankedMemory(ObservableMemory):
    """The flat CPU's C64 memory: reads of the ROM space follow the ``$01`` bank.

    The ROM space is sidplayfp's ROM-less ROMs, the same bytes the interrupt host
    maps (``irqhost.C64``): KERNAL is its stub, BASIC and the character ROM read
    zero. ``$01`` is the live RAM byte at ``$01`` (the tune or the PSID driver
    writes it), so a tune that re-banks to reach the RAM behind a ROM sees RAM,
    and one that banks in the I/O page sees it. Only reads are banked; writes
    always reach the ``$01`` byte and the RAM shadow beneath the ROMs.
    """

    def __init__(self, subject, roms: Optional[dict] = None):
        super().__init__(subject=subject)
        from .irqhost import kernal_stub  # lazy: irqhost imports this module

        self._kernal = kernal_stub()
        self._roms = roms

    def _rom_bytes(self, base: int) -> Optional[bytes]:
        """The real ROM at ``base`` (from the ``roms`` map), or ``None`` for stubs."""
        return self._roms.get(base) if self._roms else None

    def _rom_value(self, address: int) -> Optional[int]:
        """The ROM byte at ``address`` under the live ``$01``; ``None`` for RAM.

        A ROM space serves the real :attr:`_roms` bytes when mapped (``roms``) and
        the space is banked in, else the sidplayfp ROM-less stand-in (the KERNAL
        stub, or ``0`` for BASIC/CHAR). With no ``roms`` it is exactly those.
        """
        b = self._subject[1] & 7
        if address < 0xC000:  # $A000-$BFFF: BASIC, in when L- and H-IRAM both on
            if (b & 1) and (b & 2):
                rom = self._rom_bytes(0xA000)
                return rom[address - 0xA000] if rom is not None else 0
            return None
        if address < 0xD000:  # $C000-$CFFF: always RAM
            return None
        if address < 0xE000:  # $D000-$DFFF: CHAR ROM, else the I/O page / RAM
            if (b & 1) or (b & 2):
                if b & 4:
                    return None
                rom = self._rom_bytes(0xD000)
                return rom[address - 0xD000] if rom is not None else 0
            return None
        if b & 2:  # $E000-$FFFF: KERNAL
            rom = self._rom_bytes(0xE000)
            if rom is not None:
                return rom[address - 0xE000]
            return self._kernal[address - 0xE000]
        return None

    def __getitem__(self, address):
        if isinstance(address, int) and address >= 0xA000:
            rom = self._rom_value(address)
            if rom is not None:
                return rom
        return super().__getitem__(address)


def wire_mpu(
    subject, illegal_opcodes: bool = True, sid_model: str = "6581", ntsc: bool = False
):  # pylint: disable=unused-argument
    """Build a jennings MPU over the C64 address-space buffer ``subject``; return ``(mpu, mem)``.

    VIC raster ($D011/$D012) reads are synthesised from the cycle counter on a
    PAL (or ``ntsc``) beam, so sync spin loops exit; SID osc3/env3 ($D41B/$D41C) reads come from a
    ``sid_model`` reSIDfp chip fed the CPU's own SID writes (:class:`SidReads`).
    Reads of the C64 ROM space are ``$01``-banked (sidplayfp's ROM-less KERNAL /
    BASIC / CHAR) by a :class:`_BankedMemory`, so a tune reading KERNAL or BASIC
    sees the oracle's ROMs, not the RAM beneath. ``illegal_opcodes`` is kept for
    API compatibility; jennings decodes the NMOS illegals natively regardless.

    Raises:
      EmulatorUnavailable: jennings (a core dependency) is not installed.
    """
    try:
        from jennings.devices.mpu6502 import MPU
    except ImportError as exc:  # pragma: no cover - jennings is a core dependency
        raise EmulatorUnavailable(
            "jennings is required to run a tune: pip install pysidtracker"
        ) from exc

    mem = _BankedMemory(subject)
    holder: dict = {}
    chip = SidReads(sid_model)
    cpl = reg.NTSC_CYCLES_PER_LINE if ntsc else reg.PAL_CYCLES_PER_LINE
    lines = (reg.NTSC_CYCLES_PER_FRAME if ntsc else reg.PAL_CYCLES_PER_FRAME) // cpl

    def _on_raster(addr):
        line = (holder["mpu"].processorCycles // cpl) % lines
        if addr == reg.VIC_RASTER:
            return line & 0xFF
        return (subject[reg.VIC_CONTROL_1] & 0x7F) | (((line >> 8) & 1) << 7)

    def _on_sidwrite(addr, value):
        chip.write(access_cycle(holder["mpu"]), addr - reg.SID_BASE, value)

    def _on_sidread(addr):
        return chip.read(access_cycle(holder["mpu"]), addr - reg.SID_BASE)

    mem.subscribe_to_read([reg.VIC_CONTROL_1, reg.VIC_RASTER], _on_raster)
    mem.subscribe_to_write(_CHIP_INPUTS, _on_sidwrite)
    mem.subscribe_to_read(_CHIP_OUTPUTS, _on_sidread)
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


__all__ = ["SidReads", "access_cycle", "wire_mpu", "run_to_rts"]
