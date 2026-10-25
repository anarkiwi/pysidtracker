"""The SID half of the flat jennings host every init/replay path runs on.

The machine half (banked bus, raster reads, :func:`run_to_rts`) is
:mod:`vagrant_moses.cpu`; :func:`wire_mpu` adds a reSIDfp chip answering the SID's
readable registers, and :func:`observe_data_access` hooks the jennings engine.
"""

from __future__ import annotations

from array import array
from typing import Callable, Dict, Optional

from vagrant_moses import memmap as mm
from vagrant_moses.cpu import access_cycle, flat_mpu, run_to_rts

from .errors import EmulatorUnavailable

# osc3/env3 depend only on voice 3 and its sync/ring source, voice 2.
_CHIP_INPUTS = range(mm.SID_BASE + 0x07, mm.SID_BASE + 0x15)
_CHIP_OUTPUTS = (mm.SID_BASE + 0x1B, mm.SID_BASE + 0x1C)


class SidReads:
    """A reSIDfp chip answering osc3/env3 reads at the cycle the CPU makes them.

    Register writes are queued and the chip is clocked through them only when a
    read needs it, so a tune that never reads the chip costs nothing.
    """

    def __init__(self, model: str = "6581"):
        from pyresidfp._pyresidfp import SID, ChipModel, SamplingMethod

        chip = {"6581": ChipModel.MOS6581, "8580": ChipModel.MOS8580}[model]
        self._sid = SID(chip, SamplingMethod.DECIMATE, mm.PAL.clock_hz, 8000.0)
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


_WIRED_VM: Dict[type, type] = {}


def _observed_vm_class(engine: type) -> type:
    """A subclass whose pcode ``rd``/``wr`` also notify per-instance callbacks."""

    def read(self, address, size):
        value = engine._rd(self, address, size)
        if self._on_read is not None:
            for offset in range(size):
                self._on_read((address + offset) & 0xFFFF)
        return value

    def write(self, address, value, size):
        engine._wr(self, address, value, size)
        if self._on_write is not None:
            for offset in range(size):
                self._on_write(
                    (address + offset) & 0xFFFF, (value >> (8 * offset)) & 0xFF
                )

    return type(
        "ObservedPcodeVM",
        (engine,),
        {"_rd": read, "_wr": write, "__slots__": ("_on_read", "_on_write")},
    )


def observe_data_access(
    mpu,
    on_read: Optional[Callable[[int], None]] = None,
    on_write: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Route the engine's data ``LOAD``/``STORE`` ops (pcode ``rd``/``wr``) to callbacks.

    Instruction fetches and stack pushes never reach the callbacks.
    ``on_read(address)`` fires per byte read (value readable from the subject);
    ``on_write(address, value)`` per byte written, masked. No restoration needed.
    """
    if on_read is None and on_write is None:
        return
    engine = mpu._vm
    base = type(engine)
    for known_base, wired_cls in _WIRED_VM.items():
        if base is wired_cls or base in wired_cls.__mro__:
            base = known_base
    cls = _WIRED_VM.setdefault(base, _observed_vm_class(base))
    if isinstance(engine, cls):
        wired = engine
    else:
        wired = cls(b"")
        for attr in (
            "mem",
            "reg",
            "uniq",
            "cycles",
            "volatile",
            "vicirq",
            "ciaicr",
            "wlog",
        ):
            setattr(wired, attr, getattr(engine, attr))
        mpu._vm = wired
    wired._on_read = on_read
    wired._on_write = on_write


def wire_mpu(
    subject, illegal_opcodes: bool = True, sid_model: str = "6581", ntsc: bool = False
):  # pylint: disable=unused-argument
    """A jennings MPU over the C64 address-space buffer ``subject``; return ``(mpu, mem)``.

    The machine is :func:`vagrant_moses.cpu.flat_mpu`: ``$00``/``$01``-banked
    sidplayfp ROM-less ROMs and VIC raster reads from the cycle counter. The CPU
    port DDR is seeded to its reset value so ``$01`` banks. SID osc3/env3
    (``$D41B``/``$D41C``) reads come from a ``sid_model`` reSIDfp chip fed the CPU's
    own SID writes (:class:`SidReads`). ``illegal_opcodes`` is kept for API
    compatibility; jennings decodes the NMOS illegals natively regardless.

    Raises:
      EmulatorUnavailable: jennings (a core dependency) is not installed.
    """
    subject[mm.CPU_DDR] = mm.DDR_DEFAULT
    try:
        mpu, mem = flat_mpu(subject, ntsc=ntsc)
    except ImportError as exc:  # pragma: no cover - jennings is a core dependency
        raise EmulatorUnavailable(
            "jennings is required to run a tune: pip install pysidtracker"
        ) from exc
    chip = SidReads(sid_model)

    def _on_sidwrite(addr, value):
        chip.write(access_cycle(mpu), addr - mm.SID_BASE, value)

    def _on_sidread(addr):
        return chip.read(access_cycle(mpu), addr - mm.SID_BASE)

    mem.subscribe_to_write(_CHIP_INPUTS, _on_sidwrite)
    mem.subscribe_to_read(_CHIP_OUTPUTS, _on_sidread)
    return mpu, mem


def run_to_rts_until(mpu, mem, pc: int, acc: int, pcs, max_cycles: int) -> bool:
    """Call ``pc`` with ``A = acc`` like :func:`run_to_rts`, stopping on entering ``pcs``.

    ``pcs`` is a collection of addresses; the check is made before each
    instruction, so the stop is *at* the instruction about to run and a ``pc``
    already in ``pcs`` stops at once. True if one was entered, False if the
    routine returned or ``max_cycles`` ran out first.
    """
    start_sp = mpu.sp
    for byte in (0x00, 0x01):
        mem[0x0100 + mpu.sp] = byte
        mpu.sp = (mpu.sp - 1) & 0xFF
    mpu.a, mpu.pc = acc & 0xFF, pc
    start = mpu.processorCycles
    while mpu.sp < start_sp and mpu.processorCycles - start <= max_cycles:
        if mpu.pc in pcs:
            return True
        mpu.step()
    return False


def set_iomap(mpu, value: int) -> None:
    """Write the C64 ``$01`` bank port on ``mpu``'s bus, remapping the ROM/RAM windows."""
    mpu.memory[1] = value


__all__ = [
    "SidReads",
    "observe_data_access",
    "wire_mpu",
    "run_to_rts",
    "run_to_rts_until",
    "set_iomap",
]
