"""The SID half of the flat jennings host every init/replay path runs on.

The machine half (banked bus, raster reads, :func:`run_to_rts`) is
:mod:`vagrant_moses.cpu`; :func:`wire_mpu` adds a reSIDfp chip answering the SID's
readable registers, and :func:`observe_data_access` hooks the jennings engine.
"""

from __future__ import annotations

import weakref
from array import array
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from vagrant_moses import memmap as mm
from vagrant_moses.cpu import access_cycle, flat_mpu, run_to_rts

from .errors import EmulatorUnavailable, SidError
from .registers import SID_REG_COUNT, is_sid_reg

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
    reads=None,
    writes=None,
) -> None:
    """Route the engine's data loads and stores to callbacks.

    Instruction fetches and interrupt/subroutine stack traffic never reach the
    callbacks. ``on_read(address)`` fires per byte read (value readable from the
    subject); ``on_write(address, value)`` per byte written, masked. ``reads`` and
    ``writes`` optionally limit the callbacks to those address sets. A native MPU
    uses the core's own observer with ``stack=True``, so ``PHA``/``PLA``/``PHP``/``PLP``
    page-1 accesses are reported as the Python engine's P-Code hook reports them.
    """
    if on_read is None and on_write is None:
        return
    if getattr(mpu, "native", False):
        mpu.memory.observe_data_access(
            on_read, on_write, reads=reads, writes=writes, stack=True
        )
        return
    if reads is not None and on_read is not None:
        reads, read_cb = frozenset(reads), on_read

        def on_read(addr):  # pylint: disable=function-redefined
            if addr in reads:
                read_cb(addr)

    if writes is not None and on_write is not None:
        writes, write_cb = frozenset(writes), on_write

        def on_write(addr, value):  # pylint: disable=function-redefined
            if addr in writes:
                write_cb(addr, value)

    engine = mpu._vm  # pylint: disable=protected-access
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
        mpu._vm = wired  # pylint: disable=protected-access
    wired._on_read = on_read  # pylint: disable=protected-access
    wired._on_write = on_write  # pylint: disable=protected-access


_SUBJECTS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def sync_ram(mpu) -> None:
    """Copy a native MPU's RAM back into the buffer it was wired over.

    The native core copies the buffer in and owns the RAM from then on; code that
    reads the buffer after a run calls this first. A no-op on the Python engine,
    which runs on the buffer itself.
    """
    subject = _SUBJECTS.get(mpu)
    if subject is not None:
        subject[:] = memoryview(mpu.memory)


def unobserve_data_access(mpu) -> None:
    """Remove the callbacks :func:`observe_data_access` installed."""
    if getattr(mpu, "native", False):
        mpu.memory.observe_data_access(None, None)
    elif hasattr(mpu._vm, "_on_read"):  # pylint: disable=protected-access
        mpu._vm._on_read = mpu._vm._on_write = None  # pylint: disable=protected-access


def step_to_rts(mpu, mem, pc: int, acc: int, max_cycles: int) -> None:
    """:func:`run_to_rts` stepped from Python, so a ``mpu.step`` hook sees every instruction."""
    start_sp = mpu.sp
    for byte in (0x00, 0x01):
        mem[0x0100 + mpu.sp] = byte
        mpu.sp = (mpu.sp - 1) & 0xFF
    mpu.a, mpu.pc = acc & 0xFF, pc
    start = mpu.processorCycles
    while mpu.sp < start_sp and mpu.processorCycles - start <= max_cycles:
        mpu.step()


def run_until(mpu, stop_pcs=(), sp_above=None, max_cycles=None) -> str:
    """Step ``mpu`` until a stop condition holds before an instruction; say which.

    ``"sp"``: SP exceeds ``sp_above``; ``"cycles"``: more than ``max_cycles`` have
    elapsed since the call; ``"pc"``: PC is in ``stop_pcs``; ``"jam"``: a JAM ran (it skips a byte, as ``step`` does).
    A native MPU runs this in its own driver, otherwise the same checks are stepped.
    """
    if getattr(mpu, "native", False):
        return mpu.run_until(
            max_cycles=max_cycles, stop_pcs=stop_pcs, sp_above=sp_above
        )
    start = mpu.processorCycles
    while True:
        if sp_above is not None and mpu.sp > sp_above:
            return "sp"
        if max_cycles is not None and mpu.processorCycles - start > max_cycles:
            return "cycles"
        if mpu.pc in stop_pcs:
            return "pc"
        mpu.step()


def trace_pcs(mpu, mem, pc: int, acc: int, max_cycles: int, sink: list) -> None:
    """:func:`run_to_rts`, appending the PC of each executed instruction to ``sink``.

    A native MPU records them in its driver and extends ``sink`` once at the end;
    otherwise the PC is appended as each instruction is about to run.
    """
    if getattr(mpu, "native", False):
        sink.extend(np.asarray(mpu.run_to_rts_traced(pc, acc, max_cycles).pc).tolist())
        return
    start_sp = mpu.sp
    for byte in (0x00, 0x01):
        mem[0x0100 + mpu.sp] = byte
        mpu.sp = (mpu.sp - 1) & 0xFF
    mpu.a, mpu.pc = acc & 0xFF, pc
    start = mpu.processorCycles
    while mpu.sp < start_sp and mpu.processorCycles - start <= max_cycles:
        sink.append(mpu.pc)
        mpu.step()


def live_ram(mpu, subject):
    """The buffer ``mpu`` runs on: its own RAM when native, else ``subject``."""
    return memoryview(mpu.memory) if getattr(mpu, "native", False) else subject


def wire_mpu(
    subject,
    illegal_opcodes: bool = True,
    sid_model: str = "6581",
    ntsc: bool = False,
    native: Optional[bool] = None,
):  # pylint: disable=unused-argument
    """A jennings MPU over the C64 address-space buffer ``subject``; return ``(mpu, mem)``.

    The machine is :func:`vagrant_moses.cpu.flat_mpu`: ``$00``/``$01``-banked
    sidplayfp ROM-less ROMs and VIC raster reads from the cycle counter. The CPU
    port DDR is seeded to its reset value so ``$01`` banks. SID osc3/env3
    (``$D41B``/``$D41C``) reads come from a ``sid_model`` reSIDfp chip fed the CPU's
    own SID writes (:class:`SidReads`). ``illegal_opcodes`` is kept for API
    compatibility; jennings decodes the NMOS illegals natively regardless.
    ``native`` picks the engine as in :func:`vagrant_moses.cpu.flat_mpu` (default:
    jennings' native core when installed); a native MPU owns a copy of ``subject``,
    which :func:`sync_ram` refreshes.

    Raises:
      EmulatorUnavailable: jennings (a core dependency) is not installed.
    """
    subject[mm.CPU_DDR] = mm.DDR_DEFAULT
    try:
        mpu, mem = flat_mpu(subject, ntsc=ntsc, native=native)
    except ImportError as exc:  # pragma: no cover - jennings is a core dependency
        raise EmulatorUnavailable(
            "jennings is required to run a tune: pip install pysidtracker"
        ) from exc
    if mpu.native:
        _SUBJECTS[mpu] = subject
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
    if getattr(mpu, "native", False):
        return mpu.run_to_rts_until(pc, acc, pcs, max_cycles)
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


PLAY_CYCLES = 200_000

SidWrite = Tuple[int, Optional[int], int, int]


class SidWriteLog:
    """Each play call's SID writes, ``(register, address read last, pc, value)``.

    Attaches to ``mpu`` with :func:`observe_data_access`; ``mirrors`` also logs
    the ``$20``-mirrored band, else only ``$D400-$D418``. :meth:`start` opens a call.
    On a native MPU the address read last is the core's ``last_load``.
    """

    def __init__(self, mpu, mirrors: bool = True):
        top = mm.SID_BASE + (0x400 if mirrors else SID_REG_COUNT)
        self.registers = {
            addr: (addr - mm.SID_BASE) & 0x1F
            for addr in range(mm.SID_BASE, top)
            if is_sid_reg(addr)
        }
        self.calls: List[List[SidWrite]] = []
        self.mpu, self._last = mpu, None
        if getattr(mpu, "native", False):
            observe_data_access(mpu, on_write=self._write, writes=self.registers)
        else:
            observe_data_access(
                mpu, on_read=self._read, on_write=self._write, writes=self.registers
            )

    def _read(self, addr: int) -> None:
        self._last = addr

    def _write(self, addr: int, value: int) -> None:
        if self.calls:
            if getattr(self.mpu, "native", False):
                self._last = self.mpu.memory.last_load
            self.calls[-1].append(
                (self.registers[addr], self._last, self.mpu.pc, value)
            )

    def start(self) -> List[SidWrite]:
        """Open a new call's write list and return it."""
        self.calls.append([])
        return self.calls[-1]

    def call(self, mem, play: int, max_cycles: int = PLAY_CYCLES, error=SidError):
        """One :func:`call_play` of ``play`` logged as a new call; its writes."""
        writes = self.start()
        call_play(self.mpu, mem, play, max_cycles, error)
        return writes


def call_play(
    mpu, mem, play: int, max_cycles: int = PLAY_CYCLES, error: type = SidError
) -> None:
    """Run one call of ``play`` (``A = 0``); raise ``error`` if it does not return."""
    depth = mpu.sp
    run_to_rts(mpu, mem, play, 0, max_cycles)
    if mpu.sp < depth:
        raise error("play call did not return")


def set_iomap(mpu, value: int) -> None:
    """Write the C64 ``$01`` bank port on ``mpu``'s bus, remapping the ROM/RAM windows."""
    mpu.memory[1] = value


__all__ = [
    "PLAY_CYCLES",
    "SidReads",
    "SidWriteLog",
    "call_play",
    "observe_data_access",
    "wire_mpu",
    "run_to_rts",
    "run_to_rts_until",
    "run_until",
    "live_ram",
    "set_iomap",
    "step_to_rts",
    "trace_pcs",
    "sync_ram",
    "unobserve_data_access",
]
