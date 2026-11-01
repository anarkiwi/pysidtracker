"""Run a playspec on the emulator memory a tune's own init leaves.

:class:`SpecPlayer` boots like :class:`~pysidtracker.oracle.EmuPlayer`; a play
call runs the spec directly at the modelled entry, else the 6502 play routine runs
with every call into a modelled entry trapped and run as the spec. A tune with play
address 0 boots on an :class:`~pysidtracker.irqhost.InterruptPlayer`, its interrupts
live, with each call its handlers make into a modelled entry run as the spec. Once
the interrupts have been taken ``MIN_RAISES`` times (the raises ``play_source`` needs
to call a source the play source) without any call reaching a modelled entry, the tune
is playing only its own code and ``SidParseError`` is raised.
"""

from __future__ import annotations

import struct
from collections import Counter
from typing import Callable, Collection, Dict, List, Optional, Tuple, Union

import numpy as np
from jennings.devices.mpu6502 import MPU
from vagrant_moses import memmap as mm
from vagrant_moses.bus import banking

from ..emu import live_ram, run_to_rts, run_until, set_iomap, wire_mpu
from ..errors import SidParseError
from ..image import SidImage
from ..irqhost import InterruptPlayer
from ..oracle import ENTRY_SOURCES, PAL_FLAG, EmuPlayer
from ..poweron import psid_init_bank, psid_play_bank
from . import native as native_module
from .ir import Program
from .lower import elaborate
from .pyexec import compile_program, controls
from .syntax import Spec
from .validate import StreamValidator

MAX_CYCLES = 8_000_000
BAILS = 8
BAIL_FRAMES = 20
MIN_RAISES = 3
_CPU_FIELDS = ("pc", "sp", "a", "x", "y", "p", "processorCycles")
_FLAGS = {"c": 0x01, "z": 0x02, "i": 0x04, "d": 0x08, "v": 0x40, "n": 0x80}
Binder = Callable[[bytes, int], Union[dict, Tuple[Spec, dict]]]
_HEADER = 0x7C
SID_BASE, SID_REGS = 0xD400, 25


def _container(image: bytes, load: int, init: int, psid: bool) -> bytes:
    """A play-address-0 PSID (RSID if not ``psid``) container around ``image``."""
    head = bytearray(_HEADER)
    head[0:4] = b"PSID" if psid else b"RSID"
    struct.pack_into(">HHHHHHHI", head, 4, 2, _HEADER, load, init, 0, 1, 1, 0)
    return bytes(head) + bytes(image)


class NativeEscape(Exception):
    """A ``native`` block left the spec: the CPU ran the rest of the call."""


class _HostMemory:
    """The machine's reads, with writes going where the spec's stores go."""

    def __init__(self, machine, store: Callable[[int, int], None]):
        self._machine, self._store = machine, store

    def __getitem__(self, addr: int) -> int:
        return self._machine[addr]

    def __setitem__(self, addr: int, value: int) -> None:
        self._store(addr, value)


class SpecPlayer(EmuPlayer):
    """A :class:`~pysidtracker.oracle.EmuPlayer` whose play calls run ``spec``.

    ``bind(view, entry)`` is the layout binding of the instance at ``entry`` in the
    CPU's memory ``view``, or ``(variant, binding)`` to run a variant of ``spec``
    (one that ``extends`` it, say) for that instance; ``entries(view)`` lists the modelled entries (default: the
    play address). ``validate`` checks opened streams against their grammars.

    With ``play`` 0 the tune runs its own interrupts on an
    :class:`~pysidtracker.irqhost.InterruptPlayer`: ``sid`` is its PSID/RSID (default: a
    header built around ``image``), ``clock`` its machine's clock. A call a handler
    makes into a modelled entry runs as the spec, as an instant at the cycle it is
    reached; ``render_grid`` frames at the play source's raises. ``entry`` is the last
    entry trapped.

    If the first play call reaches no modelled entry (a wrapper that alternates
    entries or skips the player), ``entry`` starts as the lowest of ``entries(view)``;
    :class:`~pysidtracker.SidParseError` is raised only when that set is empty. Frames
    are unaffected: each call dispatches on the entry the CPU reaches.

    With ``notes`` (a clock in Hz) every spec read of a cell declared ``notes(...)``
    is the standard note word of that note at the clock, not memory: the register
    grid is then the tune's as played from the standard equal-tempered table. The
    tune's own 6502 code and ``native`` blocks still read their native table.

    With ``inputs`` (``inputs(address) -> byte``) every spec read of a ``volatile``
    cell takes its value from it instead of the bus, so a run can be fed the values
    the 6502 read in the same call (:attr:`~pysidtracker.oracle.EmuPlayer.reads`).

    ``transform(program, view)`` maps each lowered program before it is compiled, given
    the memory it is bound in (e.g. :func:`~pysidtracker.playspec.outorder.reorder`).

    ``writes`` lists the SID writes ``(register, value)`` the spec's stores made in the last
    frame, in order, those of every play call of the frame and every write of a register (a
    write of the value it holds included); a write by a ``native`` block or the tune's own code
    is not among them.

    A spec call advances no cycles, so with ``cycles_per_frame`` scheduling the
    player plays a call the real routine would have overrun and had dropped, and runs
    one call ahead from there.
    """

    def __init__(
        self,
        spec: Spec,
        image: bytes,
        load: int,
        init: int,
        play: int,
        subtune: int = 0,
        *,
        bind: Binder,
        entries: Optional[Callable[[bytes], Collection[int]]] = None,
        psid: bool = True,
        validate: bool = False,
        sid: Union[bytes, SidImage, None] = None,
        clock: Optional[str] = None,
        notes: Optional[int] = None,
        inputs: Optional[Callable[[int], int]] = None,
        transform: Optional[Callable[[Program, bytes], Program]] = None,
        native: Optional[bool] = None,
        **kwargs,
    ):
        # pylint: disable=too-many-arguments
        self.spec = spec
        self._bind = bind
        self._entries_of = entries or (lambda view: {play})
        self._psid = psid
        self._bus = None
        self._compiled: Dict[Tuple[int, int], tuple] = {}
        self._entry_sets: Dict[int, frozenset] = {}
        self.validator = StreamValidator() if validate else None
        self._native_state: Optional[dict] = None
        self._in_call = False
        self.writes: List[Tuple[int, int]] = []
        self.entry: Optional[int] = None
        self._hosted: Optional[InterruptPlayer] = None
        self._stepper = None
        self._sid, self._clock, self._raw = sid, clock, bytes(image)
        self._standard = notes
        self._inputs = inputs
        self._transform = transform
        self._fast = native_module.available() if native is None else native
        self._core = None
        self._registered: set = set()
        self._bails = 0
        self._played = 0
        kwargs.setdefault("max_cycles", MAX_CYCLES)
        super().__init__(image, load, init, play, subtune, **kwargs)

    @property
    def memory(self) -> bytes:
        """The flat 64 KiB RAM the player runs in."""
        return bytes(self._bus.ram if self._hosted else self._mem)

    def view(self) -> bytes:
        """The 64 KiB as the CPU reads it under the current ``$01`` banking."""
        bus = self._bus
        out = bytearray(bus.ram)
        for page_id, page in enumerate(bus.pages):
            if page is not None and page is not bus.ram:
                span = slice(page_id << 12, (page_id + 1) << 12)
                out[span] = page[span]
        return bytes(out)

    def _entries(self) -> frozenset:
        bits = self._bus.bits
        if bits not in self._entry_sets:
            self._entry_sets[bits] = frozenset(self._entries_of(self.view()))
        return self._entry_sets[bits]

    def program(self, entry: Optional[int] = None) -> Program:
        """The lowered program of the instance at ``entry`` (default: the play entry)."""
        return self._instance(self.entry if entry is None else entry)[0]

    def _instance(self, entry: int) -> tuple:
        key = (self._bus.bits, entry)
        if key not in self._compiled:
            bound = self._bind(self.view(), entry)
            spec, binding = bound if isinstance(bound, tuple) else (self.spec, bound)
            program = elaborate(spec, binding, standard=self._standard)
            if self._transform is not None:
                program = self._transform(program, self.view())
            self._compiled[key] = (program, compile_program(program), controls(program))
        return self._compiled[key]

    def _init(self, subtune: int) -> None:
        if not self._play_addr:
            self._host(subtune)
            return
        if self._pal_flag is not None:
            self._mem[PAL_FLAG] = self._pal_flag
        self._mem[1] = psid_init_bank(self._init_addr) if self._psid else 0x37
        self._mpu, self._obs = wire_mpu(self._mem, self._illegal, self._sid_model)
        self._mem = live_ram(self._mpu, self._mem)
        run_to_rts(self._mpu, self._obs, self._init_addr, subtune, self._max_cycles)
        if self._psid:
            set_iomap(self._mpu, psid_play_bank(self._play_addr))
        self._due = self._mpu.processorCycles
        self._bus = self._obs._subject  # pylint: disable=protected-access
        if self._mpu.native:  # a bank change moves the entry set: hand back to Python
            self._obs.subscribe_to_write(
                (0, 1), lambda *_: setattr(self._mpu, "stop", True)
            )
        play = self._play_addr
        entries = self._entries()
        self.entry = play if play in entries else self._probe(play)
        if self.entry is None:
            if not entries:
                raise SidParseError(
                    f"the play routine at ${play:04x} reaches no modelled entry"
                )
            self.entry = min(entries)  # the probed call skipped the player

    def _host(self, subtune: int) -> None:
        """Boot the tune's interrupts on a host that traps modelled entries."""
        data = self._sid
        if data is None:
            data = _container(self._raw, self._load, self._init_addr, self._psid)
        self._hosted = InterruptPlayer(
            data, subtune, clock=self._clock, trap=self._trap
        )
        machine = self._hosted.host.machine
        self._mpu = machine.mpu
        self._obs = self._bus = machine
        self._stepper = MPU(memory=_HostMemory(machine, self._store))

    def _trap(self, machine) -> bool:
        """Run the call at ``pc`` as the spec if it is a modelled entry; True if run."""
        entries = self._entries()
        machine.watch = entries
        mpu = machine.mpu
        if mpu.pc not in entries:
            return False
        self.entry = mpu.pc
        self._in_call = True
        if not self._run(self.entry):  # spec ran: return to the caller as RTS does
            mpu.sp = (mpu.sp + 2) & 0xFF
            ram = machine.ram
            mpu.pc = ((ram[0x00FF + mpu.sp] | ram[0x0100 + mpu.sp] << 8) + 1) & 0xFFFF
        return True

    def _probe(self, play: int) -> Optional[int]:
        """The first modelled entry the play routine calls, leaving no trace."""
        mpu = self._mpu
        mem = bytes(self._mem)
        regs = (mpu.pc, mpu.sp, mpu.a, mpu.x, mpu.y, mpu.p, mpu.processorCycles)
        entry = self._call(play, probe=True)
        self._mem[:] = mem
        self._bus[1] = mem[1]
        mpu.pc, mpu.sp, mpu.a, mpu.x, mpu.y, mpu.p, mpu.processorCycles = regs
        return entry

    def _native(self, addr: int, regs: dict, resume: int) -> None:
        """Run the tune's 6502 code from ``addr`` until it reaches ``resume``; if
        the call returns first, the CPU has finished it and the spec is over."""
        regfile = self._mpu
        mpu = self._stepper or regfile
        if mpu is not regfile:
            for field in _CPU_FIELDS:
                setattr(mpu, field, getattr(regfile, field))
        try:
            self._native_on(mpu, addr, regs, resume)
        finally:
            if mpu is not regfile:
                for field in _CPU_FIELDS:
                    setattr(regfile, field, getattr(mpu, field))

    def _native_on(self, mpu, addr: int, regs: dict, resume: int) -> None:
        mine = self._native_state
        if mine["saved"] is None:
            mine["saved"] = (mpu.pc, mpu.sp, mpu.a, mpu.x, mpu.y, mpu.p)
            if mine["base"] is None:  # the spec runs at the play call: give it a caller
                for byte in (0x00, 0x01):
                    self._obs[0x0100 + mpu.sp] = byte
                    mpu.sp = (mpu.sp - 1) & 0xFF
                mine["base"] = mpu.sp
        for reg, value in regs.items():
            setattr(mpu, reg, value)
        mpu.pc = addr
        start = mpu.processorCycles
        while mpu.pc != resume:
            if mpu.sp > mine["base"]:
                raise NativeEscape()
            if mpu.processorCycles - start >= self._max_cycles:
                raise SidParseError(
                    f"native code at ${addr:04x} never reaches ${resume:04x}"
                )
            run_until(
                mpu,
                (resume,),
                mine["base"],
                self._max_cycles - (mpu.processorCycles - start) - 1,
            )

    def _run(self, entry: int) -> bool:
        """Run the spec at ``entry``; True if native code left it and the CPU finished the call."""
        mpu = self._mpu
        self._native_state = {"base": mpu.sp if self._in_call else None, "saved": None}
        try:
            self._frame_of(entry)
        except NativeEscape:
            saved = self._native_state["saved"]
            if not self._in_call:
                mpu.pc, mpu.sp, mpu.a, mpu.x, mpu.y, mpu.p = saved
            return True
        saved = self._native_state["saved"]
        if saved is not None:  # resumed: the spec models no register outputs
            mpu.pc, mpu.sp, mpu.a, mpu.x, mpu.y, mpu.p = saved
        return False

    def _frame_of(self, entry: int) -> None:
        program, frame, state = self._instance(entry)
        hook = None
        if self.validator is not None:
            hook = self.validator.hook(program, self._bus.__getitem__)
        mpu = self._mpu
        regs = {"a": mpu.a, "x": mpu.x, "y": mpu.y}
        regs.update((f, bool(mpu.p & bit)) for f, bit in _FLAGS.items())
        read = self._bus.__getitem__
        volatile = {
            a for c in program.cells if "volatile" in c.attrs for a in c.extent()
        }
        if self._inputs is not None and volatile:
            bus_read = read

            def read(addr):
                return self._inputs(addr) if addr in volatile else bus_read(addr)

        frame(read, self._store, state, hook, regs, self._native)

    def _store(self, addr: int, value: int) -> None:
        """A spec store: RAM (the SID band included), and the bank port on the bus."""
        if SID_BASE <= addr < SID_BASE + SID_REGS:
            self.writes.append((addr - SID_BASE, value & 0xFF))
        if self._hosted is not None:
            bus = self._bus
            if bus.io and addr >> 12 == 0xD:
                bus.io_write(addr, value)
            else:
                bus[addr] = value
            return
        self._mem[addr] = value
        if addr <= 1:
            self._bus[addr] = value

    def snapshot(self) -> List[int]:
        if self._hosted is None:
            return super().snapshot()
        return self._hosted.snapshot()

    def _require_entry(self) -> None:
        """Raise if the hosted tune's interrupts have never reached a modelled entry."""
        if self.entry is not None:
            return
        taken = Counter(
            r.value
            for r in self._hosted.host.rows()
            if r.chip is None and r.value in ENTRY_SOURCES
        )
        if max(taken.values(), default=0) >= MIN_RAISES:
            raise SidParseError(
                "the interrupts reach no modelled entry: the tune plays only its own code"
            )

    def _frame(self) -> None:
        if self._hosted is None and self._advance(1):
            return
        self._frame_python()

    def _frame_python(self) -> None:
        self.writes = []
        if self._hosted is not None:
            self._hosted.play_frame()
            self._require_entry()
        elif self._play_addr == self.entry:
            self._in_call = False
            self._run(self.entry)
        else:
            self._call(self._play_addr)

    def _bank_pages(self, bits: int) -> tuple:
        """``(io, ROM pages)`` of the memory map under CPU port setting ``bits``."""
        basic, kernal, chars, io = banking(bits)
        roms = self._bus.roms
        window = []
        if basic:
            window += [(0xA, mm.BASIC_BASE), (0xB, mm.BASIC_BASE)]
        if chars and not io:
            window.append((0xD, mm.CHARGEN_BASE))
        if kernal:
            window += [(0xE, mm.KERNAL_BASE), (0xF, mm.KERNAL_BASE)]
        return io, [
            (n, bytes(roms[base][n << 12 : (n + 1) << 12])) for n, base in window
        ]

    def _machine(self):
        """The native core with every memory map and program built so far, mapped as the bus
        now is, or None where a program is not one it runs."""
        bus = self._bus
        self._entries()
        if self._core is None:
            self._core = native_module.Core(
                self._mem, self._play_addr, self.entry, self._max_cycles
            )
            self._registered = set()
        core = self._core
        try:
            for bits, entries in list(self._entry_sets.items()):
                if bits not in self._registered:
                    core.add_bank(bits, *self._bank_pages(bits), sorted(entries))
                    self._registered.add(bits)
            for key, (program, _frame, state) in list(self._compiled.items()):
                if key not in self._registered and key[0] in self._registered:
                    core.add(key[0], key[1], *native_module.encode(program))
                    if state:
                        core.set_controls(key[0], key[1], state)
                    self._registered.add(key)
        except native_module.Unsupported:
            self._fast, self._core = False, None
            return None
        core.select(bus.bits)
        return core

    def _remap(self) -> None:
        """Bring the bus to the CPU port the native core left in RAM."""
        bus = self._bus
        bus.port_dir, bus.port_data = self._mem[0], self._mem[1]
        bus.remap()

    def _native_ok(self) -> bool:
        """Whether this player's frames may run on the native core."""
        return (
            self._fast
            and self._hosted is None
            and self.validator is None
            and self._inputs is None
            and not self._record
            and self.entry is not None
            and getattr(self._mpu, "native", False)
        )

    def _advance(self, frames: int, out: Optional[bytearray] = None) -> bool:
        """Play ``frames`` frames on the native core, each frame's 25 registers going to ``out``
        (a frame Python must play is played by it); False if the core cannot play them.
        """
        if not self._native_ok():
            return False
        done = 0
        mpu = self._mpu
        while done < frames:
            core = self._machine()
            if core is None:
                return self._fallback(frames, done, out)
            core.set_cpu(
                (mpu.pc, mpu.sp, mpu.a, mpu.x, mpu.y, mpu.p, mpu.processorCycles)
            )
            for (bits, entry), (_p, _f, state) in self._compiled.items():
                if state:
                    core.set_controls(bits, entry, state)
            played, status, _why = core.run(
                frames - done,
                out if out is not None else bytearray(25 * (frames - done)),
                done if out is not None else 0,
            )
            mpu.pc, mpu.sp, mpu.a, mpu.x, mpu.y, mpu.p, mpu.processorCycles = core.cpu()
            for (bits, entry), (_p, _f, state) in self._compiled.items():
                if state:
                    state[:] = core.controls(bits, entry)
            self._remap()
            done += played
            self._played += played
            if status == 0:
                if played:
                    self.writes = core.writes()
                break
            if status == 2:
                self._bails += 1
                self._fast = (
                    self._bails <= BAILS or self._bails * BAIL_FRAMES < self._played
                )
            self._frame_python()
            if out is not None:
                out[done * 25 : (done + 1) * 25] = bytes(self.snapshot())
            done += 1
            if not self._fast:
                return self._fallback(frames, done, out)
        return True

    def _fallback(self, frames: int, done: int, out: Optional[bytearray]) -> bool:
        """Play the frames from ``done`` on in Python."""
        for index in range(done, frames):
            self._frame_python()
            if out is not None:
                out[index * 25 : (index + 1) * 25] = bytes(self.snapshot())
        return True

    def render_grid(self, nframes: int) -> List[List[int]]:
        if self._hosted is not None:
            grid = self._hosted.render_grid(nframes)
            self._require_entry()
            return grid
        if not nframes or not self._native_ok():
            return super().render_grid(nframes)
        out = bytearray(25 * nframes)
        self._advance(nframes, out)
        rows = np.frombuffer(out, dtype=np.uint8).reshape(nframes, 25).tolist()
        self.regs, self._last_regs = list(rows[-1]), list(rows[-1])
        return rows

    def _call(self, play: int, probe: bool = False) -> Optional[int]:
        """Run the play routine on the CPU, running the spec for each modelled call."""
        mpu, bus = self._mpu, self._obs
        top = mpu.sp
        for byte in (0x00, 0x01):
            bus[0x0100 + mpu.sp] = byte
            mpu.sp = (mpu.sp - 1) & 0xFF
        mpu.pc = play
        start = mpu.processorCycles
        while mpu.sp < top and mpu.processorCycles - start < self._max_cycles:
            entries = self._entries()
            if mpu.pc not in entries:
                run_until(
                    mpu,
                    entries,
                    top - 1,
                    self._max_cycles - (mpu.processorCycles - start) - 1,
                )
                continue
            if probe:
                return mpu.pc
            self._in_call = True
            if self._run(mpu.pc):
                continue
            mpu.sp = (mpu.sp + 2) & 0xFF
            ret = bus[0x00FF + mpu.sp] | (bus[0x0100 + mpu.sp] << 8)
            mpu.pc = (ret + 1) & 0xFFFF
        return None


__all__ = ["Binder", "SpecPlayer"]
