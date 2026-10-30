"""Run a playspec on the emulator memory a tune's own init leaves.

:class:`SpecPlayer` boots like :class:`~pysidtracker.oracle.EmuPlayer`; a play
call runs the spec directly at the modelled entry, else the 6502 play routine runs
with every call into a modelled entry trapped and run as the spec. A tune with play
address 0 boots on an :class:`~pysidtracker.irqhost.InterruptPlayer`, its interrupts
live, with each call its handlers make into a modelled entry run as the spec.
"""

from __future__ import annotations

import struct
from typing import Callable, Collection, Dict, List, Optional, Tuple, Union

from ..emu import run_to_rts, set_iomap, wire_mpu
from ..errors import SidParseError
from ..image import SidImage
from ..irqhost import InterruptPlayer
from ..oracle import PAL_FLAG, EmuPlayer
from ..poweron import psid_init_bank, psid_play_bank
from .ir import Program
from .lower import elaborate
from .pyexec import compile_program, controls
from .syntax import Spec
from .validate import StreamValidator

MAX_CYCLES = 8_000_000
_FLAGS = {"c": 0x01, "z": 0x02, "i": 0x04, "d": 0x08, "v": 0x40, "n": 0x80}
Binder = Callable[[bytes, int], dict]
_HEADER = 0x7C


def _container(image: bytes, load: int, init: int, psid: bool) -> bytes:
    """A play-address-0 PSID (RSID if not ``psid``) container around ``image``."""
    head = bytearray(_HEADER)
    head[0:4] = b"PSID" if psid else b"RSID"
    struct.pack_into(">HHHHHHHI", head, 4, 2, _HEADER, load, init, 0, 1, 1, 0)
    return bytes(head) + bytes(image)


class NativeEscape(Exception):
    """A ``native`` block left the spec: the CPU ran the rest of the call."""


class SpecPlayer(EmuPlayer):
    """A :class:`~pysidtracker.oracle.EmuPlayer` whose play calls run ``spec``.

    ``bind(view, entry)`` is the layout binding of the instance at ``entry`` in the
    CPU's memory ``view``; ``entries(view)`` lists the modelled entries (default: the
    play address). ``validate`` checks opened streams against their grammars.

    With ``play`` 0 the tune runs its own interrupts on an
    :class:`~pysidtracker.irqhost.InterruptPlayer`: ``sid`` is its PSID/RSID (default: a
    header built around ``image``), ``clock`` its machine's clock. A call a handler
    makes into a modelled entry runs as the spec, as an instant at the cycle it is
    reached; ``render_grid`` frames at the play source's raises. ``entry`` is the last
    entry trapped.
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
        self.entry: Optional[int] = None
        self._hosted: Optional[InterruptPlayer] = None
        self._sid, self._clock, self._raw = sid, clock, bytes(image)
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
            program = elaborate(self.spec, self._bind(self.view(), entry))
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
        run_to_rts(self._mpu, self._obs, self._init_addr, subtune, self._max_cycles)
        if self._psid:
            set_iomap(self._mpu, psid_play_bank(self._play_addr))
        self._due = self._mpu.processorCycles
        self._bus = self._obs._subject  # pylint: disable=protected-access
        play = self._play_addr
        self.entry = play if play in self._entries() else self._probe(play)
        if self.entry is None:
            raise SidParseError(
                f"the play routine at ${play:04x} reaches no modelled entry"
            )

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
        mpu = self._mpu
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
            mpu.step()

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
        frame(self._bus.__getitem__, self._store, state, hook, regs, self._native)

    def _store(self, addr: int, value: int) -> None:
        """A spec store: RAM (the SID band included), and the bank port on the bus."""
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

    def render_grid(self, nframes: int) -> List[List[int]]:
        if self._hosted is None:
            return super().render_grid(nframes)
        return self._hosted.render_grid(nframes)

    def _frame(self) -> None:
        if self._hosted is not None:
            self._hosted.play_frame()
        elif self._play_addr == self.entry:
            self._in_call = False
            self._run(self.entry)
        else:
            self._call(self._play_addr)

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
            if mpu.pc not in self._entries():
                mpu.step()
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
