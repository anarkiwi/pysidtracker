"""Run a tune's real interrupt structure, cycle for cycle, as the sidtrace oracle does.

:class:`InterruptHost` boots sidplayfp's PSID driver and a PSID or RSID on
:class:`vagrant_moses.machine.Machine`, the cycle-exact C64, with reSIDfp chips
as its SID devices and every SID write and interrupt event logged as sidtrace
logs it; see ``docs/irqhost.md`` for what is modelled and how it is validated.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Type, Union

from vagrant_moses import memmap as mm
from vagrant_moses import machine as vagrant_machine
from vagrant_moses.machine import SOURCES, Machine, boot_cycle
from vagrant_moses.roms import KERNAL_VECTORS

from . import registers as reg
from .asm6502 import assemble
from .cadence import Cadence, TriggerSource, _driver_base, _iomap, playroutine_cadence
from .emu import SidReads
from .errors import SidParseError
from .image import SidImage
from .oracle import (
    EmuPlayer,
    SidtraceRow,
    _raise_frames,
    nested,
    play_source,
    raise_cycles,
    sidtrace_cadence,
    sidtrace_grid,
)
from .player import MemPlayer
from .reglog import RegWrite, register_writes_from_player
from .poweron import PSID_DRIVER_CLEARED, place_tune
from .roms import poweron_writes

_IRQ_I = 0x04

_DRIVER = """
playnum: .byte 0
speed: .byte 0
initvec: .word 0
playvec: .word 0
initiomap: .byte 0
playiomap: .byte 0
video: .byte 0
clock: .byte 0
flags: .byte 0
play: JMP (playvec)
init: JMP (initvec)
cold: SEI
  LDA #$00
  STA $D01A
  LDA $D019
  STA $D019
  LDA #$7F
  STA $DC0D
  STA $DD0D
  LDA $DC0D
  LDA $DD0D
  LDA #$0F
  STA $D418
  LDA video
  BEQ ntsc
  LDX #$25
  LDY #$40
  JMP timer
ntsc: LDX #$95
  LDY #$42
timer: STX $DC04
  STY $DC05
  LDX #$9B
  LDY #$37
  EOR clock
  ORA initiomap
  BEQ vicinit
  LDX #$1B
  LDY #$00
vicinit: STX $D011
  STY $D012
  LDA initiomap
  BEQ irqinit
  LDA playiomap
  BEQ irqinit
  LDX #<irqjob
  STX $0314
irqinit: LDA speed
  BNE ciainit
  LDA #$81
  STA $D01A
  JMP setiomap
ciainit: LDA #$81
  LDX #$01
  STA $DC0D
  STX $DC0E
setiomap: LDA initiomap
  BNE setbank
  LDA #$37
setbank: STA $01
  LDA flags
  PHA
  LDA playnum
  PLP
  JSR init
  LDA initiomap
  BEQ idle
  LDA playiomap
  BEQ run
  LDA #$37
  STA $01
run: CLI
idle: JMP idle
irqjob: LDA $01
  PHA
  LDA playiomap
  STA $01
  LDA #$00
  JSR play
  PLA
  STA $01
  DEC $D019
irqret: LDA $DC0D
  PLA
  TAY
  PLA
  TAX
  PLA
  RTI
exception: .byte $02
"""


class _Sid:
    """A SID device on the machine's I/O page: a reSIDfp chip answering reads.

    Each write that changes a register's value is logged to ``rows`` as sidtrace
    logs it; the chip's queue is flushed once it holds ``0xFFFF`` writes.
    """

    __slots__ = ("rows", "chip", "reads", "last", "queue", "push")

    def __init__(self, rows: List[tuple], chip: int, model: str):
        self.rows, self.chip = rows, chip
        self.reads, self.last = SidReads(model), bytearray(0x20)
        self.queue = self.reads.queue
        self.push = self.queue.append

    def read(self, cycle: int, register: int) -> int:
        """The chip's ``register`` at ``cycle``."""
        return self.reads.read(cycle, register)

    def write(self, cycle: int, register: int, value: int) -> None:
        """Log a changed value and queue the write to the chip."""
        last = self.last
        if value != last[register]:
            self.rows.append((cycle, self.chip, register, value))
            last[register] = value
        self.push(cycle << 16 | register << 8 | value)
        if len(self.queue) > 0xFFFF:
            self.reads.flush(cycle)


class _Anywhere:
    """A watch set holding every address: the trap re-derives the real one."""

    def __contains__(self, addr: int) -> bool:
        return True


_STALE = _Anywhere()


class _PyTrapMachine(Machine):
    """The pure-Python :class:`~vagrant_moses.machine.Machine` with a trap: a fetch at
    a watched address goes to ``trap(machine)`` before decoding it.

    The native machine does this itself (``set_trap``/``arm_at``/``on_port_write``);
    this is the same contract for the Python backend.

    ``trap`` returns True once it has run the call (it has set ``mpu.pc``), and the
    fetch is made again at the new ``pc``. Nothing is watched until the CPU reaches
    ``idle`` (the tune's init has returned); after that the watch is stale at each
    write to the CPU port, and ``trap`` rebuilds it for the new banking.
    """

    def __init__(self, *args, trap: Callable[[Machine], bool], **kwargs):
        super().__init__(*args, **kwargs, backend="python")
        self.trap, self.idle, self.armed = trap, None, False
        self.watch = frozenset()

    def __setitem__(self, addr: int, value: int) -> None:
        super().__setitem__(addr, value)
        if addr <= 1 and self.armed:
            self.watch = _STALE

    def _decode(self) -> None:
        if not self.armed and self.mpu.pc == self.idle:
            self.armed, self.watch = True, _STALE
        while self.mpu.pc in self.watch and self.trap(self):
            pass
        super()._decode()


def _stale(machine) -> None:
    """A CPU port write: the trap's watch set is for the old banking."""
    machine.watch = machine.WATCH_ALL


def _boot(
    machine: Machine,
    image: SidImage,
    subtune: int,
    tune_ntsc: Optional[bool],
    poweron,
) -> Tuple[int, int]:
    """Install sidplayfp's PSID driver and the tune; reset into the driver.

    ``poweron`` is the ``(address, bytes)`` RAM image an RSID starts from
    (:func:`~pysidtracker.roms.poweron_writes`). Returns the reset cycle and the
    driver's idle loop, which the CPU reaches once the tune's init has returned.
    """
    header = image.header
    rsid = not header.is_psid
    base = _driver_base(image)
    code, labels = assemble(_DRIVER, base)
    ram = machine.ram
    ram[:PSID_DRIVER_CLEARED] = bytes(PSID_DRIVER_CLEARED)
    if rsid:
        for addr, data in poweron or KERNAL_VECTORS:
            ram[addr : addr + len(data)] = data
    video = 0 if machine.ntsc else 1
    ram[0x02A6] = video
    exception = labels["exception"]
    vectors = (labels["irqret"], exception, exception)
    for i, vector in enumerate(vectors[: 1 if rsid else 3]):
        ram[0x314 + 2 * i : 0x316 + 2 * i] = vector.to_bytes(2, "little")
    ram[0x328:0x32A] = labels["setiomap"].to_bytes(2, "little")
    init = header.init_address or image.load
    speed = 1 if rsid or header.cia_speed(subtune) else 0
    clock = video if tune_ntsc is None else int(not tune_ntsc)
    ram[base : base + len(code)] = code
    ram[base : base + 11] = bytes(
        (subtune & 0xFF, speed)
        + tuple(init.to_bytes(2, "little"))
        + tuple(header.play_address.to_bytes(2, "little"))
        + (_iomap(header, init), _iomap(header, header.play_address))
        + (video, clock, 0 if rsid else _IRQ_I)
    )
    place_tune(ram, image.load, image.mem[image.load : image.end])
    return machine.reset(labels["cold"]), labels["idle"]


def _resolve_ntsc(image: SidImage, clock: Optional[str]) -> Tuple[bool, Optional[bool]]:
    """``(machine_ntsc, tune_ntsc)``: the header clock picks the machine."""
    tune = {1: False, 2: True}.get((image.header.flags >> 2) & 3)
    if clock is not None:
        key = clock.strip().upper()
        if key not in ("PAL", "NTSC"):
            raise ValueError(f"clock must be 'PAL' or 'NTSC', got {clock!r}")
        return key == "NTSC", tune
    return bool(tune), tune


def to_rows(raw: List[tuple]) -> List[SidtraceRow]:
    """Logged ``(cycle, chip, reg, value)`` tuples as sidtrace :class:`SidtraceRow`."""
    last: Dict[str, int] = {}
    out = []
    for cycle, chip, register, value in raw:
        if chip is None and value in SOURCES:
            last[value] = cycle
        since = [cycle - last[s] if s in last else None for s in SOURCES]
        out.append(SidtraceRow(cycle, *since, chip, register, value))
    return out


class InterruptHost:
    """A PSID or RSID booted and run as sidplayfp runs it, every interrupt live.

    :meth:`rows` are what sidtrace logs for the same tune; :meth:`grid` and
    :meth:`frames` frame them at the play source's raises as the oracle is framed.
    With ``trap``, once init has returned a fetch at an address in the machine's
    ``watch`` set is handed to ``trap(machine)``. The trap sets ``watch`` (stale after
    each banking change) and returns True after running a call and moving ``mpu.pc``.
    """

    def __init__(
        self,
        data: Union[bytes, bytearray, SidImage],
        subtune: int = 0,
        *,
        clock: Optional[str] = None,
        cia_new: bool = False,
        poweron=None,
        trap: Optional[Callable[[Machine], bool]] = None,
        backend: Optional[str] = None,
    ):
        image = data if isinstance(data, SidImage) else SidImage.from_bytes(bytes(data))
        header = image.header
        if header is None:
            raise SidParseError("cannot run a bare .prg: image has no SID header")
        if header.is_rsid and header.flags & 0x02:
            raise SidParseError("RSID BASIC tunes need the BASIC interpreter")
        ntsc, tune_ntsc = _resolve_ntsc(image, clock)
        extra = [0xD000 | b << 4 for b in (header.second_sid, header.third_sid) if b]
        model = reg.sid_model_for_flags(header.flags)
        models = [model] + [
            {1: "6581", 2: "8580"}.get(header.flags >> shift & 3, model)
            for shift in (6, 8)[: len(extra)]
        ]
        self.raw: List[tuple] = []
        sids = {mm.SID_BASE: _Sid(self.raw, 0, models[0])}
        sids.update(
            {a & 0xFFE0: _Sid(self.raw, i, models[i]) for i, a in enumerate(extra, 1)}
        )
        native = backend != "python" and vagrant_machine.NATIVE
        if trap is None or native:
            self.machine = Machine(ntsc, sids, cia_new, self._event, backend=backend)
        else:
            self.machine = _PyTrapMachine(ntsc, sids, cia_new, self._event, trap=trap)
        if poweron is None and header.is_rsid:
            poweron = poweron_writes() or KERNAL_VECTORS
        self.start, idle = _boot(self.machine, image, subtune, tune_ntsc, poweron)
        if trap is not None and native:
            self.machine.set_trap(trap, self.machine.WATCH_ALL)
            self.machine.arm_at(idle)
            self.machine.on_port_write(_stale)
        elif trap is not None:
            self.machine.idle = idle
        video = mm.video(ntsc)
        self.clock_hz, self.frame_cycles = video.clock_hz, video.cycles_per_frame
        self.framed: Optional[Tuple[str, int]] = None

    def _event(self, cycle: int, name: str) -> None:
        self.raw.append((cycle, None, None, name))

    @property
    def cycle(self) -> int:
        """The machine's current cycle (sidplayfp's clock)."""
        return self.machine.t

    @property
    def jammed(self) -> bool:
        """The CPU hit a JAM opcode; sidplayfp stops there."""
        return self.machine.jammed

    def run(self, cycles: int) -> None:
        """Run ``cycles`` more cycles."""
        self.machine.run(self.machine.t + cycles)

    def rows(self) -> List[SidtraceRow]:
        """Every row logged so far, as sidtrace logs it."""
        return to_rows(self.raw)

    def timing(self, chip: int = 0) -> Tuple[Optional[str], Optional[int]]:
        """``(source, cycles_per_frame)`` the oracle frames the rows so far by."""
        rows = self.rows()
        try:
            return play_source(rows, chip) or _most_raised(rows), sidtrace_cadence(
                rows, chip=chip
            )
        except ValueError:
            return None, None

    def grid(self, nframes: int, *, chip: int = 0) -> List[List[int]]:
        """``nframes`` register rows framed as the survey frames the oracle.

        Runs until the play source's raises cover ``nframes`` frames, then frames
        every row at the median raise gap (``oracle.sidtrace_grid``), forward-filled.
        """
        limit = self.start + 4 * nframes * self.frame_cycles + 4 * self.clock_hz
        self.run(self.clock_hz)
        while not self.jammed and self.cycle < limit:
            source, cpf = self.timing(chip)
            raises = raise_cycles(self.rows(), source) if source else []
            covered = _raise_frames(raises, cpf).get(raises[-1], 0) if cpf else 0
            if covered > nframes:
                break
            self.run(max(nframes + 1 - covered, 1) * (cpf or self.frame_cycles))
        rows = self.rows()
        try:
            cpf = sidtrace_cadence(rows, chip=chip)
            grid = sidtrace_grid(rows, chip=chip, cycles_per_frame=cpf)
        except ValueError as exc:
            raise SidParseError("no interrupt was seen to play the tune") from exc
        self.framed = (play_source(rows, chip) or _most_raised(rows), cpf)
        return [grid[min(i, len(grid) - 1)] for i in range(nframes)]

    def frames(
        self,
        *,
        chip: int = 0,
        source: Optional[str] = None,
        cycles_per_frame: Optional[int] = None,
        probe_seconds: float = 2.0,
    ) -> Iterator[List[int]]:
        """Yield register rows one frame at a time, framed at ``source`` raises.

        ``source``/``cycles_per_frame`` default to the oracle's choice over the
        first ``probe_seconds``; a gap of ``k`` frames between raises repeats a
        row ``k`` times. Once the source stops raising, rows advance one frame each.
        """
        if source is None or cycles_per_frame is None:
            self.run(
                max(0, self.start + int(probe_seconds * self.clock_hz) - self.cycle)
            )
            found, cpf = self.timing(chip)
            if found is None:
                raise SidParseError("no interrupt was seen to play the tune")
            source, cycles_per_frame = source or found, cycles_per_frame or cpf
        return self._frames(chip, source, cycles_per_frame)

    def _frames(self, chip: int, source: str, cpf: int) -> Iterator[List[int]]:
        raw, state = self.raw, [0] * reg.SID_REG_COUNT
        pos, prev = 0, None

        def apply(until: int) -> None:
            nonlocal pos
            while pos < len(raw) and raw[pos][0] < until:
                _, ch, register, value = raw[pos]
                if ch == chip and register < reg.SID_REG_COUNT:
                    state[register] = (
                        value & 0x0F if register in reg.PW_HI_REGS else value
                    )
                pos += 1

        scan = 0
        while True:
            raise_at = None
            waited = 0
            while raise_at is None:
                while scan < len(raw) and raise_at is None:
                    if raw[scan][1] is None and raw[scan][3] == source:
                        raise_at = raw[scan][0]
                    scan += 1
                if raise_at is None:
                    if self.jammed or waited > 50:
                        apply(self.cycle + 1)
                        yield list(state)
                        self.run(cpf)
                        continue
                    self.run(cpf)
                    waited += 1
            if self.cycle <= raise_at:
                self.run(raise_at + 1 - self.cycle)
            apply(raise_at)
            if prev is not None:
                for _ in range(max(1, (raise_at - prev + cpf // 2) // cpf)):
                    yield list(state)
            prev = raise_at


def _most_raised(rows) -> Optional[str]:
    raised = Counter(r.value for r in rows if r.chip is None and r.value in SOURCES)
    return raised.most_common(1)[0][0] if raised else None


class InterruptPlayer(MemPlayer):
    """A :class:`~pysidtracker.player.MemPlayer` over an :class:`InterruptHost`.

    Each frame is one play-source raise, as the oracle frames its grid; a fresh
    player's :meth:`render_grid` is the host's oracle-framed :meth:`InterruptHost.grid`.
    """

    def __init__(
        self,
        data: Union[bytes, bytearray, SidImage],
        subtune: int = 0,
        *,
        clock: Optional[str] = None,
        trap: Optional[Callable[[Machine], bool]] = None,
        backend: Optional[str] = None,
    ):
        self._data, self._clock, self._trap = data, clock, trap
        self._backend = backend
        self.host: Optional[InterruptHost] = None
        self._iter: Optional[Iterator[List[int]]] = None
        self._row = [0] * reg.SID_REG_COUNT
        self._played = 0
        super().__init__(b"", 0, subtune)

    def _init(self, subtune: int) -> None:
        self.host = InterruptHost(
            self._data,
            subtune,
            clock=self._clock,
            trap=self._trap,
            backend=self._backend,
        )

    def _frame(self) -> None:
        if self._iter is None:
            source, cpf = self.host.framed or (None, None)
            self._iter = self.host.frames(source=source, cycles_per_frame=cpf)
            for _ in range(self._played):
                next(self._iter)
        self._row = next(self._iter)
        self._played += 1

    def snapshot(self) -> List[int]:
        return list(self._row)

    def render_grid(self, nframes: int) -> List[List[int]]:
        if self._played:
            return super().render_grid(nframes)
        grid = self.host.grid(nframes)
        self._played = nframes
        self._row = list(grid[-1])
        self.regs = list(self._row)
        return grid


class TunePlayer(EmuPlayer):
    """A PSID/RSID's own ``init``/``play``, one frame per play call, as the oracle plays it.

    A tune with a play address runs as an :class:`~pysidtracker.oracle.EmuPlayer`
    booted from its header (load, init, SID model, PAL/NTSC flag); play address 0
    (the init installs its own interrupt) runs on an :class:`InterruptPlayer`,
    framed at its play source's raises. ``cadence`` is the tune's
    :func:`~pysidtracker.cadence.playroutine_cadence`, ``cycles_per_frame`` its
    period. Errors are raised as ``error_class``, which a format package sets to its
    own :class:`~pysidtracker.errors.SidParseError` subclass.
    """

    error_class: Type[SidParseError] = SidParseError

    def __init__(
        self,
        data: Union[bytes, bytearray, SidImage],
        subtune: int = 0,
        *,
        max_cycles: int = 8_000_000,
    ):
        image = data if isinstance(data, SidImage) else SidImage.from_bytes(bytes(data))
        header = image.header
        if header is None:
            raise self.error_class("cannot play a bare .prg: image has no SID header")
        init = header.init_address or header.real_load_address
        self._sid = image.container + image.image
        self._subtune = subtune
        self._cadence: Optional[Cadence] = None
        self._hosted = None if header.play_address else InterruptPlayer(image, subtune)
        super().__init__(
            image.image,
            image.load,
            init,
            header.play_address or init,
            subtune,
            max_cycles=max_cycles,
            sid_model=reg.sid_model_for_flags(header.flags),
            ntsc=header.is_ntsc,
        )

    @property
    def cadence(self) -> Cadence:
        """The tune's play-routine :class:`~pysidtracker.cadence.Cadence` (cached)."""
        if self._cadence is None:
            self._cadence = playroutine_cadence(self._sid, subtune=self._subtune)
        return self._cadence

    @property
    def cycles_per_frame(self) -> int:
        """CPU cycles between play calls (the period a WAV render spaces frames by)."""
        return self.cadence.cycles_per_call

    def _init(self, subtune: int) -> None:
        if self._hosted is None:
            super()._init(subtune)

    def _frame(self) -> None:
        if self._hosted is None:
            super()._frame()
        else:
            self._hosted.play_frame()

    def snapshot(self) -> List[int]:
        if self._hosted is None:
            return super().snapshot()
        return self._hosted.snapshot()

    def render_grid(self, nframes: int) -> List[List[int]]:
        """``nframes`` rows; a hosted tune's are framed at its play source's raises.

        A hosted tune whose CPU never entered an interrupt vector (an ``*_entry``
        row) is not playing: it raises ``error_class``.
        """
        if self._hosted is None:
            return super().render_grid(nframes)
        try:
            grid = self._hosted.render_grid(nframes)
        except SidParseError as exc:
            raise self.error_class(str(exc)) from exc
        if not any(
            r[1] is None and r[3].endswith("_entry") for r in self._hosted.host.raw
        ):
            raise self.error_class("no interrupt was seen to play the tune")
        return grid

    def register_writes(self, nframes: int) -> Iterator[RegWrite]:
        """The post-init baseline at clock 0, then ``nframes`` frames a period apart."""
        return register_writes_from_player(self, nframes, self.cycles_per_frame)


def host_cadence(
    image_or_bytes: Union[SidImage, bytes, bytearray],
    *,
    clock: Optional[str] = None,
    subtune: int = 0,
    seconds: float = 10.0,
) -> Cadence:
    """The :class:`~pysidtracker.cadence.Cadence` sidplayfp plays the tune at.

    Runs the host ``seconds`` and measures the play source (most distinct SID
    registers, raised at least 3 times) over its raises after the first.
    """
    host = InterruptHost(image_or_bytes, subtune, clock=clock)
    host.run(int(seconds * host.clock_hz))
    rows = host.rows()
    frame, hz = host.frame_cycles, host.clock_hz
    source = play_source(rows, min_raises=3) if nested(rows) else None
    steady = raise_cycles(rows, source)[1:] if source else []
    if len(steady) < 2:
        return Cadence(frame, None, hz)
    gaps = [b - a for a, b in zip(steady, steady[1:])]
    span = steady[-1] - steady[0]
    trigger = {
        "video_irq": (
            TriggerSource.NTSC_VIDEO
            if frame == mm.NTSC.cycles_per_frame
            else TriggerSource.PAL_VIDEO
        ),
        "cia_irq": TriggerSource.CIA_TIMER,
        "nmi": TriggerSource.NMI_TIMER,
    }[source]
    dynamic = source != "video_irq" and len(set(gaps)) > 1
    per_frame = frame * len(gaps) / span
    return Cadence(round(span / len(gaps)), trigger, hz, None, dynamic, per_frame)


__all__ = [
    "KERNAL_VECTORS",
    "InterruptHost",
    "InterruptPlayer",
    "TunePlayer",
    "boot_cycle",
    "host_cadence",
    "to_rows",
]
