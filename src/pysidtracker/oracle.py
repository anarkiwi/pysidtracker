"""Ground-truth per-frame SID register grids for byte-exact player validation.

Two ways to build the reference grid every format package validates its player
against, plus the readers/aligner around them:

* :func:`register_grid` -- run a tune's ``init`` then ``play`` on a jennings 6502
  (reusing :mod:`pysidtracker.trace`'s run-to-RTS mechanic) and sample the 25
  SID registers ``$D400..$D418`` per frame. Requires jennings (a core dependency).
* :func:`grid_from_writes` -- the pure-stdlib framer that turns a
  ``(clock, reg, val)`` write stream (e.g. a ``preframr-sidtrace`` capture read
  by :func:`read_sidwr`, or a :mod:`pysidtracker.reglog` log) into the same
  per-frame grid: anchor frame 0 to the first play call, forward-fill, and
  nibble-mask the pulse-width-high registers.

:func:`aligned_match` compares a rendered grid to an oracle grid, tolerating a
few leading silent frames. This consolidates the jennings oracle (pydefmon,
pyjch) and the sidtrace framer (pyjch/pymusicassembler/pyfuturecomposer
conftests, pydmcsid helpers).
"""

from __future__ import annotations

import bisect
import csv
import io
import math
import os
import shutil
import statistics
import struct
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

from vagrant_moses import memmap as mm

from . import registers as reg
from .cadence import playroutine_cadence
from .emu import run_to_rts, run_to_rts_until, set_iomap, wire_mpu
from .errors import SidError, SidParseError
from .image import SidImage
from .player import MemPlayer
from .poweron import place_tune, power_on_ram, psid_init_bank, psid_play_bank

# preframr-sidtrace ``.sidwr.bin`` record: (clock, addr, reg, val).
_SIDWR_REC = struct.Struct("<qHBB")

# The published ``sidplayfp`` register-trace oracle (deterministic power-on delay).
SIDTRACE_IMAGE = "anarkiwi/sidtrace:latest"

# The ``.csv`` column order emitted by the sidtrace oracle.
SIDTRACE_COLUMNS = (
    "cycle",
    "cycle_since_nmi",
    "cycle_since_video_irq",
    "cycle_since_cia_irq",
    "chip",
    "reg",
    "value",
)

# Interrupt raise rows (sidtrace >= v0.5.0), and the source each handler-entry
# row (sidtrace >= v0.6.0) enters.
RAISE_SOURCES = ("video_irq", "cia_irq", "nmi")
ENTRY_SOURCES = {f"{s}_entry": s for s in RAISE_SOURCES}

# The libsidplayfp PSID driver (``psiddrv.a65``) writes maximum volume to $D418
# before running a tune's init, so a tune that never sets volume itself is still
# audible. Mirror that cold-start default when rendering a reference grid.
SID_VOLUME = mm.SID_BASE + 0x18
DRIVER_VOLUME = 0x0F
# The KERNAL PAL/NTSC flag the PSID driver sets for the machine it plays on: 1 PAL.
PAL_FLAG = 0x02A6


class ShortTrace(ValueError):
    """A sidtrace trace covers fewer play calls than its reference needs."""


class SidtraceUnavailable(SidError):
    """The sidtrace oracle could not be run (Docker missing or the render failed)."""


def boot_flat(
    image: SidImage,
    subtune: int = 0,
    max_cycles: int = 8_000_000,
    illegal_opcodes: bool = True,
    setup=None,
    init_run=None,
):
    """``(subject, mpu, mem, play)``: ``image`` booted on jennings, ``init`` run, ready to play.

    Power-on RAM, the tune placed with the BASIC ``LOAD`` pointers
    (:func:`~pysidtracker.poweron.place_tune`), ``$D418`` at the driver's maximum volume, the KERNAL PAL/NTSC flag
    (:data:`PAL_FLAG`) for the header's clock (NTSC only for an NTSC-only tune),
    ``init`` banked and run with ``A = subtune``, then the play address's iomap, as
    the PSID driver does.
    ``setup(mpu, subject)``, if given, runs before ``init`` (e.g. to install a
    :func:`~pysidtracker.emu.observe_data_access` observer that sees init too).
    ``init_run(mpu, mem, init, subtune, max_cycles)``, if given, runs ``init`` in
    place of :func:`~pysidtracker.emu.run_to_rts` (e.g. to stop it part way).
    """
    subject = power_on_ram()
    subject[PAL_FLAG] = int(not image.header.is_ntsc)
    place_tune(subject, image.load, image.image)
    subject[SID_VOLUME] = DRIVER_VOLUME  # PSID driver cold-start: maximum volume
    init = image.header.init_address or image.header.real_load_address
    subject[1] = psid_init_bank(init)
    mpu, mem = wire_mpu(
        subject, illegal_opcodes, reg.sid_model_for_flags(image.header.flags)
    )
    if setup is not None:
        setup(mpu, subject)
    (init_run or run_to_rts)(mpu, mem, init, subtune, max_cycles)
    play = image.header.play_address or init
    set_iomap(mpu, psid_play_bank(play))
    return subject, mpu, mem, play


class Entry(NamedTuple):
    """The machine state at the instruction :func:`run_until` stopped before."""

    pc: int
    a: int
    x: int
    y: int
    sp: int
    p: int
    cycles: int
    phase: str  # "init", or "play" for the play call it entered in
    call: int  # 0 for init, else the 1-based play call
    memory: Optional[bytes]  # the 64K RAM image, or None without ``snapshot``


def run_until(
    image: SidImage,
    pcs: Iterable[int],
    *,
    subtune: int = 0,
    play_calls: int = 0,
    snapshot: bool = True,
    max_cycles: int = 8_000_000,
    illegal_opcodes: bool = True,
    setup=None,
) -> Optional[Entry]:
    """Step ``image``'s ``init`` (then up to ``play_calls`` ``play`` calls, booted as
    :func:`boot_flat` does) until execution enters an address in ``pcs``.

    Returns the :class:`Entry` -- registers at the instruction about to run, plus
    a copy of memory when ``snapshot`` -- or ``None`` if none is entered.
    """
    targets = frozenset(pcs)
    found: List[Entry] = []

    def stop(mpu, subject, phase, call):
        found.append(
            Entry(
                mpu.pc,
                mpu.a,
                mpu.x,
                mpu.y,
                mpu.sp,
                mpu.p,
                mpu.processorCycles,
                phase,
                call,
                bytes(subject) if snapshot else None,
            )
        )

    subject_box: List[bytearray] = []

    def init_run(mpu, mem, init, tune, cycles):
        if run_to_rts_until(mpu, mem, init, tune, targets, cycles):
            stop(mpu, subject_box[0], "init", 0)

    def wrapped_setup(mpu, subject):
        subject_box.append(subject)
        if setup is not None:
            setup(mpu, subject)

    subject, mpu, mem, play = boot_flat(
        image, subtune, max_cycles, illegal_opcodes, wrapped_setup, init_run
    )
    for call in range(1, play_calls + 1):
        if found:
            break
        if run_to_rts_until(mpu, mem, play, 0, targets, max_cycles):
            stop(mpu, subject, "play", call)
    return found[0] if found else None


def register_grid(
    image_or_bytes,
    nframes: int,
    *,
    subtune: int = 0,
    illegal_opcodes: bool = False,
    max_cycles: int = 8_000_000,
) -> List[List[int]]:
    """Per-frame SID register grid from running a tune on jennings.

    ``image_or_bytes`` is a :class:`~pysidtracker.image.SidImage` or PSID/RSID
    (or ``.prg``) bytes. Runs ``init`` (accumulator = ``subtune``), then
    ``nframes`` ``play`` calls, sampling ``$D400..$D418`` (25 registers) after
    each play. ``illegal_opcodes`` is retained for API compatibility; jennings
    decodes the NMOS illegals natively regardless (replays such as defMON need them).

    ``$D418`` (volume) is pre-seeded to ``$0F``, mirroring the libsidplayfp PSID
    driver's cold-start default, so a tune that relies on the driver's maximum
    volume (rather than setting it itself) matches the sidtrace oracle.

    The image starts as libsidplayfp's power-on RAM (:func:`~pysidtracker.poweron.power_on_ram`),
    so a register the tune never writes reads the C64 power-on byte, not zero, as
    the sidtrace oracle records it.

    Requires jennings; raises :class:`~pysidtracker.errors.EmulatorUnavailable` if
    it is missing and :class:`~pysidtracker.errors.SidParseError` if the image
    has no init address.
    """
    if isinstance(image_or_bytes, (bytes, bytearray)):
        image = SidImage.from_bytes(bytes(image_or_bytes))
    else:
        image = image_or_bytes
    if image.header is None:
        raise SidParseError("cannot build a register grid: image has no SID header")

    subject, mpu, mem, play_address = boot_flat(
        image, subtune, max_cycles, illegal_opcodes
    )
    rows: List[List[int]] = []
    for _ in range(nframes):
        run_to_rts(mpu, mem, play_address, 0, max_cycles)
        rows.append(
            [
                (
                    (subject[mm.SID_BASE + i] & 0x0F)
                    if i in reg.PW_HI_REGS
                    else subject[mm.SID_BASE + i]
                )
                for i in range(reg.SID_REG_COUNT)
            ]
        )
    return rows


class EmuPlayer(MemPlayer):
    """A :class:`~pysidtracker.player.MemPlayer` that runs the tune's OWN driver.

    Where a native transcription models one specific playroutine in Python, this
    plays *any* driver version byte-exactly by executing the tune's real 6502
    ``init`` + ``play`` code on jennings (the same mechanic as :func:`register_grid`),
    one ``play`` call per :meth:`~pysidtracker.player.MemPlayer.play_frame`. Use
    it for driver versions that have no native transcription.

    With ``cycles_per_frame`` each play call starts a period after the last, as
    the interrupt calls it, so raster and osc3/env3 reads see the real elapsed
    time, and an interrupt raised while the previous call still runs is lost;
    ``sid_model`` is the chip those reads come from. The KERNAL PAL/NTSC flag
    (:data:`PAL_FLAG`) is set for ``ntsc`` unless the image covers it, as
    :func:`boot_flat` sets it.

    :attr:`~pysidtracker.player.MemPlayer.MASK_PW_HI` is set: ``snapshot``
    masks the pulse-width-high registers to a nibble (the SID ignores their
    upper bits), so the per-frame grid matches the reference-grid convention.
    The frame *cadence* (PAL/NTSC) is the caller's concern -- one ``play`` call is one frame; an
    oracle grid it is compared against is framed at the tune's clock (see
    :func:`~pysidtracker.registers.cycles_per_frame_for_flags`).

    Requires jennings (a core dependency).
    """

    MASK_PW_HI = True

    def __init__(
        self,
        image: bytes,
        load: int,
        init: int,
        play: int,
        subtune: int = 0,
        *,
        illegal_opcodes: bool = False,
        max_cycles: int = 8_000_000,
        sid_model: str = "6581",
        cycles_per_frame: Optional[int] = None,
        ntsc: bool = False,
    ):
        self._pal_flag = None if load <= PAL_FLAG < load + len(image) else int(not ntsc)
        self._init_addr = init & 0xFFFF
        self._play_addr = play & 0xFFFF
        self._illegal = illegal_opcodes
        self._max_cycles = max_cycles
        self._sid_model = sid_model
        self._period = cycles_per_frame
        self._mpu = None
        self._obs = None
        self._due = 0
        super().__init__(image, load, subtune)

    def _init(self, subtune: int) -> None:
        if self._pal_flag is not None:
            self._mem[PAL_FLAG] = self._pal_flag
        # The driver banks the init's ROM space before calling it (KERNAL/BASIC in).
        self._mem[1] = psid_init_bank(self._init_addr)
        self._mpu, self._obs = wire_mpu(self._mem, self._illegal, self._sid_model)
        run_to_rts(self._mpu, self._obs, self._init_addr, subtune, self._max_cycles)
        set_iomap(self._mpu, psid_play_bank(self._play_addr))
        self._due = self._mpu.processorCycles

    def _frame(self) -> None:
        if self._period:
            late = self._mpu.processorCycles > self._due
            self._mpu.processorCycles = max(self._mpu.processorCycles, self._due)
            self._due += self._period
            if late:
                return
        run_to_rts(self._mpu, self._obs, self._play_addr, 0, self._max_cycles)


def _frame_grid(
    framed: Iterable[Tuple[int, int, int]], reg_count: int, pw_hi_regs: Iterable[int]
) -> List[List[int]]:
    """A forward-filled grid from ``(frame, reg, val)`` writes in frame order.

    Frame ``-1`` is the baseline frame 0 starts from; pulse-width highs keep 4 bits.
    """
    pw = set(pw_hi_regs)
    cur_row = [0] * reg_count
    rows: List[List[int]] = []
    for frame, register, val in framed:
        if 0 <= register < reg_count:
            while len(rows) < frame:
                rows.append(cur_row[:])
            cur_row[register] = (val & 0x0F) if register in pw else val
    rows.append(cur_row[:])
    return rows


def grid_from_writes(
    writes: Sequence[Tuple[int, int, int]],
    *,
    cycles_per_frame: int = mm.PAL.cycles_per_frame,
    reg_count: int = 25,
    pw_hi_regs: Iterable[int] = reg.PW_HI_REGS,
    gap: int = 10000,
) -> List[List[int]]:
    """Frame a ``(clock, reg, val)`` write stream into a per-frame grid.

    Frame 0 is anchored to the first play call -- the first write after a
    ``> gap``-cycle gap; earlier writes form its baseline. A write lands in the
    nearest frame (``(clock - t0 + cpf // 2) // cpf``).
    """
    if not writes:
        return []
    cyc = [w[0] for w in writes]
    t0 = next((cur for prev, cur in zip(cyc, cyc[1:]) if cur - prev > gap), cyc[0])
    half = cycles_per_frame // 2
    return _frame_grid(
        (
            (-1 if clock < t0 else (clock - t0 + half) // cycles_per_frame, r, v)
            for clock, r, v in writes
        ),
        reg_count,
        pw_hi_regs,
    )


def read_sidwr(path, *, reg_count: int = 25) -> List[Tuple[int, int, int]]:
    """Read a ``preframr-sidtrace`` ``.sidwr.bin`` into ``(clock, reg, val)``.

    Each fixed-size record is ``struct.Struct("<qHBB")`` = (clock, addr, reg,
    val); the addr field is dropped and records whose ``reg >= reg_count`` are
    skipped, matching the existing per-repo readers.
    """
    blob = Path(path).read_bytes()
    out: List[Tuple[int, int, int]] = []
    for off in range(0, len(blob) - _SIDWR_REC.size + 1, _SIDWR_REC.size):
        clock, _addr, register, val = _SIDWR_REC.unpack_from(blob, off)
        if register < reg_count:
            out.append((clock, register, val))
    return out


def aligned_match(
    oracle: Sequence[Sequence[int]],
    rendered: Sequence[Sequence[int]],
    *,
    max_lead: int = 4,
) -> bool:
    """True if ``rendered`` matches ``oracle`` within ``max_lead`` silent frames.

    Tries aligning ``oracle`` at each lead offset ``0..max_lead`` for which the
    skipped leading frames of ``rendered`` are all equal to its first frame
    (silent lead-in), returning True on the first exact match.
    """
    if not rendered:
        return False
    baseline = rendered[0]
    for lead in range(max_lead + 1):
        if lead and (lead > len(rendered) or rendered[lead - 1] != baseline):
            break
        aligned = rendered[lead : lead + len(oracle)]
        if len(aligned) < len(oracle):
            continue
        if all(oracle[f] == aligned[f] for f in range(len(oracle))):
            return True
    return False


class SidtraceRow(NamedTuple):
    """One sidtrace CSV row: a changed SID register write or an interrupt event.

    A write has int ``chip``, ``reg`` and ``value``; an interrupt row has ``chip``
    and ``reg`` ``None`` and ``value`` a raise, a ``*_entry`` or ``"rti"`` (see
    ``docs/oracle-testing.md``). Delta fields are ``None`` until their source fires.
    """

    cycle: int
    since_nmi: Optional[int]
    since_video_irq: Optional[int]
    since_cia_irq: Optional[int]
    chip: Optional[int]
    reg: Optional[int]
    value: Union[int, str]


def read_sidtrace(path) -> List[SidtraceRow]:
    """Read a sidtrace CSV into :class:`SidtraceRow` list.

    A ``.zst`` suffix is transparently zstd-decompressed (``zstandard`` is
    imported lazily, so it is only needed when reading compressed traces). Empty
    interrupt-delta cells become ``None``; the header row is skipped.
    """
    path = Path(path)
    data = path.read_bytes()
    if path.suffix == ".zst":
        import zstandard  # lazy: only needed to read compressed traces

        data = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)).read()
    reader = csv.reader(io.StringIO(data.decode("utf-8")))
    next(reader, None)  # header
    rows: List[SidtraceRow] = []
    for rec in reader:
        if not rec:
            continue
        cyc, nmi, vid, cia, chip, register, val = rec
        rows.append(
            SidtraceRow(
                int(cyc),
                int(nmi) if nmi else None,
                int(vid) if vid else None,
                int(cia) if cia else None,
                int(chip) if chip else None,
                int(register) if register else None,
                int(val) if chip else val,
            )
        )
    return rows


def nested(rows: Sequence[SidtraceRow]) -> bool:
    """Whether the trace logs RTIs (sidtrace >= v0.6.0), so handler nesting is known."""
    return any(r.value == "rti" and r.chip is None for r in rows)


def handler_sources(rows: Sequence[SidtraceRow]) -> List[Optional[str]]:
    """The interrupt source whose handler executes each row, ``None`` otherwise.

    Each ``*_entry`` pushes its source (``brk_entry`` the running one) and each
    ``rti`` pops, so a write belongs to the innermost running handler. A trace
    with no ``rti`` rows (a handler that never returned, sidtrace < v0.6.0, or a
    main loop that set the I flag) cannot be nested, so every row is ``None``.
    """
    if not nested(rows):
        return [None] * len(rows)
    stack: List[Optional[str]] = [None]
    out: List[Optional[str]] = []
    for row in rows:
        if row.chip is None:
            if row.value == "rti":
                if len(stack) > 1:
                    stack.pop()
            elif row.value == "brk_entry":
                stack.append(stack[-1])
            elif row.value in ENTRY_SOURCES:
                stack.append(ENTRY_SOURCES[row.value])
        out.append(stack[-1])
    return out


def play_source(
    rows: Sequence[SidtraceRow], chip: Optional[int] = None, min_raises: int = 1
) -> Optional[str]:
    """The source whose handlers drive the most distinct SID registers, then writes.

    Only writes to ``chip`` count (every chip when ``None``), and only sources
    raised at least ``min_raises`` times compete; ``None`` if none wrote.
    """
    raised = Counter(r.value for r in rows if r.chip is None)
    regs: Dict[str, set] = {s: set() for s in RAISE_SOURCES}
    writes: Counter = Counter()
    for row, source in zip(rows, handler_sources(rows)):
        if source and row.chip is not None and chip in (None, row.chip):
            regs[source].add((row.chip, row.reg))
            writes[source] += 1
    return max(
        (s for s in RAISE_SOURCES if writes[s] and raised[s] >= min_raises),
        key=lambda s: (len(regs[s]), writes[s]),
        default=None,
    )


def raise_cycles(rows: Sequence[SidtraceRow], source: str) -> List[int]:
    """The cycles ``source`` was raised, from its interrupt rows."""
    return [r.cycle for r in rows if r.chip is None and r.value == source]


def _raises(rows: Sequence[SidtraceRow], chip: int = 0) -> List[int]:
    """Raise cycles of the :func:`play_source` of ``chip``, else the most raised.

    ``ValueError`` without interrupt rows (nothing the tune could play at).
    """
    raised = Counter(r.value for r in rows if r.value in RAISE_SOURCES)
    if not raised:
        raise ValueError("sidtrace trace has no interrupt rows: re-render it")
    source = play_source(rows, chip) or raised.most_common(1)[0][0]
    return raise_cycles(rows, source)


def sidtrace_cadence(rows: Sequence[SidtraceRow], *, chip: int = 0) -> Optional[int]:
    """The play cadence (cycles per call): the median gap between :func:`_raises`.

    ``None`` with fewer than two raises, a never-returning tune, or no interrupt
    rows (the latter raises ``ValueError``).
    """
    raises = _raises(rows, chip)
    diffs = [b - a for a, b in zip(raises, raises[1:])]
    return int(statistics.median(diffs)) if diffs else None


def _raise_frames(raises: Sequence[int], cpf: int) -> Dict[int, int]:
    """Each raise's frame at a fixed ``cpf`` cadence: one per raise, plus gap calls."""
    out: Dict[int, int] = {}
    frame, prev = 0, None
    for at in raises:
        if prev is not None:
            frame += max(1, (at - prev + cpf // 2) // cpf)
        out[at], prev = frame, at
    return out


def sidtrace_grid(
    rows: Sequence[SidtraceRow],
    *,
    chip: int = 0,
    reg_count: int = 25,
    cycles_per_frame: Optional[int] = None,
    pw_hi_regs: Iterable[int] = reg.PW_HI_REGS,
) -> List[List[int]]:
    """Frame a sidtrace CSV into a per-frame register grid for ``chip``.

    Each interrupt raise of the timing source (:func:`_raises`) is one frame,
    or ``cycles_per_frame`` fixes the cadence; a write outside any interrupt
    goes to the last raise before it, or the baseline.
    """
    raises = _raises(rows, chip)
    frames = (
        _raise_frames(raises, cycles_per_frame)
        if cycles_per_frame
        else {at: i for i, at in enumerate(raises)}
    )

    def frame_of(row: SidtraceRow) -> int:
        i = bisect.bisect_right(raises, row.cycle)
        return frames[raises[i - 1]] if i else -1

    framed = sorted(
        ((frame_of(row), row.reg, row.value) for row in rows if row.chip == chip),
        key=lambda w: w[0],
    )
    return _frame_grid(framed, reg_count, pw_hi_regs)


def sidtrace_image_id(image: str = SIDTRACE_IMAGE, docker: str = "docker"):
    """The local id of ``image``, pulled if absent; ``None`` without docker or image."""
    inspect = [docker, "image", "inspect", "-f", "{{.Id}}", image]
    try:
        for cmd in (inspect, [docker, "pull", "-q", image], inspect):
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if cmd is inspect and res.returncode == 0:
                return res.stdout.strip()
    except FileNotFoundError:
        pass
    return None


def sidtrace_calls(
    rows: Sequence[SidtraceRow],
    *,
    chip: int = 0,
    cycles_per_frame: Optional[int] = None,
) -> int:
    """The play calls ``rows`` covers: the :func:`sidtrace_grid` frames before the
    last raise of its timing source, whose handler the render may have cut short.

    ``cycles_per_frame`` counts gap calls as :func:`sidtrace_grid` frames them;
    ``None`` counts one per raise. A trace without interrupt rows covers none.
    """
    try:
        raises = _raises(rows, chip)
    except ValueError:
        return 0
    if cycles_per_frame:
        return _raise_frames(raises, cycles_per_frame)[raises[-1]]
    return len(raises) - 1


def _trace_calls(rows: Sequence[SidtraceRow]) -> int:
    """:func:`sidtrace_calls` framed at the trace's own cadence."""
    try:
        cpf = sidtrace_cadence(rows)
    except ValueError:
        return 0
    return sidtrace_calls(rows, cycles_per_frame=cpf)


def render_seconds(seconds: float) -> int:
    """Whole sidtrace ``-t`` seconds covering ``seconds`` of play: rounded up, plus
    one for the tune's init and the last call the time limit cuts short."""
    return math.ceil(seconds) + 1


def play_seconds(tune, calls: int, song: Optional[int] = None) -> float:
    """Seconds ``calls`` play calls of 1-based ``song`` (default the header's start
    song) of ``tune`` (a path or bytes) span at its :func:`playroutine_cadence`."""
    data = (
        bytes(tune) if isinstance(tune, (bytes, bytearray)) else Path(tune).read_bytes()
    )
    image = SidImage.from_bytes(data)
    if song is None:
        song = image.header.start_song if image.header is not None else 1
    cadence = playroutine_cadence(image, subtune=max(song, 1) - 1)
    return calls / cadence.calls_per_second


def cached_sidtrace(
    tune_path, csv_path, *, calls: int = 0, force: bool = False, **kw
) -> List[SidtraceRow]:
    """The rows of the trace cached at ``csv_path``, rendered when it is missing.

    :func:`run_sidtrace` (``kw``) renders it when missing, ``force``d, or covering
    fewer than ``calls`` play calls (:func:`sidtrace_calls` at the trace's own
    cadence); :class:`ShortTrace` if a fresh render still falls short.
    Whether a cached trace is still current is decided by the caller's oracle
    cache stamp (the sidtrace image id, see ``survey.oracle_cache``), which
    wipes traces rendered by an older sidtrace. A valid trace without ``rti``
    rows (a never-returning main loop) is reused as-is, since re-rendering
    cannot add one.
    """
    csv_path = Path(csv_path)
    fresh = force or not csv_path.exists()
    if fresh:
        run_sidtrace(tune_path, csv_path, **kw)
    rows = read_sidtrace(csv_path)
    if calls and _trace_calls(rows) < calls and not fresh:
        run_sidtrace(tune_path, csv_path, **kw)
        rows = read_sidtrace(csv_path)
    have = _trace_calls(rows) if calls else calls
    if have < calls:
        raise ShortTrace(
            f"{Path(tune_path).name}: a fresh sidtrace render covers {have} "
            f"of {calls} play calls"
        )
    return rows


def run_sidtrace(
    tune_path,
    out_path,
    *,
    seconds: int = 60,
    image: str = SIDTRACE_IMAGE,
    docker: str = "docker",
    extra_args: Sequence[str] = (),
    song: Optional[int] = None,
    user: Optional[str] = None,
) -> Path:
    """Render ``tune_path`` under the sidtrace Docker oracle to ``out_path``.

    Renders in a temp dir beside ``out_path`` and moves it in atomically. ``song``
    (1-based) adds ``-o<song>``; ``user`` (``uid:gid``) sets the owner. Raises
    :class:`SidtraceUnavailable` on failure.
    """
    tune_path = Path(tune_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Mount a private dir beside the destination (same filesystem => atomic move,
    # and a Docker-daemon-visible path).
    work = Path(tempfile.mkdtemp(dir=out_path.parent, prefix=".sidtrace-"))
    try:
        tune_local = work / tune_path.name
        tune_local.write_bytes(tune_path.read_bytes())
        result = work / "trace.csv.zst"
        if user:
            work.chmod(0o777)
        cmd = [
            docker,
            "run",
            "--rm",
            *(["--user", user] if user else []),
            "-v",
            f"{work.resolve()}:/work",
            image,
            result.name,
            tune_local.name,
            f"-t{seconds}",
            *([f"-o{song}"] if song else []),
            *extra_args,
        ]
        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        except FileNotFoundError as exc:  # docker not installed
            raise SidtraceUnavailable(f"{docker} not found: {exc}") from exc
        except subprocess.CalledProcessError as exc:
            err = exc.stderr.decode("utf-8", "replace") if exc.stderr else ""
            raise SidtraceUnavailable(
                f"sidtrace render of {tune_path.name} failed: {err.strip()}"
            ) from exc
        if not result.exists():
            raise SidtraceUnavailable(
                f"sidtrace produced no output for {tune_path.name}"
            )
        os.replace(result, out_path)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out_path
