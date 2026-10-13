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
import os
import shutil
import statistics
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from . import registers as reg
from .emu import run_to_rts, wire_mpu
from .errors import SidError, SidParseError
from .image import SidImage
from .player import MemPlayer

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

# The libsidplayfp PSID driver (``psiddrv.a65``) writes maximum volume to $D418
# before running a tune's init, so a tune that never sets volume itself is still
# audible. Mirror that cold-start default when rendering a reference grid.
SID_VOLUME = reg.SID_BASE + 0x18
DRIVER_VOLUME = 0x0F


class SidtraceUnavailable(SidError):
    """The sidtrace oracle could not be run (Docker missing or the render failed)."""


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

    subject = image.mem
    subject[SID_VOLUME] = DRIVER_VOLUME  # PSID driver cold-start: maximum volume
    mpu, mem = wire_mpu(
        subject, illegal_opcodes, reg.sid_model_for_flags(image.header.flags)
    )

    init_address = image.header.init_address or image.header.real_load_address
    run_to_rts(mpu, mem, init_address, subtune, max_cycles)

    play_address = image.header.play_address or init_address
    rows: List[List[int]] = []
    for _ in range(nframes):
        run_to_rts(mpu, mem, play_address, 0, max_cycles)
        rows.append([subject[reg.SID_BASE + i] for i in range(reg.SID_REG_COUNT)])
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
    ``sid_model`` is the chip those reads come from.

    ``snapshot`` masks the pulse-width-high registers to a nibble (the SID
    ignores their upper bits, see :data:`~pysidtracker.registers.PW_HI_REGS`), so
    the per-frame grid matches the reference-grid convention. The frame *cadence*
    (PAL/NTSC) is the caller's concern -- one ``play`` call is one frame; an
    oracle grid it is compared against is framed at the tune's clock (see
    :func:`~pysidtracker.registers.cycles_per_frame_for_flags`).

    Requires jennings (a core dependency).
    """

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
    ):
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
        self._mpu, self._obs = wire_mpu(self._mem, self._illegal, self._sid_model)
        run_to_rts(self._mpu, self._obs, self._init_addr, subtune, self._max_cycles)
        self._due = self._mpu.processorCycles

    def _frame(self) -> None:
        if self._period:
            late = self._mpu.processorCycles > self._due
            self._mpu.processorCycles = max(self._mpu.processorCycles, self._due)
            self._due += self._period
            if late:
                return
        run_to_rts(self._mpu, self._obs, self._play_addr, 0, self._max_cycles)

    def snapshot(self) -> List[int]:
        base = self.SID_BASE
        return [
            (self._mem[base + i] & 0x0F) if i in reg.PW_HI_REGS else self._mem[base + i]
            for i in range(self.REG_COUNT)
        ]


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
    cycles_per_frame: int = reg.PAL_CYCLES_PER_FRAME,
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
    """One row of a sidtrace oracle CSV (a single changed SID register write).

    The three interrupt-delta fields are ``None`` until their source first fires
    (an empty CSV cell). ``reg`` is the register OFFSET ``0..31`` relative to the
    chip's base, ``chip`` the SID index (``0`` = ``$D400``).
    """

    cycle: int
    since_nmi: Optional[int]
    since_video_irq: Optional[int]
    since_cia_irq: Optional[int]
    chip: int
    reg: int
    value: int


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
                int(chip),
                int(register),
                int(val),
            )
        )
    return rows


def _raise_cycle(row: SidtraceRow) -> Optional[int]:
    """The cycle the interrupt that made ``row``'s write was raised, if any."""
    since = (
        row.since_video_irq if row.since_video_irq is not None else row.since_cia_irq
    )
    return None if since is None else row.cycle - since


def sidtrace_cadence(rows: Sequence[SidtraceRow], *, chip: int = 0) -> Optional[int]:
    """The play cadence (cycles per frame) implied by a sidtrace CSV.

    Each write records the cycle offset since its frame's interrupt was raised
    (``since_video_irq`` for VIC-timed tunes, else ``since_cia_irq``), so
    ``cycle - offset`` is that frame's interrupt-raise cycle. The median gap
    between consecutive distinct raise cycles is the frame period, robust to
    multi-speed and variable-tempo players. ``None`` if fewer than two frames.
    """
    raises = sorted(
        {
            at
            for row in rows
            if row.chip == chip and (at := _raise_cycle(row)) is not None
        }
    )
    diffs = [b - a for a, b in zip(raises, raises[1:])]
    return int(statistics.median(diffs)) if diffs else None


def _raise_frames(raises: Iterable[int], cpf: int) -> Dict[int, int]:
    """Each interrupt's frame: one per interrupt, and cadence-filled gaps between.

    A play call that changed no register leaves no row, so a gap longer than
    the cadence counts the calls it hides.
    """
    out: Dict[int, int] = {}
    frame, prev = 0, None
    for at in sorted(set(raises)):
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

    A play call's writes belong to the frame of the interrupt that made it
    (:func:`_raise_frames`, over every chip's interrupts); a write outside any
    interrupt belongs to the last one raised before it, or else to the baseline.
    """
    cpf = (
        cycles_per_frame
        or sidtrace_cadence(rows, chip=chip)
        or reg.PAL_CYCLES_PER_FRAME
    )
    raises = [_raise_cycle(row) for row in rows]
    frames = _raise_frames((at for at in raises if at is not None), cpf)
    starts = sorted(frames)

    def frame_of(row: SidtraceRow, at: Optional[int]) -> int:
        if at is not None:
            return frames[at]
        i = bisect.bisect_right(starts, row.cycle)
        return frames[starts[i - 1]] if i else -1

    framed = sorted(
        (
            (frame_of(row, at), row.reg, row.value)
            for row, at in zip(rows, raises)
            if row.chip == chip
        ),
        key=lambda w: w[0],
    )
    return _frame_grid(framed, reg_count, pw_hi_regs)


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

    Renders in a private temp dir beside ``out_path``, then moves the ``.csv.zst``
    in atomically. ``song`` (1-based) adds ``-o<song>``; ``user`` (``uid:gid``)
    keeps the trace non-root-owned. Raises :class:`SidtraceUnavailable` on failure.
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
