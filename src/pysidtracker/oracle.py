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

import csv
import io
import os
import shutil
import statistics
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple

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
    each play. ``illegal_opcodes=True`` installs the NMOS illegal opcodes that
    replays such as defMON need (default off, so other callers are unaffected).

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
    mpu, mem = wire_mpu(subject, illegal_opcodes)

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
    ):
        self._init_addr = init & 0xFFFF
        self._play_addr = play & 0xFFFF
        self._illegal = illegal_opcodes
        self._max_cycles = max_cycles
        self._mpu = None
        self._obs = None
        super().__init__(image, load, subtune)

    def _init(self, subtune: int) -> None:
        self._mpu, self._obs = wire_mpu(self._mem, self._illegal)
        run_to_rts(self._mpu, self._obs, self._init_addr, subtune, self._max_cycles)

    def _frame(self) -> None:
        run_to_rts(self._mpu, self._obs, self._play_addr, 0, self._max_cycles)

    def snapshot(self) -> List[int]:
        base = self.SID_BASE
        return [
            (self._mem[base + i] & 0x0F) if i in reg.PW_HI_REGS else self._mem[base + i]
            for i in range(self.REG_COUNT)
        ]


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
    ``> gap``-cycle gap; the leading init writes form frame 0's baseline. Each
    frame's registers forward-fill from the previous frame, and the
    pulse-width-high registers (``pw_hi_regs``) are masked to 4 bits. Frame
    assignment rounds to nearest (``(clock - t0 + cpf // 2) // cpf``).
    """
    if not writes:
        return []
    pw = set(pw_hi_regs)
    cyc = [w[0] for w in writes]
    t0 = cyc[0]
    for prev, cur in zip(cyc, cyc[1:]):
        if cur - prev > gap:
            t0 = cur
            break
    cur_row = [0] * reg_count
    rows: List[List[int]] = []
    idx = 0
    while idx < len(writes) and writes[idx][0] < t0:
        _c, register, val = writes[idx]
        if 0 <= register < reg_count:
            cur_row[register] = (val & 0x0F) if register in pw else val
        idx += 1

    def frame_of(clock: int) -> int:
        return (clock - t0 + cycles_per_frame // 2) // cycles_per_frame

    nframes = frame_of(writes[-1][0]) + 1
    for frame in range(nframes):
        while idx < len(writes) and frame_of(writes[idx][0]) == frame:
            _c, register, val = writes[idx]
            if 0 <= register < reg_count:
                cur_row[register] = (val & 0x0F) if register in pw else val
            idx += 1
        rows.append(cur_row[:])
    return rows


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
            row.cycle
            - (
                row.since_video_irq
                if row.since_video_irq is not None
                else row.since_cia_irq
            )
            for row in rows
            if row.chip == chip
            and (row.since_video_irq is not None or row.since_cia_irq is not None)
        }
    )
    diffs = [b - a for a, b in zip(raises, raises[1:])]
    return int(statistics.median(diffs)) if diffs else None


def sidtrace_grid(
    rows: Sequence[SidtraceRow],
    *,
    chip: int = 0,
    reg_count: int = 25,
    cycles_per_frame: Optional[int] = None,
    pw_hi_regs: Iterable[int] = reg.PW_HI_REGS,
    gap: int = 10000,
) -> List[List[int]]:
    """Frame a sidtrace CSV into a per-frame register grid for ``chip``.

    Reuses :func:`grid_from_writes` with the cadence from
    :func:`sidtrace_cadence` (falling back to the PAL frame length), so the
    oracle grid is directly comparable to :func:`register_grid` /
    :meth:`~pysidtracker.player.MemPlayer.render_grid` via :func:`aligned_match`.
    """
    cpf = (
        cycles_per_frame
        or sidtrace_cadence(rows, chip=chip)
        or reg.PAL_CYCLES_PER_FRAME
    )
    writes = [
        (row.cycle, row.reg, row.value)
        for row in rows
        if row.chip == chip and 0 <= row.reg < reg_count
    ]
    return grid_from_writes(
        writes,
        cycles_per_frame=cpf,
        reg_count=reg_count,
        pw_hi_regs=pw_hi_regs,
        gap=gap,
    )


def run_sidtrace(
    tune_path,
    out_path,
    *,
    seconds: int = 60,
    image: str = SIDTRACE_IMAGE,
    docker: str = "docker",
    extra_args: Sequence[str] = (),
) -> Path:
    """Render ``tune_path`` under the sidtrace Docker oracle to ``out_path``.

    The container reads and writes relative to ``/work``, so the render runs in a
    private temporary directory (bind-mounted at ``/work``) and the finished
    trace is moved into ``out_path`` atomically. This keeps concurrent renders --
    e.g. pytest-xdist workers sharing one cache directory -- from colliding on
    the mount directory, on same-named tunes, or on a half-written cache file.

    Returns ``out_path`` (a ``.csv.zst``). Raises :class:`SidtraceUnavailable` if
    Docker is missing or the render fails -- this never silently skips.
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
        cmd = [
            docker,
            "run",
            "--rm",
            "-v",
            f"{work.resolve()}:/work",
            image,
            result.name,
            tune_local.name,
            f"-t{seconds}",
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
