"""Detect compressed / packed / relocating playroutines in a first parse.

A ``.sid`` header advertises a load address, an init address and a subtune
count, but for many real tunes these describe a *loader* or *packer*, not the
playroutine: the actual song tables only exist after the init routine unpacks or
relocates them into memory. The header is therefore untrustworthy for locating
data.

The shared strategy (generalised from pygoattracker's ``.sid`` decompiler):

1. **Static recognition.** Ask the format's ``recognize`` callback to find its
   signature/anchor in the freshly loaded image. If it succeeds the tune is
   :attr:`PlayroutineKind.DIRECT` -- a plain direct-load image.
2. **Emulated init.** Otherwise the playroutine is packed/relocating: run its
   init routine in a 6502 emulator so the data lands where it really goes, then
   recognise again. Whether init had to *expand* memory (decompression) or only
   *move*/rewrite the existing image classifies it as
   :attr:`PlayroutineKind.PACKED` vs :attr:`PlayroutineKind.RELOCATED`.
3. If recognition still fails, :attr:`PlayroutineKind.UNKNOWN`.

Each parser supplies only the cheap ``recognize`` predicate; this module owns
the untrustworthy-header handling so every format gets it identically.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from vagrant_moses import memmap as mm

from .emu import observe_data_access, run_to_rts, wire_mpu
from .errors import SidParseError
from .poweron import psid_init_bank
from .registers import sid_model_for_flags
from .header import SidHeader
from .image import SidImage

JMP_ABS = 0x4C

# recognize(image) -> anchor (any truthy value) if the format is found, else None.
Recognizer = Callable[[SidImage], object]


class PlayroutineKind(enum.Enum):
    """How a tune's playroutine presents its data in the loaded image."""

    DIRECT = "direct"  # data found statically, as loaded
    RELOCATED = "relocated"  # found only after init moved/rewrote the image
    PACKED = "packed"  # found only after init expanded (decompressed) memory
    UNKNOWN = "unknown"  # not recognised, even after running init


@dataclass
class Detection:
    """Outcome of :func:`detect_playroutine`.

    Attributes:
      kind: the :class:`PlayroutineKind`.
      ran_init: whether the init routine had to be emulated.
      anchor: whatever the ``recognize`` callback returned (``None`` if
        unrecognised) -- often the address/offset it found, reusable by the
        caller's full parse.
      changed_inside: bytes within the as-loaded region that init rewrote
        (self-modifying / in-place unpacking).
      written_outside: bytes outside the as-loaded region that init wrote
        (relocation targets / decompression output).
    """

    kind: PlayroutineKind
    ran_init: bool
    anchor: object = None
    changed_inside: int = 0
    written_outside: int = 0

    @property
    def recognised(self) -> bool:
        return self.kind is not PlayroutineKind.UNKNOWN

    @property
    def trustworthy_header(self) -> bool:
        """True when the header described the data directly (no init needed)."""
        return self.kind is PlayroutineKind.DIRECT


def detect_playroutine(
    image: SidImage,
    recognize: Recognizer,
    *,
    init: bool = True,
    subtune: int = 0,
    native: bool = False,
) -> Detection:
    """Classify how ``image`` presents its playroutine data.

    ``recognize`` inspects the (possibly unpacked) image and returns a truthy
    anchor when it finds the format, else ``None``. When static recognition
    fails and ``init`` is true, the init routine is emulated (requires the core
    ``jennings`` dependency) and recognition retried.

    When ``native`` is true, an exomizer-packed image is first decrunched
    natively (:func:`pysidtracker.decrunch.native_decrunch`, no init emulated);
    if that unpacks a recognisable image it is reported as
    :attr:`PlayroutineKind.PACKED` with ``ran_init=False``. Native decrunch is
    opt-in and only a first try -- it falls back to the emulated-init path
    below when the image is not exomizer-packed or stays unrecognised, so the
    default behaviour is unchanged.

    Raises :class:`EmulatorUnavailable` if emulation is needed but ``jennings`` is
    not installed.
    """
    from .locate import AFTER_INIT, NATIVE, STATIC, Strategy, locate

    def anchors(staged: SidImage) -> List[object]:
        return [recognize(staged)]

    def decode(_staged: SidImage, anchor: object) -> object:
        return anchor or None

    strategies = [Strategy("recognize", STATIC, decode, anchors)]
    if native:
        strategies.append(Strategy("native", NATIVE, decode, anchors))
    strategies.append(Strategy("init", AFTER_INIT, decode, anchors))
    located = locate(
        image,
        strategies,
        init=init,
        subtune=subtune,
        in_place=True,
        strict=True,
    )
    return located.detection


# The 6502 stack page is scratch space the init run (and our return-address
# stub) churns; it is never song data, so it is excluded from the diff metrics.
_STACK_LO = 0x0100
_STACK_HI = 0x0200


def _count_changed(mem: bytearray, load: int, snapshot: bytes) -> int:
    return sum(1 for i, b in enumerate(snapshot) if mem[load + i] != b)


def _environment(addr: int) -> bool:
    """A C64 bank register (``$00``/``$01``) or the I/O hardware page: registers
    the driver and a tune's init legitimately write, never song data -- so they
    are excluded from "bytes written outside the loaded region"."""
    return addr <= 0x01 or 0xD000 <= addr <= 0xDFFF


def _count_written_outside(mem: bytearray, load: int, end: int) -> int:
    written = 0
    for addr, value in enumerate(mem):
        if not value:
            continue
        if load <= addr < end:
            continue
        if _STACK_LO <= addr < _STACK_HI:
            continue
        if _environment(addr):
            continue
        written += 1
    return written


def resolve_entry_points(
    header: Optional[SidHeader],
    load: int,
    default_init: Optional[int] = None,
    default_play: Optional[int] = None,
) -> Tuple[int, int]:
    """Resolve a tune's ``(init, play)`` addresses with the SID zero fallback.

    When ``header`` is present, an ``init``/``play`` field of ``0`` means "use
    the load address" (the documented SID convention). For a bare ``.prg``
    (``header is None``) the ``default_init``/``default_play`` are used, each
    falling back to ``load`` when not given.
    """
    if header is not None:
        init = header.init_address or load
        play = header.play_address or load
    else:
        init = load if default_init is None else default_init
        play = load if default_play is None else default_play
    return init, play


def is_jmp_vector(image: SidImage, addr: int) -> bool:
    """True if ``addr`` holds a ``JMP abs`` ($4C) whose target is in the image."""
    if not image.contains(addr) or image.peek(addr) != JMP_ABS:
        return False
    target = image.peek(addr + 1) | (image.peek(addr + 2) << 8)
    return image.contains(target)


def run_init(image: SidImage, subtune: int = 0, max_cycles: int = 8_000_000) -> None:
    """Run the tune's init routine in a 6502 emulator so data lands in place.

    Mutates ``image.mem`` directly. ``subtune`` is passed to init in the
    accumulator (the SID calling convention). Requires the optional ``jennings``
    dependency; raises :class:`EmulatorUnavailable` if it is missing and
    :class:`SidParseError` if the image has no init address to call.
    """
    if image.header is None:
        raise SidParseError("cannot run init: image has no SID header")
    init_address = image.header.init_address or image.header.real_load_address
    # The driver banks the init's ROM space before calling it (KERNAL/BASIC in).
    image.mem[1] = psid_init_bank(init_address)
    mpu, mem = wire_mpu(image.mem, sid_model=sid_model_for_flags(image.header.flags))
    run_to_rts(mpu, mem, init_address, subtune, max_cycles)


# init effects: what one song's init leaves behind (song -> subtune resolution).

# Hardware registers, not song data: excluded from recorded cell traffic.
_IO_PAGE_LO = 0xD000
_IO_PAGE_HI = 0xDFFF

# The play/IRQ vectors and timer latches a tune installs (as trace_init reports).
_JUMP_CELLS = frozenset(
    (
        mm.RAM_IRQ_VECTOR_LO,
        mm.RAM_IRQ_VECTOR_HI,
        mm.CPU_IRQ_VECTOR,
        mm.CPU_IRQ_VECTOR + 1,
        mm.CIA1_TIMER_A_LO,
        mm.CIA1_TIMER_A_HI,
        mm.CIA2_TIMER_A_LO,
        mm.CIA2_TIMER_A_HI,
    )
)


@dataclass(frozen=True)
class CellWrite:
    """The final store init made to one RAM cell: the value left and the PC of
    the store that left it."""

    value: int
    pc: int


@dataclass(frozen=True)
class CellRead:
    """One read of a RAM cell during init: the value returned and the PC of the
    load (which table the song number indexed)."""

    value: int
    pc: int


@dataclass
class Jumps:
    """The play/IRQ vectors and timer latches a tune's init installed.

    A vector is ``None`` when neither of its two bytes was written, exactly as
    in :class:`~pysidtracker.trace.InitTrace`.
    """

    irq_vector: Optional[int] = None
    hw_irq_vector: Optional[int] = None
    cia1_timer_latch: Optional[int] = None
    cia2_timer_latch: Optional[int] = None


@dataclass
class InitEffects:
    """What one song's init left in RAM.

    Attributes:
      song: the song number that was passed in the accumulator.
      writes: the final store per cell (value, store PC); stack and I/O pages
        excluded, or only the ``watch``ed cells when given.
      reads: each cell read in execution order (value, load PC), same scoping.
      jumps: the installed play/IRQ vectors and CIA latches (always complete).
    """

    song: int
    writes: Dict[int, CellWrite]
    reads: Dict[int, List[CellRead]]
    jumps: Jumps


def _cell_watched(address: int, watch: Optional[Set[int]]) -> bool:
    if watch is not None:
        return address in watch
    if _STACK_LO <= address < _STACK_HI:
        return False
    return not _IO_PAGE_LO <= address <= _IO_PAGE_HI


def _vector(writes: Dict[int, int], lo: int, hi: int) -> Optional[int]:
    if lo not in writes and hi not in writes:
        return None
    return writes.get(lo, 0) | (writes.get(hi, 0) << 8)


def init_effects(
    image: SidImage,
    song: int,
    *,
    watch: Optional[Iterable[int]] = None,
    max_cycles: int = 8_000_000,
) -> InitEffects:
    """Run the tune's init once with ``A = song`` and record what it left in RAM.

    Records the data stores init makes to RAM cells (final value per cell and
    the store's PC), the data reads (value and the load's PC, in execution
    order) -- instruction fetches and stack traffic stay out -- and the
    installed play/IRQ vectors and CIA latches (``jumps``), like
    :func:`trace_init`. The run happens on a private snapshot, so the caller's
    ``image`` is never mutated; ``watch`` limits ``writes``/``reads`` to the
    given cells. ``song`` goes to init in the accumulator (the SID calling
    convention) and is not range-checked here.

    Raises:
      SidParseError: the image has no SID header.
    """
    if image.header is None:
        raise SidParseError("cannot observe init effects: image has no SID header")
    init_address = image.header.init_address or image.header.real_load_address
    if watch is not None:
        watch = set(watch)
    # private snapshot: the image stays pristine and every song starts from the same machine
    mem = bytearray(image.mem)
    # The driver banks the init's ROM space before calling it (KERNAL/BASIC in).
    mem[1] = psid_init_bank(init_address)
    mpu, obsmem = wire_mpu(mem, sid_model=sid_model_for_flags(image.header.flags))

    writes: Dict[int, CellWrite] = {}
    reads: Dict[int, List[CellRead]] = {}
    vectors: Dict[int, int] = {}

    def _on_write(address: int, value: int) -> None:
        if _cell_watched(address, watch):
            writes[address] = CellWrite(value & 0xFF, mpu.pc)
        if address in _JUMP_CELLS:
            vectors[address] = value & 0xFF

    def _on_read(address: int) -> None:
        if _cell_watched(address, watch):
            reads.setdefault(address, []).append(CellRead(mem[address], mpu.pc))

    observe_data_access(mpu, on_read=_on_read, on_write=_on_write)
    run_to_rts(mpu, obsmem, init_address, song, max_cycles)

    return InitEffects(
        song=song & 0xFF,
        writes={addr: writes[addr] for addr in sorted(writes)},
        reads={addr: reads[addr] for addr in sorted(reads)},
        jumps=Jumps(
            irq_vector=_vector(vectors, mm.RAM_IRQ_VECTOR_LO, mm.RAM_IRQ_VECTOR_HI),
            hw_irq_vector=_vector(vectors, mm.CPU_IRQ_VECTOR, mm.CPU_IRQ_VECTOR + 1),
            cia1_timer_latch=_vector(vectors, mm.CIA1_TIMER_A_LO, mm.CIA1_TIMER_A_HI),
            cia2_timer_latch=_vector(vectors, mm.CIA2_TIMER_A_LO, mm.CIA2_TIMER_A_HI),
        ),
    )


def subtune_selection(
    image: SidImage, songs: Iterable[int], cell: int
) -> Dict[int, int]:
    """The value each header song leaves in a player cell after its own init.

    Returns ``{song: value}`` from one :func:`init_effects` run per song, each
    from a fresh machine; a cell init never writes to keeps its pre-init value.
    """
    if image.header is None:
        raise SidParseError("cannot map subtunes: image has no SID header")
    cell &= 0xFFFF
    song = list(songs)
    count = image.header.songs
    for s in song:
        if not 1 <= s <= count:
            raise SidParseError(f"song {s} out of range (header: {count} songs)")

    selection: Dict[int, int] = {}
    for s in song:
        effects = init_effects(image, s, watch=[cell])
        selection[s] = (
            effects.writes[cell].value if cell in effects.writes else image.mem[cell]
        )

    by_value: Dict[int, int] = {}
    for s, value in selection.items():
        alias = by_value.get(value)
        if alias is not None and alias != s:
            raise SidParseError(
                f"songs {alias} and {s} alias cell {cell:#06x} (both select {value:#03x})"
            )
        by_value[value] = s
    return selection


def init_diff(image: SidImage, songs: Iterable[int]) -> Set[int]:
    """The cells whose final value differs between the inits of ``songs``.

    One whole-address-space :func:`init_effects` run per song (fresh machine
    each); a cell no song wrote is compared at its pre-init value. This finds
    the selection cell a library does not know and exposes wrappers that copy
    a different module per song.
    """
    finals: List[Dict[int, CellWrite]] = []
    cells: Set[int] = set()
    for s in songs:
        effects = init_effects(image, s)
        finals.append(effects.writes)
        cells.update(effects.writes)

    def final(writes: Dict[int, CellWrite], addr: int) -> int:
        write = writes.get(addr)
        return write.value if write is not None else image.mem[addr]

    return {
        addr for addr in cells if len({final(writes, addr) for writes in finals}) > 1
    }
