"""Run a tracker editor's own code on its decrunched RAM as a fidelity oracle.

:func:`run_editor` boots 64 KiB of editor RAM on :func:`~pysidtracker.oracle.boot_flat`,
places a payload as the KERNAL ``LOAD`` leaves it and runs editor entry points;
:func:`editor_replay` wraps the editor's live player over the result as a PSID.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

from vagrant_moses import memmap as mm

from .emu import run_to_rts, run_to_rts_until, set_iomap
from .errors import SidError
from .header import write_psid
from .image import SidImage
from .oracle import SID_VOLUME, boot_flat

RAM_SIZE = 0x10000
Entry = Union[int, Tuple[int, Union[int, Iterable[int]]]]

LOAD_END = 0x00AE
FLAGS_6581 = 0x10


def ram_image(
    ram: bytes, init: int, play: Optional[int] = None, flags: int = FLAGS_6581
) -> SidImage:
    """The whole 64 KiB ``ram`` as a flat :class:`SidImage` loaded at ``$0000``.

    Its header carries ``init``, ``play`` (default ``init``) and ``flags``, so
    :func:`~pysidtracker.oracle.boot_flat` runs ``init`` over the RAM as it is.
    """
    stub = write_psid(
        load=0x1000,
        init=init,
        play=init if play is None else play,
        image=bytes(2),
        flags=flags,
    )
    mem = bytearray(ram)
    header = SidImage.from_bytes(stub).header
    return SidImage(mem, 0, RAM_SIZE, header, bytes(mem), b"")


def run_editor(
    ram: bytes,
    payload: bytes,
    at: int,
    entries: Sequence[Entry],
    *,
    load_end: Iterable[int] = (LOAD_END,),
    keep_bank: bool = True,
    keep_volume: bool = True,
    max_cycles: int = 30_000_000,
) -> bytearray:
    """The editor ``ram`` after loading ``payload`` at ``at`` and running ``entries``.

    Each ``load_end`` address gets the end word ``at + len(payload)``; ``keep_bank``
    and ``keep_volume`` restore the RAM's own ``$01`` and ``$D418`` over the PSID
    driver's. ``entries[0]`` runs as init, the rest in order with ``A = 0``.

    An entry is an address run to its ``rts``, or a ``(start, stop)`` pair run from
    ``start`` until the PC reaches ``stop`` (an address or a collection of them),
    then the stack pointer is put back as after a return. Raises
    :class:`~pysidtracker.errors.SidError` if ``max_cycles`` runs out or the
    routine returns before ``stop``.
    """
    if len(ram) != RAM_SIZE:
        raise ValueError(f"editor RAM must be {RAM_SIZE} bytes, got {len(ram)}")
    if not payload:
        raise ValueError("payload is empty")
    if not entries:
        raise ValueError("no entry points to run")
    bank, volume = ram[mm.CPU_PORT], ram[SID_VOLUME]
    end = ((at + len(payload)) & 0xFFFF).to_bytes(2, "little")

    def place(mpu, subject) -> None:
        if keep_bank:
            set_iomap(mpu, bank)
        if keep_volume:
            subject[SID_VOLUME] = volume
        subject[at : at + len(payload)] = payload
        for addr in load_end:
            subject[addr : addr + 2] = end

    subject, mpu, mem, _play = boot_flat(
        ram_image(ram, entries[0]), max_cycles=max_cycles, setup=place
    )
    if keep_bank:
        set_iomap(mpu, bank)
    for entry in entries[1:]:
        if isinstance(entry, int):
            run_to_rts(mpu, mem, entry, 0, max_cycles)
        else:
            _run_until(mpu, mem, entry, max_cycles)
    return subject


def _run_until(mpu, mem, entry, max_cycles: int) -> None:
    """Run a ``(start, stop)`` entry to ``stop`` and restore the stack pointer."""
    start, stop = entry
    stops = {stop} if isinstance(stop, int) else set(stop)
    sp = mpu.sp
    if not run_to_rts_until(mpu, mem, start, 0, stops, max_cycles):
        if mpu.sp < sp:
            raise SidError(f"${start:04X} never reached a stop in {max_cycles} cycles")
        raise SidError(f"${start:04X} returned before reaching a stop")
    mpu.sp = sp


def editor_replay(
    ram: bytes,
    payload: bytes,
    at: int,
    entries: Sequence[Entry],
    window: Tuple[int, int],
    init: int,
    play: int,
    *,
    flags: int = 0,
    **run_kwargs,
) -> bytes:
    """A PSID of ``[lo, hi)`` of :func:`run_editor`'s memory, loaded at ``lo``.

    ``init`` and ``play`` are the editor's live player, so an
    :class:`~pysidtracker.oracle.EmuPlayer` renders the song as the editor plays it.
    """
    lo, hi = window
    mem = run_editor(ram, payload, at, entries, **run_kwargs)
    return write_psid(
        load=lo, init=init, play=play, image=bytes(mem[lo:hi]), flags=flags
    )


__all__ = ["LOAD_END", "RAM_SIZE", "editor_replay", "ram_image", "run_editor"]
