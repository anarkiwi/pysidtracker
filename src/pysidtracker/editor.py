"""Run a tracker editor's own code on its decrunched RAM as a fidelity oracle.

:func:`run_editor` boots 64 KiB of editor RAM on :func:`~pysidtracker.oracle.boot_flat`,
places a payload as the KERNAL ``LOAD`` leaves it and runs editor entry points;
:func:`editor_replay` wraps the editor's live player over the result as a PSID.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

from vagrant_moses import memmap as mm

from .emu import run_to_rts, set_iomap
from .header import write_psid
from .image import SidImage
from .oracle import SID_VOLUME, boot_flat

RAM_SIZE = 0x10000
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
    entries: Sequence[int],
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
        run_to_rts(mpu, mem, entry, 0, max_cycles)
    return subject


def editor_replay(
    ram: bytes,
    payload: bytes,
    at: int,
    entries: Sequence[int],
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
