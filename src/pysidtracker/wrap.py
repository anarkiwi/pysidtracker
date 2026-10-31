"""A tune's own play routine around a native player tick.

The image's play code runs in the jennings emulator; each time it reaches one of
the player's entries a Python ``tick`` stands in for the player, its register
blocks are written to emulated memory, and the routine resumes as from ``rts``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Sequence

import numpy as np

from .emu import PLAY_CYCLES, SidWriteLog, call_play, run_until
from .image import SidImage
from .mos6502 import OP_JMP, OP_JSR
from .oracle import boot_flat
from .player import MemPlayer

BOOT_CYCLES = 8_000_000
STACK = 0x0100
_SPIN_STEPS = 1024
_SPIN_PCS = 8

Tick = Callable[[], Mapping[int, Sequence[int]]]


@dataclass(frozen=True)
class Wrap:
    """The play routine at ``play`` around the player's ``entries``, of the image
    booted for header ``song``; ``sink`` is the ghost block a tick's registers go
    to for the routine to flush (None: the SID)."""

    play: int
    entries: frozenset
    song: int = 0
    sink: Optional[int] = None


def enter(mpu, mem, pc: int, acc: int = 0) -> int:
    """Enter ``pc`` with ``A = acc`` as a ``jsr`` from outside: the stack depth
    its final ``rts`` returns to."""
    depth = mpu.sp
    for byte in (0x00, 0x01):
        mem[STACK + mpu.sp] = byte
        mpu.sp = (mpu.sp - 1) & 0xFF
    mpu.a, mpu.pc = acc & 0xFF, pc
    return depth


def leave(mpu, mem) -> None:
    """Return from the routine being run as its ``rts`` would."""
    lo, hi = (mem[STACK + ((mpu.sp + k) & 0xFF)] for k in (1, 2))
    mpu.sp = (mpu.sp + 2) & 0xFF
    mpu.pc = ((hi << 8 | lo) + 1) & 0xFFFF


def _running(mpu, depth: int, start: int, max_cycles: int) -> bool:
    return mpu.sp < depth and mpu.processorCycles - start <= max_cycles


def idle_init(mpu, mem, address: int, song: int, max_cycles: int) -> None:
    """Run an init that may never return (an IRQ-driven player's) until it does,
    spins in an idle loop of at most a few addresses, or ``max_cycles`` pass."""
    depth, start = enter(mpu, mem, address, song), mpu.processorCycles
    pcs, steps = set(), 0
    while _running(mpu, depth, start, max_cycles):
        mpu.step()
        pcs.add(mpu.pc)
        steps += 1
        if steps == _SPIN_STEPS:
            if len(pcs) <= _SPIN_PCS:
                return
            pcs, steps = set(), 0


def boot(image: SidImage, song: int = 0, max_cycles: int = BOOT_CYCLES):
    """``(subject, mpu, mem, play)``: :func:`~pysidtracker.oracle.boot_flat`, with
    :func:`idle_init` running the init of a tune without a play address."""
    run = None if image.header.play_address else idle_init
    return boot_flat(image, song, max_cycles, init_run=run)


def measure_wrap(
    image: SidImage,
    song: int,
    play: Optional[int],
    entries,
    max_cycles: int = PLAY_CYCLES,
) -> Optional[Wrap]:
    """The :class:`Wrap` of a ``play`` routine that is not one of ``entries``
    (None: it is, or none is known). Where the second play call's SID writes all
    fall outside the entered player, ``sink`` is the block most of them copy."""
    if play is None or not entries or play in entries:
        return None
    entries = frozenset(entries)
    _, mpu, mem, _ = boot(image, song)
    log = SidWriteLog(mpu, mirrors=False)
    flags: List[bool] = []
    state = {"entered": None}
    mem.subscribe_to_write(
        list(log.registers), lambda *_: flags.append(state["entered"] is not None)
    )
    for _ in range(2):
        writes, flags[:] = log.start(), []
        depth, start = enter(mpu, mem, play), mpu.processorCycles
        state["entered"] = None
        while _running(mpu, depth, start, max_cycles):
            entered = state["entered"]
            if entered is None and mpu.pc in entries:
                state["entered"] = mpu.sp
            elif entered is not None and mpu.sp > entered:
                state["entered"] = None
            else:
                left = max_cycles - (mpu.processorCycles - start)
                stops, above = (
                    (entries, depth - 1) if entered is None else ((), entered)
                )
                run_until(mpu, stops, above, left)
                continue
            mpu.step()
    inner = flags
    if any(inner):
        return Wrap(play, entries, song)
    bases = Counter(read - reg for reg, read, _, _ in writes if read is not None)
    sink = bases.most_common(1)[0][0] if bases else None
    return Wrap(play, entries, song, sink)


def count_entries(
    image: SidImage,
    song: int,
    play: int,
    entries,
    max_cycles: int = PLAY_CYCLES,
) -> int:
    """How many times the play call after the first enters the target of its
    first ``jsr``/``jmp``, where that target is one of ``entries`` (else 1)."""
    ram, mpu, mem, _ = boot(image, song)
    call_play(mpu, mem, play, max_cycles)
    depth, start = enter(mpu, mem, play, mpu.a), mpu.processorCycles
    jumps = frozenset(
        np.flatnonzero(np.isin(np.frombuffer(ram, np.uint8), (OP_JSR, OP_JMP))).tolist()
    )
    while _running(mpu, depth, start, max_cycles):
        pc = mpu.pc
        if pc in jumps:
            break
        left = max_cycles - (mpu.processorCycles - start)
        run_until(mpu, jumps, depth - 1, left)
    else:
        return 1
    target = ram[pc + 1] | ram[pc + 2] << 8
    count = 0
    mpu.step()
    while _running(mpu, depth, start, max_cycles):
        left = max_cycles - (mpu.processorCycles - start)
        if run_until(mpu, (target,), depth - 1, left) == "pc":
            count += 1
            mpu.step()
        elif mpu.pc != target:
            break
    return max(count, 1) if target in entries else 1


class Wrapped(MemPlayer):
    """``tick`` inside the image's own play routine ``wrap``: each frame runs the
    routine, and each entry into the player writes ``tick()``'s ``{address:
    values}`` blocks to emulated memory and returns as by ``rts``."""

    MASK_PW_HI = True

    def __init__(
        self, tick: Tick, image: SidImage, wrap: Wrap, max_cycles: int = PLAY_CYCLES
    ):
        self._tick, self._image, self._wrap = tick, image, wrap
        self.max_cycles = max_cycles
        super().__init__(b"", 0, wrap.song)

    def _init(self, subtune: int) -> None:
        self._mem, self._mpu, self._bus, _ = boot(self._image, subtune)

    def _frame(self) -> None:
        mpu, bus, wrap = self._mpu, self._bus, self._wrap
        depth, start = enter(mpu, bus, wrap.play), mpu.processorCycles
        while _running(mpu, depth, start, self.max_cycles):
            if mpu.pc not in wrap.entries:
                left = self.max_cycles - (mpu.processorCycles - start)
                run_until(mpu, wrap.entries, depth - 1, left)
                continue
            for addr, values in self._tick().items():
                for offset, value in enumerate(values):
                    bus[(addr + offset) & 0xFFFF] = value & 0xFF
            leave(mpu, bus)


__all__ = [
    "BOOT_CYCLES",
    "Wrap",
    "Wrapped",
    "boot",
    "count_entries",
    "enter",
    "idle_init",
    "leave",
    "measure_wrap",
]
