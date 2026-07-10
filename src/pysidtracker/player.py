"""A per-frame SID player scaffold shared by the ``py*`` playroutines.

Every faithful playroutine transcription (pyjch, pydmcsid, pysidwizard, ...)
mounts its image into a flat 64 KiB memory, runs ``init`` once, then emits per
frame the SID registers that changed. :class:`MemPlayer` owns that machinery --
memory mount, ``_rd``/``_wr``, the post-init snapshot, the diffing
``play_frame``, and the ``iter_frames``/``render_grid`` drivers -- so a concrete
player implements only ``_init`` and ``_frame``.
"""

from __future__ import annotations

import abc
from typing import Iterator, List, Optional, Tuple

from .registers import SID_BASE, SID_REG_COUNT

MEM_SIZE = 0x10000


class MemPlayer(abc.ABC):
    """Abstract per-frame player over a flat 64 KiB memory image.

    Subclasses implement :meth:`_init` (run the tune's init, programming the
    initial SID register file) and :meth:`_frame` (advance one play call). Both
    use :meth:`_rd`/:meth:`_wr`; SID writes land in ``$D400..`` and are read
    back by :meth:`snapshot`.
    """

    SID_BASE = SID_BASE
    REG_COUNT = SID_REG_COUNT

    def __init__(self, image: bytes, load: int, subtune: int = 0):
        self._mem = bytearray(MEM_SIZE)
        self._load = load & 0xFFFF
        self._mem[self._load : self._load + len(image)] = image
        self.regs: List[int] = [0] * self.REG_COUNT
        self._last_regs: Optional[List[int]] = None
        self._init(subtune)
        self.regs = self.snapshot()

    def _rd(self, addr: int) -> int:
        """Read a byte (16-bit address wrap)."""
        return self._mem[addr & 0xFFFF]

    def _wr(self, addr: int, val: int) -> None:
        """Write a byte (16-bit address wrap, value masked to 8 bits)."""
        self._mem[addr & 0xFFFF] = val & 0xFF

    @abc.abstractmethod
    def _init(self, subtune: int) -> None:
        """Run the tune's init routine (program the initial SID register file)."""

    @abc.abstractmethod
    def _frame(self) -> None:
        """Advance one play call, writing this frame's SID registers to memory."""

    def snapshot(self) -> List[int]:
        """The current SID register file (``$D400..$D400+REG_COUNT``)."""
        base = self.SID_BASE
        return [self._mem[base + i] for i in range(self.REG_COUNT)]

    def play_frame(self) -> List[Tuple[int, int]]:
        """Run one frame; return its ``(reg, val)`` writes.

        The first frame returns all :data:`REG_COUNT` registers; later frames
        return only registers whose value changed.
        """
        self._frame()
        self.regs = self.snapshot()
        if self._last_regs is None:
            writes = list(enumerate(self.regs))
        else:
            writes = [
                (reg, value)
                for reg, (value, last) in enumerate(zip(self.regs, self._last_regs))
                if value != last
            ]
        self._last_regs = list(self.regs)
        return writes

    def iter_frames(
        self, max_frames: Optional[int] = None
    ) -> Iterator[List[Tuple[int, int]]]:
        """Yield per-frame ``(reg, val)`` write lists (forever if unbounded)."""
        frame = 0
        while max_frames is None or frame < max_frames:
            yield self.play_frame()
            frame += 1

    def render_grid(self, nframes: int) -> List[List[int]]:
        """Render ``nframes`` of forward-filled per-frame register snapshots."""
        rows: List[List[int]] = []
        for _ in range(nframes):
            self.play_frame()
            rows.append(self.regs[:])
        return rows
