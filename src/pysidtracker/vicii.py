"""The VIC-II as libsidplayfp's MOS656X models it: raster IRQ, badline and sprite BA.

A port of ``mos656x.cpp`` and ``sprites.h`` onto :mod:`~pysidtracker.eventsched`,
so raster interrupts and the cycles the VIC steals from the CPU land on the same
cycles as in the sidtrace oracle. The light pen is not modelled.
"""

from __future__ import annotations

from typing import Callable

from .eventsched import PHI1, Event, Scheduler

FIRST_DMA_LINE, LAST_DMA_LINE = 0x30, 0xF7
_FETCH_CYCLE, _TEXT_COLS = 11, 40
_IRQ_RASTER = 1
PAL_GEOMETRY, NTSC_GEOMETRY = (312, 63), (263, 65)


class Sprites:
    """Sprite DMA state (``sprites.h``): which sprites fetch data this line."""

    __slots__ = ("regs", "exp_flop", "dma", "mc_base", "mc")

    def __init__(self, regs: bytearray):
        self.regs = regs
        self.reset()

    def reset(self) -> None:
        """Power-on state: no DMA, expansion flip-flops set."""
        self.exp_flop, self.dma = 0xFF, 0
        self.mc_base, self.mc = [0] * 8, [0] * 8

    def update_mc(self) -> None:
        """Advance the data counter of every sprite doing DMA."""
        for i in range(8):
            if self.dma >> i & 1:
                self.mc[i] = (self.mc[i] + 3) & 0x3F

    def update_mc_base(self) -> None:
        """Latch the data counter; a sprite whose data ran out stops its DMA."""
        for i in range(8):
            if self.exp_flop >> i & 1:
                self.mc_base[i] = self.mc[i]
                if self.mc_base[i] == 0x3F:
                    self.dma &= ~(1 << i)

    def check_exp(self) -> None:
        """Toggle the Y-expansion flip-flops of expanded sprites doing DMA."""
        self.exp_flop ^= self.dma & self.regs[0x17]

    def check_display(self) -> None:
        """Reload the data counters from their bases."""
        self.mc = list(self.mc_base)

    def check_dma(self, raster_y: int) -> None:
        """Start DMA for enabled sprites whose Y coordinate matches the raster."""
        regs, y = self.regs, raster_y & 0xFF
        for i in range(8):
            mask = 1 << i
            if regs[0x15] & mask and y == regs[2 * i + 1] and not self.dma & mask:
                self.dma |= mask
                self.mc_base[i] = 0
                self.exp_flop |= mask

    def line_crunch(self, data: int, line_cycle: int) -> None:
        """A ``$D017`` write clearing expansion mid-sprite (sprite crunch)."""
        for i in range(8):
            mask = 1 << i
            if not data & mask and not self.exp_flop & mask:
                if line_cycle == 14:
                    mc, base = self.mc[i], self.mc_base[i]
                    self.mc[i] = (0x2A & (base & mc)) | (0x15 & (base | mc))
                self.exp_flop |= mask


class Vic:
    """MOS6569 (PAL) or MOS6567R8 (NTSC) raster, interrupt and BA timing."""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        sched: Scheduler,
        ntsc: bool,
        irq: Callable[[bool], None],
        set_ba: Callable[[bool], None],
    ):
        self.sched = sched
        self.max_rasters, self.cycles_per_line = NTSC_GEOMETRY if ntsc else PAL_GEOMETRY
        self.clock = self._clock_ntsc if ntsc else self._clock_pal
        self.irq, self.set_ba = irq, set_ba
        self.regs = bytearray(0x40)
        self.sprites = Sprites(self.regs)
        self.event = Event(self._event)
        self.badline_event = Event(self._badline_change)
        self.edge_event = Event(self._edge)
        self.reset()

    def reset(self) -> None:
        """Power-on state, the beam on the last line, its first event due now."""
        self.irq_flags = self.irq_mask = self.yscroll = 0
        self.raster_y = self.max_rasters - 1
        self.line_cycle = self.raster_clk = 0
        self.bad_lines_enabled = self.bad_line = self.raster_cond = False
        self.vblanking = False
        self.regs[:] = bytes(0x40)
        self.sprites.reset()
        self.sched.cancel(self.event)
        self.sched.schedule(self.event, 0, PHI1)

    def _compare(self) -> int:
        return self.regs[0x12] | (self.regs[0x11] & 0x80) << 1

    def _edge(self) -> None:
        old, self.raster_cond = self.raster_cond, self.raster_y == self._compare()
        if self.raster_cond and not old:
            self.irq_flags |= _IRQ_RASTER
            self._irq_state()

    def _badline_change(self) -> None:
        self.set_ba(not self.bad_line)

    def _irq_state(self) -> None:
        if self.irq_flags & self.irq_mask & 0x0F:
            if not self.irq_flags & 0x80:
                self.irq(True)
                self.irq_flags |= 0x80
        elif self.irq_flags & 0x80:
            self.irq(False)
            self.irq_flags &= 0x7F

    def _den(self) -> bool:
        return bool(self.regs[0x11] & 0x10)

    def _is_bad_line(self) -> bool:
        y = self.raster_y
        return (
            self.bad_lines_enabled
            and FIRST_DMA_LINE <= y <= LAST_DMA_LINE
            and (y & 7) == self.yscroll
        )

    def _check_vblank(self) -> None:
        if self.raster_y == self.max_rasters - 1:
            self.vblanking = True
        if (
            self.raster_y == FIRST_DMA_LINE
            and not self.bad_lines_enabled
            and self._den()
        ):
            self.bad_lines_enabled = True
        if self.raster_y == LAST_DMA_LINE:
            self.bad_lines_enabled = False
        self.bad_line = False
        if not self.vblanking:
            self.raster_y += 1
            self._edge()
            if self.raster_y == FIRST_DMA_LINE and not self.bad_lines_enabled:
                self.bad_lines_enabled = self._den()
        if self._is_bad_line():
            self.bad_line = True

    def _vblank(self) -> None:
        if self.vblanking:
            self.vblanking = False
            self.raster_y = 0
            self._edge()

    def _start_dma(self, n: int) -> None:
        if n == 0:
            self.set_ba(not self.sprites.dma & 1)
        elif self.sprites.dma >> n & 1:
            self.set_ba(False)

    def _end_dma(self, n: int) -> None:
        if n == 7 or not self.sprites.dma & (6 << n):
            self.set_ba(True)

    def _event(self) -> None:
        sched = self.sched
        phase = sched.now & 1
        cycles = sched.time(phase) - self.raster_clk
        if cycles:
            self.raster_clk += cycles
            self.line_cycle = (self.line_cycle + cycles) % self.cycles_per_line
            delay = self.clock()
        else:
            delay = 1
        sched.schedule(self.event, delay - phase, PHI1)

    def sync(self) -> None:
        """Catch the beam up to now (a CPU access)."""
        self.sched.cancel(self.event)
        self._event()

    def _clock_common(self, lc: int):
        """Line cycles 11-15, shared by PAL and NTSC; ``None`` for others."""
        if lc == 11:
            if self.bad_line:
                self.set_ba(False)
            return 3
        if lc == 12:
            return 2
        if lc == 13:
            return 1
        if lc == 14:
            self.sprites.update_mc()
            return 1
        if lc == 15:
            self.sprites.update_mc_base()
            return 39
        return None

    def _clock_pal(self) -> int:
        # pylint: disable=too-many-branches,too-many-return-statements
        lc, sp = self.line_cycle, self.sprites
        if lc == 0:
            self._check_vblank()
            self._end_dma(2)
        elif lc == 1:
            self._vblank()
            self._start_dma(5)
            if not sp.dma & 0xF8:
                return 10
        elif lc in (2, 4):
            self._end_dma(lc // 2 + 2)
        elif lc in (3, 5):
            self._start_dma(lc // 2 + 5)
        elif lc == 6:
            self._end_dma(5)
            return 2 if sp.dma & 0xC0 else 5
        elif lc == 8:
            self._end_dma(6)
            return 2
        elif lc == 10:
            self._end_dma(7)
        elif 11 <= lc <= 15:
            return self._clock_common(lc)
        elif lc == 54:
            sp.check_dma(self.raster_y)
            self._start_dma(0)
        elif lc == 55:
            sp.check_dma(self.raster_y)
            sp.check_exp()
            self._start_dma(0)
        elif lc == 57:
            sp.check_display()
            if not sp.dma & 0x1F:
                return 6
        elif lc in (56, 58, 60, 62):
            self._start_dma((lc - 54) >> 1)
        elif lc in (59, 61):
            self._end_dma((lc - 59) >> 1)
        elif lc not in (7, 9):
            return 54 - lc
        return 1

    def _clock_ntsc(self) -> int:
        # pylint: disable=too-many-branches,too-many-return-statements
        lc, sp = self.line_cycle, self.sprites
        if lc == 0:
            self._check_vblank()
            self._start_dma(5)
        elif lc == 1:
            self._vblank()
            self._end_dma(3)
            if not sp.dma & 0xF0:
                return 10
        elif lc in (2, 4):
            self._start_dma(lc // 2 + 5)
        elif lc == 3:
            self._end_dma(4)
        elif lc == 5:
            self._end_dma(5)
            return 2 if sp.dma & 0xC0 else 6
        elif lc == 7:
            self._end_dma(6)
            return 2
        elif lc == 9:
            self._end_dma(7)
            return 2
        elif 11 <= lc <= 15:
            return self._clock_common(lc)
        elif lc == 54:
            self.set_ba(True)
        elif lc in (55, 56):
            sp.check_dma(self.raster_y)
            if lc == 55:
                sp.check_exp()
            self._start_dma(0)
        elif lc == 58:
            sp.check_display()
            if not sp.dma & 0x1F:
                return 7
        elif lc in (57, 59, 61, 63):
            self._start_dma((lc - 55) >> 1)
        elif lc in (60, 62, 64):
            self._end_dma((lc - 60) >> 1)
        elif lc not in (6, 8, 10):
            return 54 - lc
        return 1

    def read(self, addr: int) -> int:
        """A CPU read of register ``addr`` (mirrored every 64 bytes)."""
        addr &= 0x3F
        self.sync()
        if addr == 0x11:
            return (self.regs[0x11] & 0x7F) | (self.raster_y & 0x100) >> 1
        if addr == 0x12:
            return self.raster_y & 0xFF
        if addr == 0x19:
            return self.irq_flags | 0x70
        if addr == 0x1A:
            return self.irq_mask | 0xF0
        if addr in (0x13, 0x14):
            return 0
        if addr < 0x20:
            return self.regs[addr]
        return self.regs[addr] | 0xF0 if addr < 0x2F else 0xFF

    def write(self, addr: int, data: int) -> None:
        """A CPU write of ``data`` to register ``addr``."""
        addr &= 0x3F
        self.regs[addr] = data
        self.sync()
        if addr == 0x11:
            self._write_control(data)
        if addr in (0x11, 0x12):
            self.sched.schedule(self.edge_event, 0, PHI1)
        elif addr == 0x17:
            self.sprites.line_crunch(data, self.line_cycle)
        elif addr == 0x19:
            self.irq_flags &= (~data & 0x0F) | 0x80
            self._irq_state()
        elif addr == 0x1A:
            self.irq_mask = data & 0x0F
            self._irq_state()

    def _write_control(self, data: int) -> None:
        old_yscroll, self.yscroll = self.yscroll, data & 7
        was_enabled, y = self.bad_lines_enabled, self.raster_y
        if y == FIRST_DMA_LINE and self.line_cycle == 0:
            self.bad_lines_enabled = self._den()
        old_y = (y if y > 0 else self.max_rasters) - 1
        if old_y == FIRST_DMA_LINE and self._den():
            self.bad_lines_enabled = True
        changed = old_yscroll != self.yscroll or self.bad_lines_enabled != was_enabled
        if not changed or not FIRST_DMA_LINE <= y <= LAST_DMA_LINE:
            return
        was_bad = was_enabled and old_yscroll == (y & 7)
        now_bad = self.bad_lines_enabled and self.yscroll == (y & 7)
        if now_bad == was_bad:
            return
        old_bad = self.bad_line
        if was_bad:
            if self.line_cycle < _FETCH_CYCLE:
                self.bad_line = False
        elif self.line_cycle <= _FETCH_CYCLE + _TEXT_COLS + 6:
            self.bad_line = True
        if self.bad_line != old_bad:
            self.sched.schedule(self.badline_event, 0, PHI1)


__all__ = ["Vic", "Sprites", "PAL_GEOMETRY", "NTSC_GEOMETRY"]
