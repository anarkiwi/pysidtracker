"""The 6526/8521 CIA timers and interrupt logic as libsidplayfp models them.

A port of ``timer.cpp``, ``interrupt.cpp`` and ``mos652x.cpp`` (VICE's ``ciatimer``
state machine) onto :mod:`~pysidtracker.eventsched`. TOD and the serial port are
not modelled: their registers read back what was written.
"""

from __future__ import annotations

from typing import Callable

from .eventsched import PHI1, PHI2, Event, Scheduler

_START, _STEP, _CR_ONESHOT, _CR_FLOAD, _PHI2IN = 0x01, 0x04, 0x08, 0x10, 0x20
_CR_MASK = _START | _CR_ONESHOT | _CR_FLOAD | _PHI2IN
_COUNT2, _COUNT3 = 0x100, 0x200
_ONESHOT0, _ONESHOT = _CR_ONESHOT << 8, _CR_ONESHOT << 16
_LOAD1, _LOAD = _CR_FLOAD << 8, _CR_FLOAD << 16
_OUT = 0x80000000
_UNWANTED = _OUT | _CR_FLOAD | _LOAD1 | _LOAD
_STEADY = _START | _PHI2IN | _COUNT2 | _COUNT3
UNDERFLOW_A, UNDERFLOW_B, REQUEST = 0x01, 0x02, 0x80
PRA, PRB, DDRA, DDRB, TAL, TAH, TBL, TBH = range(8)
ICR, CRA, CRB = 13, 14, 15


class Timer:
    """One CIA timer: the cycle-exact count/load/one-shot pipeline."""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, sched: Scheduler, underflow: Callable[[], None]):
        self.sched, self.underflow = sched, underflow
        self.event = Event(self._event)
        self.skip_event = Event(self._skip)
        self.timer = self.latch = 0xFFFF
        self.state = self.last_control = self.pause = 0
        self.pb_toggle = False

    def reset(self) -> None:
        """Stopped, latch and counter ``$FFFF``, first clock next cycle."""
        self.sched.cancel(self.event)
        self.timer = self.latch = 0xFFFF
        self.pb_toggle = False
        self.state = self.last_control = self.pause = 0
        self.sched.schedule(self.event, 1, PHI1)

    def set_control(self, cr: int) -> None:
        """Latch control register bits into the pipeline state."""
        self.state = (self.state & ~_CR_MASK) | ((cr & _CR_MASK) ^ _PHI2IN)
        self.last_control = cr

    def sync(self) -> None:
        """Bring a skipping or sleeping timer up to the CPU's cycle."""
        if self.pause > 0:
            self.sched.cancel(self.skip_event)
            elapsed = self.sched.time(PHI2) - self.pause
            if elapsed >= 0:
                self.timer = (self.timer - elapsed) & 0xFFFF
                self.clock()
        if self.pause == 0:
            self.sched.cancel(self.event)
        self.pause = -1

    def wake(self) -> None:
        """Resume per-cycle clocking from the next PHI1."""
        self.pause = 0
        self.sched.schedule(self.event, 0, PHI1)

    def _event(self) -> None:
        self.clock()
        self._reschedule()

    def _skip(self) -> None:
        elapsed = self.sched.time(PHI1) - self.pause
        self.pause = 0
        self.timer = (self.timer - elapsed) & 0xFFFF
        self._event()

    def clock(self) -> None:
        """One PHI1 tick of the pipeline; may underflow."""
        state = self.state
        if state & _COUNT3:
            self.timer = (self.timer - 1) & 0xFFFF
        adj = state & (_START | _CR_ONESHOT | _PHI2IN)
        if state & (_START | _PHI2IN) == _START | _PHI2IN:
            adj |= _COUNT2
        if state & _COUNT2 or state & (_STEP | _START) == _STEP | _START:
            adj |= _COUNT3
        adj |= (state & (_CR_FLOAD | _CR_ONESHOT | _LOAD1 | _ONESHOT0)) << 8
        self.state = adj
        if self.timer == 0 and adj & _COUNT3:
            self.state |= _LOAD | _OUT
            if self.state & (_ONESHOT | _ONESHOT0):
                self.state &= ~(_START | _COUNT2)
            toggle = self.last_control & 0x06 == 6
            self.pb_toggle = toggle and not self.pb_toggle
            self.underflow()
        if self.state & _LOAD:
            self.timer = self.latch
            self.state &= ~_COUNT3

    def _reschedule(self) -> None:
        state, sched = self.state, self.sched
        if state & _UNWANTED:
            sched.schedule_same_phase(self.event, 1)
        elif state & _COUNT3:
            if self.timer > 2 and state & _STEADY == _STEADY:
                self.pause = sched.time(PHI1) + 1
                sched.schedule_same_phase(self.skip_event, self.timer - 1)
            else:
                sched.schedule_same_phase(self.event, 1)
        elif (
            state & (_START | _PHI2IN) == _START | _PHI2IN
            or state & (_START | _STEP) == _START | _STEP
        ):
            sched.schedule_same_phase(self.event, 1)
        else:
            self.pause = -1

    def latch_lo(self, data: int) -> None:
        """Write the latch low byte."""
        self.latch = (self.latch & 0xFF00) | data
        if self.state & _LOAD:
            self.timer = self.latch

    def latch_hi(self, data: int) -> None:
        """Write the latch high byte; a stopped timer reloads."""
        self.latch = (self.latch & 0xFF) | data << 8
        if self.state & _LOAD:
            self.timer = self.latch
        elif not self.state & _START:
            self.state |= _LOAD1

    def cascade(self) -> None:
        """Count one Timer A underflow (Timer B in cascade mode)."""
        self.sync()
        self.state |= _STEP
        self.wake()

    def pb(self, cr: int) -> bool:
        """The PB6/PB7 output: toggle or pulse."""
        return self.pb_toggle if cr & 0x04 else bool(self.state & _OUT)


class InterruptSource:
    """The ICR/IDR pair, raising ``interrupt`` 1 cycle late on a 6526."""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, sched: Scheduler, interrupt: Callable[[bool], None], new: bool):
        self.sched, self.interrupt, self.new = sched, interrupt, new
        self.interrupt_event = Event(self._interrupt)
        self.update_idr_event = Event(self._update_idr)
        self.set_irq_event = Event(self._set_irq)
        self.clear_irq_event = Event(self._clear_irq)
        self.reset()

    def reset(self) -> None:
        """No sources masked in or flagged; nothing asserted."""
        self.last_clear = self.last_set = 0
        self.icr = self.idr = self.idr_temp = 0
        for event in (
            self.update_idr_event,
            self.set_irq_event,
            self.clear_irq_event,
            self.interrupt_event,
        ):
            self.sched.cancel(event)
        self.scheduled = self.asserted = False

    def _ack0(self) -> bool:
        return self.sched.time(PHI2) == self.last_clear + 1

    def _write0(self) -> bool:
        return self.sched.time(PHI2) == self.last_set + 1

    def _interrupt(self) -> None:
        if not self.idr & REQUEST:
            self.idr |= REQUEST
            if self.new:
                self.idr_temp |= REQUEST
                if self._ack0():
                    self.sched.schedule(self.set_irq_event, 1, PHI1)
            self._set_irq()
        self.scheduled = False

    def _update_idr(self) -> None:
        self.idr = self.idr_temp
        if self._ack0():
            self.sched.schedule(self.update_idr_event, 1, PHI1)
            self.idr_temp = 0

    def _set_irq(self) -> None:
        if not self._ack0() and not self.asserted:
            self.interrupt(True)
            self.asserted = True

    def _clear_irq(self) -> None:
        if self.asserted:
            self.interrupt(False)
            self.asserted = False

    def trigger(self, mask: int) -> None:
        """Flag source ``mask`` (0 re-evaluates after an ICR write)."""
        self.idr |= mask
        self.idr_temp |= mask
        if (mask or self.idr) & self.icr:
            if not self.scheduled:
                self.sched.schedule(self.interrupt_event, 0 if self.new else 1, PHI1)
                self.scheduled = True
        elif not mask and self._write0() and self.scheduled:
            self.sched.cancel(self.interrupt_event)
            self.scheduled = False
        if not self.new and mask == UNDERFLOW_B and self._ack0():
            self.idr &= ~UNDERFLOW_B
            self.idr_temp &= ~UNDERFLOW_B

    def set(self, data: int) -> None:
        """An ICR write: set or clear mask bits."""
        if data & REQUEST:
            self.icr |= data & ~REQUEST
        else:
            self.icr &= ~data
        if not self._ack0():
            self.trigger(0)
        self.last_set = self.sched.time(PHI2)

    def clear(self) -> int:
        """An ICR read: the flags, cleared after the read."""
        self.last_clear = self.sched.time(PHI2)
        self.sched.schedule(self.clear_irq_event, 0, PHI1)
        if not self.update_idr_event.pending:
            self.sched.schedule(self.update_idr_event, 1, PHI1)
            self.idr_temp = 0
        idr = self.idr
        if not self.new:
            self.idr &= REQUEST
        return idr


class Cia:
    """One CIA: two timers, Timer B cascading on A, and the interrupt source."""

    def __init__(self, sched: Scheduler, interrupt: Callable[[bool], None], new: bool):
        self.sched = sched
        self.regs = bytearray(16)
        self.timer_a = Timer(sched, self._underflow_a)
        self.timer_b = Timer(sched, self._underflow_b)
        self.source = InterruptSource(sched, interrupt, new)
        self.b_tick = Event(self.timer_b.cascade)

    def reset(self) -> None:
        """Power-on: registers zero, timers stopped at ``$FFFF``."""
        self.regs[:] = bytes(16)
        self.timer_a.reset()
        self.timer_b.reset()
        self.source.reset()
        self.sched.cancel(self.b_tick)

    def _underflow_a(self) -> None:
        self.source.trigger(UNDERFLOW_A)
        if self.regs[CRB] & 0x41 == 0x41 and self.timer_b.state & _START:
            self.sched.schedule(self.b_tick, 0, PHI2)

    def _underflow_b(self) -> None:
        self.source.trigger(UNDERFLOW_B)

    def _port_b(self) -> int:
        data = (self.regs[PRB] | ~self.regs[DDRB]) & 0xFF
        if self.regs[CRA] & 0x02:
            data = data & 0xBF | (0x40 if self.timer_a.pb(self.regs[CRA]) else 0)
        if self.regs[CRB] & 0x02:
            data = data & 0x7F | (0x80 if self.timer_b.pb(self.regs[CRB]) else 0)
        return data

    def read(self, addr: int) -> int:
        """A CPU read of register ``addr`` (mirrored every 16 bytes)."""
        addr &= 0x0F
        for timer in (self.timer_a, self.timer_b):
            timer.sync()
            timer.wake()
        if addr == PRA:
            return (self.regs[PRA] | ~self.regs[DDRA]) & 0xFF
        if addr == PRB:
            return self._port_b()
        if TAL <= addr <= TBH:
            timer = self.timer_b if addr >= TBL else self.timer_a
            return timer.timer >> 8 * (addr & 1) & 0xFF
        if addr == ICR:
            return self.source.clear()
        if addr in (CRA, CRB):
            timer = self.timer_b if addr == CRB else self.timer_a
            return (self.regs[addr] & 0xEE) | (timer.state & 1)
        return self.regs[addr]

    def write(self, addr: int, data: int) -> None:
        """A CPU write of ``data`` to register ``addr``."""
        addr &= 0x0F
        self.timer_a.sync()
        self.timer_b.sync()
        old, self.regs[addr] = self.regs[addr], data
        if TAL <= addr <= TBH:
            timer = self.timer_b if addr >= TBL else self.timer_a
            (timer.latch_hi if addr & 1 else timer.latch_lo)(data)
        elif addr == ICR:
            self.source.set(data)
        elif addr == CRA:
            if data & 1 and not old & 1:
                self.timer_a.pb_toggle = True
            self.timer_a.set_control(data)
        elif addr == CRB:
            if data & 1 and not old & 1:
                self.timer_b.pb_toggle = True
            self.timer_b.set_control(data | (data & 0x40) >> 1)
        self.timer_a.wake()
        self.timer_b.wake()


__all__ = ["Cia", "Timer", "InterruptSource"]
