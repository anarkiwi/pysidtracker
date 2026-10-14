"""A two-phase discrete event scheduler with libsidplayfp's ordering.

Time counts half-cycles: cycle ``t`` has a PHI1 slot ``2t`` (VIC-II, CIAs) and a
PHI2 slot ``2t + 1`` (CPU). Events due at the same slot fire in scheduling order.
"""

from __future__ import annotations

from typing import Callable, List

PHI1, PHI2 = 0, 1
NEVER = 1 << 62


class Event:
    """A callback the scheduler fires at its ``trigger`` half-cycle."""

    __slots__ = ("trigger", "fire", "pending")

    def __init__(self, fire: Callable[[], None]):
        self.fire = fire
        self.trigger = NEVER
        self.pending = False


class Scheduler:
    """Pending events in trigger order; ``now`` is the current half-cycle."""

    __slots__ = ("now", "queue", "next")

    def __init__(self):
        self.now = 0
        self.queue: List[Event] = []
        self.next = NEVER

    def _insert(self, event: Event, trigger: int) -> None:
        if event.pending:
            self.queue.remove(event)
        event.trigger, event.pending = trigger, True
        queue = self.queue
        i = len(queue)
        while i and queue[i - 1].trigger > trigger:
            i -= 1
        queue.insert(i, event)
        self.next = queue[0].trigger

    def schedule(self, event: Event, cycles: int, phase: int) -> None:
        """Fire ``event`` ``cycles`` cycles after the next ``phase`` slot from now."""
        now = self.now
        self._insert(event, now + ((now & 1) ^ phase) + (cycles << 1))

    def schedule_same_phase(self, event: Event, cycles: int) -> None:
        """Fire ``event`` ``cycles`` whole cycles from now, in the current phase."""
        self._insert(event, self.now + (cycles << 1))

    def cancel(self, event: Event) -> None:
        """Drop ``event`` if it is pending."""
        if event.pending:
            event.pending = False
            self.queue.remove(event)
            self.next = self.queue[0].trigger if self.queue else NEVER

    def time(self, phase: int) -> int:
        """The current cycle as seen from ``phase`` (libsidplayfp ``getTime``)."""
        return (self.now + (phase ^ 1)) >> 1

    def run(self, limit: int) -> None:
        """Fire every event due at or before half-cycle ``limit``, in order."""
        queue = self.queue
        while self.next <= limit:
            event = queue.pop(0)
            self.next = queue[0].trigger if queue else NEVER
            event.pending = False
            self.now = event.trigger
            event.fire()


__all__ = ["PHI1", "PHI2", "NEVER", "Event", "Scheduler"]
