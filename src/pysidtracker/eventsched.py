"""The two-phase event scheduler: :mod:`vagrant_moses.eventsched` re-exported.

Importing :mod:`pysidtracker.eventsched` still works; every name below is the
canonical machine model from vagrant-moses.
"""

from __future__ import annotations

from vagrant_moses.eventsched import NEVER, Event, PHI1, PHI2, Scheduler

__all__ = ["PHI1", "PHI2", "NEVER", "Event", "Scheduler"]
