"""The 6526/8521 CIA model: :mod:`vagrant_moses.cia` re-exported for compatibility.

Importing :mod:`pysidtracker.cia` still works; every name below is the canonical
machine model from vagrant-moses.
"""

from __future__ import annotations

from vagrant_moses.cia import Cia, InterruptSource, Timer

__all__ = ["Cia", "Timer", "InterruptSource"]
