"""The VIC-II model: :mod:`vagrant_moses.vicii` re-exported for compatibility.

Importing :mod:`pysidtracker.vicii` still works; every name below is the
canonical machine model from vagrant-moses.
"""

from __future__ import annotations

from vagrant_moses.vicii import NTSC_GEOMETRY, PAL_GEOMETRY, Sprites, Vic

__all__ = ["Vic", "Sprites", "PAL_GEOMETRY", "NTSC_GEOMETRY"]
