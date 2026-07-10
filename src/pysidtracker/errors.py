"""Exception hierarchy shared by the ``py*`` SID tracker parsers.

Every parser raises errors that derive from :class:`SidError`, so a caller can
``except SidError`` across formats. Individual packages subclass these to keep
their own format-specific names (e.g. a GoatTracker ``SngParseError``) while
staying catchable through the shared base.
"""

from typing import NamedTuple


class SidError(Exception):
    """Base class for all pysidtracker errors."""


class SidParseError(SidError):
    """A ``.sid``/``.prg`` image could not be parsed."""


class SidFormatError(SidParseError):
    """The container is not a recognised PSID/RSID/PRG, or is truncated."""


class EmulatorUnavailable(SidError):
    """The 6502 emulator (``py65``) is needed but not importable.

    ``py65`` is a core dependency, so this only fires on a broken install;
    ``pip install pysidtracker`` restores it.
    """


class AudioUnavailable(SidError):
    """The optional SID audio backend (``pyresidfp``) is needed but missing.

    Raised by :mod:`pysidtracker.audio` when a WAV/sample render needs an
    emulated SID and ``pyresidfp`` is not installed. Install the extra with
    ``pip install pysidtracker[audio]``.
    """


class PackageErrors(NamedTuple):
    """A leaf package's error triple (unpackable as ``root, parse, format``)."""

    error: type
    parse_error: type
    format_error: type


def make_package_errors(prefix: str) -> "PackageErrors":
    """Build a leaf package's error hierarchy rooted at both its name and the base.

    Returns ``<Prefix>Error(SidError)`` plus ``<Prefix>ParseError`` and
    ``<Prefix>FormatError`` that subclass BOTH the package root AND the matching
    base :class:`SidParseError`/:class:`SidFormatError`, so a base
    ``except SidParseError`` still catches a leaf's own-named errors.
    """
    root = type(f"{prefix}Error", (SidError,), {"__doc__": f"{prefix} error."})
    parse_error = type(
        f"{prefix}ParseError",
        (root, SidParseError),
        {"__doc__": f"{prefix} parse error."},
    )
    format_error = type(
        f"{prefix}FormatError",
        (root, SidFormatError),
        {"__doc__": f"{prefix} format error."},
    )
    return PackageErrors(root, parse_error, format_error)
