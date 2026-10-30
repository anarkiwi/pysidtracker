"""Exception hierarchy shared by the ``py*`` SID tracker parsers.

Every parser raises errors that derive from :class:`SidError`, so a caller can
``except SidError`` across formats. Individual packages subclass these to keep
their own format-specific names (e.g. a GoatTracker ``SngParseError``) while
staying catchable through the shared base.
"""

from contextlib import contextmanager
from typing import Iterator, NamedTuple

from vagrant_moses import kernal as _kernal


class SidError(Exception):
    """Base class for all pysidtracker errors."""


class SidParseError(SidError):
    """A ``.sid``/``.prg`` image could not be parsed."""


class SidFormatError(SidParseError):
    """The container is not a recognised PSID/RSID/PRG, or is truncated."""


class IncompleteComparison(SidError):
    """A render comparison covers less than the song's whole length: a reference
    or render cut to a window cannot vouch for the frames past it."""


class EmulatorUnavailable(SidError):
    """The 6502 emulator (``jennings``) is needed but not importable.

    ``jennings`` is a core dependency, so this only fires on a broken install;
    ``pip install pysidtracker`` restores it.
    """


class KernalHostError(SidError, _kernal.KernalHostError):
    """A whole-program KERNAL run (:mod:`pysidtracker.kernalhost`) could not finish.

    The step cap was reached before the exit condition, or a BASIC stub has no
    numeric ``SYS``. Also a :class:`vagrant_moses.kernal.KernalHostError`.
    """


class AudioUnavailable(SidError):
    """The SID audio backend (``pyresidfp``) is needed but not importable.

    ``pyresidfp`` is a core dependency, so this only fires on a broken install;
    raised by :mod:`pysidtracker.audio` when a WAV/sample render needs an
    emulated SID and it cannot be imported.
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


@contextmanager
def reraise_as(error: type, catch: type = SidParseError) -> Iterator[None]:
    """Re-raise a ``catch`` raised in the block as ``error`` (same message).

    A writer re-reading its own output wraps the reads so a parse error
    surfaces as its validation error: ``with reraise_as(SongValidationError):``.
    """
    try:
        yield
    except catch as exc:
        raise error(str(exc)) from exc
