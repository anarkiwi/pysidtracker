"""Exception hierarchy shared by the ``py*`` SID tracker parsers.

Every parser raises errors that derive from :class:`SidError`, so a caller can
``except SidError`` across formats. Individual packages subclass these to keep
their own format-specific names (e.g. a GoatTracker ``SngParseError``) while
staying catchable through the shared base.
"""


class SidError(Exception):
    """Base class for all pysidtracker errors."""


class SidParseError(SidError):
    """A ``.sid``/``.prg`` image could not be parsed."""


class SidFormatError(SidParseError):
    """The container is not a recognised PSID/RSID/PRG, or is truncated."""


class EmulatorUnavailable(SidError):
    """The optional 6502 emulator (``py65``) is needed but not installed.

    Raised when detection has to run a packed/relocating tune's init routine to
    materialise its data and :mod:`py65` is missing. Install the extra with
    ``pip install pysidtracker[emu]``.
    """


class AudioUnavailable(SidError):
    """The optional SID audio backend (``pyresidfp``) is needed but missing.

    Raised by :mod:`pysidtracker.audio` when a WAV/sample render needs an
    emulated SID and ``pyresidfp`` is not installed. Install the extra with
    ``pip install pysidtracker[audio]``.
    """
