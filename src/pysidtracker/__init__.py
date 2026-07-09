"""Shared base for the pure-Python C64 SID tracker parsers.

Provides the one PSID/RSID container parser, the loaded-image model, the source
dispatch, the error hierarchy, and the packed/relocating playroutine detector
that the ``py*`` format packages (pygoattracker, pysidwizard, pydmcsid,
pyfuturecomposer, pymusicassembler, pydefmon, pyjch) build on for a consistent
API.
"""

from .base import BaseSidParser
from .detect import (
    Detection,
    PlayroutineKind,
    Recognizer,
    detect_playroutine,
    run_init,
)
from .errors import (
    EmulatorUnavailable,
    SidError,
    SidFormatError,
    SidParseError,
)
from .header import PSID_MAGIC, RSID_MAGIC, SidHeader, parse_sid_header
from .image import MEM_SIZE, SidImage
from .source import Source, read_bytes

__version__ = "0.1.0"

__all__ = [
    "BaseSidParser",
    "Detection",
    "EmulatorUnavailable",
    "MEM_SIZE",
    "PSID_MAGIC",
    "PlayroutineKind",
    "RSID_MAGIC",
    "Recognizer",
    "SidError",
    "SidFormatError",
    "SidHeader",
    "SidImage",
    "SidParseError",
    "Source",
    "__version__",
    "detect_playroutine",
    "parse_sid_header",
    "read_bytes",
    "run_init",
]
