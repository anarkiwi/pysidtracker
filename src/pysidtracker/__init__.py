"""Shared base for the pure-Python C64 SID tracker parsers.

Provides the one PSID/RSID container parser, the loaded-image model, the source
dispatch, the error hierarchy, and the packed/relocating playroutine detector
that the ``py*`` format packages (pygoattracker, pysidwizard, pydmcsid,
pyfuturecomposer, pymusicassembler, pydefmon, pyjch) build on for a consistent
API.
"""

from . import registers
from .base import BaseSidParser
from .codescan import CodePattern, Match, find_code_all, find_code_first
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
from .registers import RegisterStore, find_register_stores
from .source import Source, read_bytes
from .trace import InitTrace, trace_init

__version__ = "0.2.0"

__all__ = [
    "BaseSidParser",
    "CodePattern",
    "Detection",
    "EmulatorUnavailable",
    "InitTrace",
    "MEM_SIZE",
    "Match",
    "PSID_MAGIC",
    "PlayroutineKind",
    "RSID_MAGIC",
    "Recognizer",
    "RegisterStore",
    "SidError",
    "SidFormatError",
    "SidHeader",
    "SidImage",
    "SidParseError",
    "Source",
    "__version__",
    "detect_playroutine",
    "find_code_all",
    "find_code_first",
    "find_register_stores",
    "parse_sid_header",
    "read_bytes",
    "registers",
    "run_init",
    "trace_init",
]
