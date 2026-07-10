"""Shared base for the pure-Python C64 SID tracker parsers.

Provides the one PSID/RSID container parser, the loaded-image model, the source
dispatch, the error hierarchy, and the packed/relocating playroutine detector
that the ``py*`` format packages (pygoattracker, pysidwizard, pydmcsid,
pyfuturecomposer, pymusicassembler, pydefmon, pyjch) build on for a consistent
API.
"""

from . import registers
from .audio import render_samples, render_wav
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
    AudioUnavailable,
    EmulatorUnavailable,
    SidError,
    SidFormatError,
    SidParseError,
)
from .header import PSID_MAGIC, RSID_MAGIC, SidHeader, parse_sid_header
from .image import MEM_SIZE, SidImage
from .oracle import aligned_match, grid_from_writes, read_sidwr, register_grid
from .registers import (
    NTSC_CLOCK_HZ,
    NTSC_CYCLES_PER_FRAME,
    PAL_CLOCK_HZ,
    PAL_CYCLES_PER_FRAME,
    PW_HI_REGS,
    SID_BASE,
    SID_REG_COUNT,
    SID_VOICE_OFFSET,
    RegisterStore,
    find_register_stores,
)
from .reglog import (
    DEFAULT_WRITE_SPACING,
    REGLOG_HEADER,
    RegWrite,
    frame_writes,
    read_reglog,
    write_reglog,
)
from .source import Source, read_bytes
from .testing import (
    DEFAULT_MIRROR,
    TuneFetchError,
    fetch_tune,
    make_tune_fixtures,
    resolve_tune,
)
from .trace import InitTrace, trace_init

__version__ = "0.3.0"

__all__ = [
    "AudioUnavailable",
    "BaseSidParser",
    "CodePattern",
    "DEFAULT_MIRROR",
    "DEFAULT_WRITE_SPACING",
    "Detection",
    "EmulatorUnavailable",
    "InitTrace",
    "MEM_SIZE",
    "Match",
    "NTSC_CLOCK_HZ",
    "NTSC_CYCLES_PER_FRAME",
    "PAL_CLOCK_HZ",
    "PAL_CYCLES_PER_FRAME",
    "PSID_MAGIC",
    "PW_HI_REGS",
    "PlayroutineKind",
    "REGLOG_HEADER",
    "RSID_MAGIC",
    "Recognizer",
    "RegWrite",
    "RegisterStore",
    "SID_BASE",
    "SID_REG_COUNT",
    "SID_VOICE_OFFSET",
    "SidError",
    "SidFormatError",
    "SidHeader",
    "SidImage",
    "SidParseError",
    "Source",
    "TuneFetchError",
    "__version__",
    "aligned_match",
    "detect_playroutine",
    "fetch_tune",
    "find_code_all",
    "find_code_first",
    "find_register_stores",
    "frame_writes",
    "grid_from_writes",
    "make_tune_fixtures",
    "parse_sid_header",
    "read_bytes",
    "read_reglog",
    "read_sidwr",
    "register_grid",
    "registers",
    "render_samples",
    "render_wav",
    "resolve_tune",
    "run_init",
    "trace_init",
    "write_reglog",
]
