"""Shared base for the pure-Python C64 SID tracker parsers.

Provides the one PSID/RSID container parser, the loaded-image model, the source
dispatch, the error hierarchy, and the packed/relocating playroutine detector
that the ``py*`` format packages (pygoattracker, pysidwizard, pydmcsid,
pyfuturecomposer, pymusicassembler, pydefmon, pyjch) build on for a consistent
API.
"""

from . import mos6502, notefreq, registers
from .audio import (
    default_device,
    frames_to_seconds,
    render_samples,
    render_wav,
    resolve_device,
    seconds_to_frames,
)
from .base import BaseSidParser
from .cadence import Cadence, TriggerSource, cadence_from_latch, playroutine_cadence
from .cli import add_reglog_command, add_wav_command, print_info, run_cli
from .codescan import CodePattern, Match, find_code_all, find_code_first
from .detect import (
    Detection,
    PlayroutineKind,
    Recognizer,
    detect_playroutine,
    is_jmp_vector,
    resolve_entry_points,
    run_init,
)
from .decrunch import native_decrunch
from .errors import (
    AudioUnavailable,
    EmulatorUnavailable,
    PackageErrors,
    SidError,
    SidFormatError,
    SidParseError,
    make_package_errors,
)
from .header import (
    PSID_MAGIC,
    RSID_MAGIC,
    SidHeader,
    decode_cstr,
    encode_cstr,
    parse_sid_header,
    write_psid,
)
from .image import MEM_SIZE, SidImage, parse_prg
from .io import ByteCursor, byte_range, check
from .mos6502 import (
    OP_LEN,
    SidWriteCapturingMemory,
    adc,
    s8,
    sbc,
    walk_until,
)
from .notefreq import (
    NTSC_FREQ_HI,
    NTSC_FREQ_LO,
    PAL_FREQ_HI,
    PAL_FREQ_LO,
    NoteFreqTable,
    locate_note_freq,
)
from .oracle import aligned_match, grid_from_writes, read_sidwr, register_grid
from .player import MemPlayer
from .registers import (
    NTSC_CLOCK_HZ,
    NTSC_CYCLES_PER_FRAME,
    PAL_CLOCK_HZ,
    PAL_CYCLES_PER_FRAME,
    PW_HI_REGS,
    SID_BASE,
    SID_REG_COUNT,
    SID_VOICE_OFFSET,
    SID_VOICES,
    RegisterStore,
    attack_decay,
    find_register_stores,
    sustain_release,
)
from .reglog import (
    DEFAULT_WRITE_SPACING,
    REGLOG_HEADER,
    RegWrite,
    frame_writes,
    read_reglog,
    register_writes_from_player,
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

__version__ = "0.5.0"

__all__ = [
    "AudioUnavailable",
    "BaseSidParser",
    "ByteCursor",
    "Cadence",
    "CodePattern",
    "DEFAULT_MIRROR",
    "DEFAULT_WRITE_SPACING",
    "Detection",
    "EmulatorUnavailable",
    "InitTrace",
    "MEM_SIZE",
    "Match",
    "MemPlayer",
    "NTSC_CLOCK_HZ",
    "NTSC_CYCLES_PER_FRAME",
    "NTSC_FREQ_HI",
    "NTSC_FREQ_LO",
    "NoteFreqTable",
    "OP_LEN",
    "PAL_CLOCK_HZ",
    "PAL_CYCLES_PER_FRAME",
    "PAL_FREQ_HI",
    "PAL_FREQ_LO",
    "PSID_MAGIC",
    "PW_HI_REGS",
    "PackageErrors",
    "PlayroutineKind",
    "REGLOG_HEADER",
    "RSID_MAGIC",
    "Recognizer",
    "RegWrite",
    "RegisterStore",
    "SID_BASE",
    "SID_REG_COUNT",
    "SID_VOICES",
    "SID_VOICE_OFFSET",
    "SidError",
    "SidFormatError",
    "SidHeader",
    "SidImage",
    "SidParseError",
    "SidWriteCapturingMemory",
    "Source",
    "TriggerSource",
    "TuneFetchError",
    "__version__",
    "adc",
    "add_reglog_command",
    "add_wav_command",
    "aligned_match",
    "attack_decay",
    "byte_range",
    "cadence_from_latch",
    "check",
    "decode_cstr",
    "default_device",
    "detect_playroutine",
    "encode_cstr",
    "fetch_tune",
    "find_code_all",
    "find_code_first",
    "find_register_stores",
    "frame_writes",
    "frames_to_seconds",
    "grid_from_writes",
    "is_jmp_vector",
    "locate_note_freq",
    "make_package_errors",
    "make_tune_fixtures",
    "mos6502",
    "native_decrunch",
    "notefreq",
    "parse_prg",
    "parse_sid_header",
    "playroutine_cadence",
    "print_info",
    "read_bytes",
    "read_reglog",
    "read_sidwr",
    "register_grid",
    "register_writes_from_player",
    "registers",
    "render_samples",
    "render_wav",
    "resolve_device",
    "resolve_entry_points",
    "resolve_tune",
    "run_cli",
    "run_init",
    "s8",
    "sbc",
    "seconds_to_frames",
    "sustain_release",
    "trace_init",
    "walk_until",
    "write_psid",
    "write_reglog",
]
