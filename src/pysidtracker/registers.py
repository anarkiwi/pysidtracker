"""SID register semantics, PSID flag helpers and absolute-store scanning.

The C64 memory map (chip windows, vectors, video timing) is
:mod:`vagrant_moses.memmap`; the few of its names dependent packages import from
here are re-exported. Several ``py*`` parsers key their relocation-invariant
fingerprints on stores to fixed hardware addresses (Soundmonitor on the CIA-timer
latch ``$DC04``/``$DC05``, defMON on the SID write band ``$D400..$D406``, JCH on
``STA $D405``/``$D40C``), found with :func:`find_register_stores`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set

import numpy as _np
from vagrant_moses import memmap as _mm
from vagrant_moses.memmap import (  # pylint: disable=unused-import
    # re-exported for dependent packages
    CIA1_TIMER_A_HI,
    CIA1_TIMER_A_LO,
    SID_BASE,
    SID_LAST,
    SID_STRIDE,
)

from .image import SidImage

PAL_CLOCK_HZ, NTSC_CLOCK_HZ = _mm.PAL.clock_hz, _mm.NTSC.clock_hz
PAL_CYCLES_PER_FRAME = _mm.PAL.cycles_per_frame
NTSC_CYCLES_PER_FRAME = _mm.NTSC.cycles_per_frame

# --- SID (MOS 6581/8580) ----------------------------------------------------
# The SID decodes in the vagrant_moses.memmap SID window, mirrored every $20.
SID_REG_COUNT = 0x19  # $D400..$D418 are the real registers
SID_CTRL_V1 = 0xD404  # voice-1 control (waveform + gate)
SID_CTRL_V2 = 0xD40B  # voice-2 control
SID_CTRL_V3 = 0xD412  # voice-3 control
SID_FILTER_LO = 0xD415  # cutoff lo (3 bits)
SID_FILTER_HI = 0xD416  # cutoff hi
SID_RES_FILT = 0xD417  # resonance + filter routing
SID_MODE_VOL = 0xD418  # filter mode + master volume

# The three SID voices are identical 7-register blocks; add a voice offset to a
# voice-1 register to reach the same register in voice 2 or 3.
SID_VOICES = 3
SID_VOICE_REG_SIZE = 7
SID_VOICE_OFFSET = (0, 7, 14)

# Voice-relative SID register indices (offset within a voice's 7-register block).
FREQ_LO = 0
FREQ_HI = 1
PW_LO = 2
PW_HI = 3
CTRL = 4
AD = 5
SR = 6

# Global SID register indices (offset within the 25-register file from $D400).
FC_LO = 0x15
FC_HI = 0x16
RES_FILT = 0x17
MODE_VOL = 0x18
# The pulse-width-high registers ($D403/$D40A/$D411): only the low 4 bits are
# significant (12-bit pulse width), so a register grid masks them to a nibble.
PW_HI_REGS = (0x03, 0x0A, 0x11)


def cycles_per_frame_for_flags(flags: int) -> int:
    """PAL/NTSC CPU cycles per frame from PSID header ``flags`` (clock bits 2-3).

    The two clock bits encode 1=PAL, 2=NTSC, 0=unknown, 3=PAL and NTSC; unknown
    and both default to PAL (the HVSC-dominant clock). A player's frame cadence
    follows the tune's clock, so an oracle grid must be framed at this rate.
    """
    return (
        NTSC_CYCLES_PER_FRAME if ((flags >> 2) & 0b11) == 0b10 else PAL_CYCLES_PER_FRAME
    )


def sid_model_for_flags(flags: int) -> str:
    """The SID model a player emulates from PSID header ``flags`` (model bits 4-5).

    ``"8580"`` when the header names the 8580 alone, else ``"6581"``: unknown and
    both fall back to the 6581, as libsidplayfp's default model does.
    """
    return "8580" if ((flags >> 4) & 0b11) == 0b10 else "6581"


# Absolute-store opcodes and their addressing mode (for the scanner).
STA_ABS = 0x8D
STX_ABS = 0x8E
STY_ABS = 0x8C
STA_ABSX = 0x9D
STA_ABSY = 0x99
STORE_ABS_OPCODES = (STA_ABS, STX_ABS, STY_ABS, STA_ABSX, STA_ABSY)

_STORE_MNEMONIC = {
    STA_ABS: "STA",
    STX_ABS: "STX",
    STY_ABS: "STY",
    STA_ABSX: "STA",
    STA_ABSY: "STA",
}


def sid_register(addr: int) -> int:
    """The base SID register ($D400..$D418) an address mirrors to."""
    return SID_BASE + ((addr - SID_BASE) % SID_STRIDE)


def is_sid_reg(addr: int) -> bool:
    """True if ``addr`` is a SID register (including the $20 mirror band)."""
    return (
        SID_BASE <= addr <= SID_LAST and (addr - SID_BASE) % SID_STRIDE < SID_REG_COUNT
    )


def is_sid_control(addr: int) -> bool:
    """True if ``addr`` mirrors a SID voice control register."""
    return is_sid_reg(addr) and sid_register(addr) in (
        SID_CTRL_V1,
        SID_CTRL_V2,
        SID_CTRL_V3,
    )


@dataclass(frozen=True)
class RegisterStore:
    """One absolute-addressed store instruction targeting a watched address.

    Attributes:
      addr: C64 address of the store opcode.
      opcode: the store opcode byte (see :data:`STORE_ABS_OPCODES`).
      target: the absolute operand address written to.
      mnemonic: ``"STA"`` / ``"STX"`` / ``"STY"``.
    """

    addr: int
    opcode: int
    target: int
    mnemonic: str


def _store_sites(buf: bytes, opcode: int) -> Dict[int, int]:
    """Map ``{site_addr: target}`` for every ``opcode lo hi`` in ``buf``."""
    arr = _np.frombuffer(buf, dtype=_np.uint8)
    sites = _np.nonzero(arr[:-2] == opcode)[0]
    lo = arr[sites + 1].astype(_np.int32)
    hi = arr[sites + 2].astype(_np.int32)
    targets = (lo | (hi << 8)).tolist()
    return dict(zip(sites.tolist(), targets))


def find_register_stores(image: SidImage, addrs: Iterable[int]) -> List[RegisterStore]:
    """Every absolute store in ``image`` that targets one of ``addrs``.

    Scans ``image.mem`` for ``STA``/``STX``/``STY`` absolute and ``STA``
    absolute-indexed instructions (opcodes ``8D``/``8E``/``8C``/``9D``/``99``)
    whose operand address is in ``addrs``. Returns :class:`RegisterStore`
    records in ascending site order. Relocation-invariant when ``addrs`` are
    fixed hardware registers, which never move.
    """
    wanted: Set[int] = set(addrs)
    buf = bytes(image.mem)
    out: List[RegisterStore] = []
    for opcode in STORE_ABS_OPCODES:
        for site, target in _store_sites(buf, opcode).items():
            if target in wanted:
                out.append(RegisterStore(site, opcode, target, _STORE_MNEMONIC[opcode]))
    out.sort(key=lambda s: s.addr)
    return out


def sid_write_band(*, lo: int = SID_BASE, hi: int = SID_MODE_VOL) -> Sequence[int]:
    """The inclusive SID register address range ``lo..hi`` as a sequence."""
    return range(lo, hi + 1)


def _nibble(value: int, what: str) -> int:
    if not 0 <= value <= 15:
        raise ValueError(f"{what} out of nibble range (0..15): {value}")
    return value


def attack_decay(attack: int, decay: int) -> int:
    """Pack an ADSR attack/decay byte (``attack<<4 | decay``); nibbles validated."""
    return (_nibble(attack, "attack") << 4) | _nibble(decay, "decay")


def sustain_release(sustain: int, release: int) -> int:
    """Pack an ADSR sustain/release byte (``sustain<<4 | release``); validated."""
    return (_nibble(sustain, "sustain") << 4) | _nibble(release, "release")
