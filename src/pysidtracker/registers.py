"""C64 hardware register map and absolute-store scanning.

The addresses and layouts below are documented C64 hardware facts (SID, CIA, VIC,
the RAM interrupt vectors and the CPU vector table), not copyrightable expression.
Several ``py*`` parsers key their relocation-invariant fingerprints on stores to
these fixed addresses (Soundmonitor on the CIA-timer latch ``$DC04``/``$DC05``,
defMON on the SID write band ``$D400..$D406``, JCH on ``STA $D405``/``$D40C``),
so this module gives them shared named constants, predicates, and a scanner for
absolute stores that target any of a set of addresses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set

from .image import SidImage

import numpy as _np

# --- SID (MOS 6581/8580) ----------------------------------------------------
# The SID occupies $D400..$D41F and is mirrored every $20 up to $D7FF.
SID_BASE = 0xD400
SID_LAST = 0xD7FF
SID_STRIDE = 0x20
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

# --- C64 frame timing (documented hardware facts) ---------------------------
# CPU cycles in one video frame and the CPU clock, per video standard.
PAL_CYCLES_PER_FRAME = 19656
NTSC_CYCLES_PER_FRAME = 17095
PAL_CLOCK_HZ = 985248
NTSC_CLOCK_HZ = 1022727


def cycles_per_frame_for_flags(flags: int) -> int:
    """PAL/NTSC CPU cycles per frame from PSID header ``flags`` (clock bits 2-3).

    The two clock bits encode 1=PAL, 2=NTSC, 0=unknown, 3=PAL and NTSC; unknown
    and both default to PAL (the HVSC-dominant clock). A player's frame cadence
    follows the tune's clock, so an oracle grid must be framed at this rate.
    """
    return (
        NTSC_CYCLES_PER_FRAME if ((flags >> 2) & 0b11) == 0b10 else PAL_CYCLES_PER_FRAME
    )


# --- CIA #1 ($DC00..$DC0F) --------------------------------------------------
CIA1_BASE = 0xDC00
CIA1_LAST = 0xDC0F
CIA1_TIMER_A_LO = 0xDC04
CIA1_TIMER_A_HI = 0xDC05
CIA1_TIMER_B_LO = 0xDC06
CIA1_TIMER_B_HI = 0xDC07
CIA1_ICR = 0xDC0D
CIA1_CRA = 0xDC0E
CIA1_CRB = 0xDC0F

# --- CIA #2 ($DD00..$DD0F) --------------------------------------------------
CIA2_BASE = 0xDD00
CIA2_LAST = 0xDD0F
CIA2_TIMER_A_LO = 0xDD04
CIA2_TIMER_A_HI = 0xDD05
CIA2_TIMER_B_LO = 0xDD06
CIA2_TIMER_B_HI = 0xDD07
CIA2_ICR = 0xDD0D
CIA2_CRA = 0xDD0E
CIA2_CRB = 0xDD0F

_CIA1_TIMERS = frozenset(
    (CIA1_TIMER_A_LO, CIA1_TIMER_A_HI, CIA1_TIMER_B_LO, CIA1_TIMER_B_HI)
)
_CIA2_TIMERS = frozenset(
    (CIA2_TIMER_A_LO, CIA2_TIMER_A_HI, CIA2_TIMER_B_LO, CIA2_TIMER_B_HI)
)

# --- VIC-II ($D000..$D02E) --------------------------------------------------
VIC_BASE = 0xD000
VIC_LAST = 0xD02E
VIC_CONTROL_1 = 0xD011  # bit7 = raster compare bit 8; also screen control
VIC_RASTER = 0xD012  # raster compare lo (also raster read)
VIC_IRQ_STATUS = 0xD019
VIC_IRQ_ENABLE = 0xD01A

# --- RAM interrupt vectors --------------------------------------------------
RAM_IRQ_VECTOR_LO = 0x0314
RAM_IRQ_VECTOR_HI = 0x0315
RAM_NMI_VECTOR_LO = 0x0318
RAM_NMI_VECTOR_HI = 0x0319

# --- CPU vector table (top of memory) ---------------------------------------
CPU_NMI_VECTOR = 0xFFFA
CPU_RESET_VECTOR = 0xFFFC
CPU_IRQ_VECTOR = 0xFFFE

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


def is_cia1_reg(addr: int) -> bool:
    """True if ``addr`` is a CIA #1 register."""
    return CIA1_BASE <= addr <= CIA1_LAST


def is_cia2_reg(addr: int) -> bool:
    """True if ``addr`` is a CIA #2 register."""
    return CIA2_BASE <= addr <= CIA2_LAST


def is_cia_timer(addr: int) -> bool:
    """True if ``addr`` is a CIA #1 or #2 timer latch/counter register."""
    return addr in _CIA1_TIMERS or addr in _CIA2_TIMERS


def is_vic_reg(addr: int) -> bool:
    """True if ``addr`` is a VIC-II register."""
    return VIC_BASE <= addr <= VIC_LAST


def is_irq_vector(addr: int) -> bool:
    """True if ``addr`` is a RAM IRQ vector byte ($0314/$0315)."""
    return addr in (RAM_IRQ_VECTOR_LO, RAM_IRQ_VECTOR_HI)


def is_nmi_vector(addr: int) -> bool:
    """True if ``addr`` is a RAM NMI vector byte ($0318/$0319)."""
    return addr in (RAM_NMI_VECTOR_LO, RAM_NMI_VECTOR_HI)


def is_cpu_vector(addr: int) -> bool:
    """True if ``addr`` is a CPU vector byte ($FFFA..$FFFF)."""
    return CPU_NMI_VECTOR <= addr <= 0xFFFF


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
