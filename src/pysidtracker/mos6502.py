"""Shared MOS 6502 primitives the ``py*`` players and readers reinvented.

Instruction lengths, opcode-class sets, the two binary-mode ALU ops, a SID
write-capturing flat memory, and a linear-disassembly walker -- the small pieces
every faithful 6502 transcription in the family needs, in one place.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

from .registers import SID_BASE, SID_REG_COUNT

MEM_SIZE = 0x10000

# 6502 instruction lengths (1/2/3 bytes) indexed by opcode. Undefined/illegal
# opcodes take the legal length of their addressing-mode column, which keeps a
# linear walk aligned.
OP_LEN: Tuple[int, ...] = tuple(
    bytes.fromhex(
        "0102010202020202010201020303030302020102020202020103010303030303"
        "0302010202020202010201020303030302020102020202020103010303030303"
        "0102010202020202010201020303030302020102020202020103010303030303"
        "0102010202020202010201020303030302020102020202020103010303030303"
        "0202020202020202010201020303030302020102020202020103010303030303"
        "0202020202020202010201020303030302020102020202020103010303030303"
        "0202020202020202010201020303030302020102020202020103010303030303"
        "0202020202020202010201020303030302020102020202020103010303030303"
    )
)

# Conditional-branch opcodes (2-byte, PC-relative).
BRANCH_OPS = frozenset((0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0))

# Side-effect-free opcodes (immediate loads, register transfers, in-register
# arithmetic/shift, compares, flag ops) -- none touch memory or divert control.
INERT_OPS = frozenset(
    (
        0xA9,
        0xA2,
        0xA0,
        0xAD,
        0xAE,
        0xAC,
        0xA5,
        0xA6,
        0xA4,
        0xBD,
        0xB9,
        0xBC,
        0xBE,
        0xB5,
        0xB4,
        0xB6,
        0xAA,
        0xA8,
        0x8A,
        0x98,
        0xE8,
        0xC8,
        0xCA,
        0x88,
        0x4A,
        0x0A,
        0x6A,
        0x2A,
        0x18,
        0x38,
        0xEA,
        0xC9,
        0xE0,
        0xC0,
        0x29,
        0x09,
        0x49,
        0x69,
        0xE9,
    )
)

# Absolute (3-byte) stores and read-modify-write opcodes.
ABS_STORE_OPS = frozenset(
    (0x8D, 0x8E, 0x8C, 0xCE, 0xDE, 0xEE, 0xFE, 0x0E, 0x4E, 0x2E, 0x6E)
)

# Zero-page (2-byte) stores and read-modify-write opcodes.
ZP_STORE_OPS = frozenset(
    (
        0x85,
        0x95,
        0x86,
        0x96,
        0x84,
        0x94,
        0xC6,
        0xD6,
        0xE6,
        0xF6,
        0x06,
        0x16,
        0x46,
        0x56,
        0x26,
        0x36,
        0x66,
        0x76,
    )
)


def is_branch(opcode: int) -> bool:
    """True if ``opcode`` is a conditional branch."""
    return opcode in BRANCH_OPS


def is_inert(opcode: int) -> bool:
    """True if ``opcode`` has no memory/control side effects."""
    return opcode in INERT_OPS


def is_abs_store(opcode: int) -> bool:
    """True if ``opcode`` is an absolute store / read-modify-write."""
    return opcode in ABS_STORE_OPS


def is_zp_store(opcode: int) -> bool:
    """True if ``opcode`` is a zero-page store / read-modify-write."""
    return opcode in ZP_STORE_OPS


def s8(byte: int) -> int:
    """Reinterpret ``byte`` as a signed 8-bit integer (the 6502 sign test)."""
    byte &= 0xFF
    return byte - 0x100 if byte & 0x80 else byte


def adc(a: int, value: int, carry: int) -> Tuple[int, int]:
    """6502 ADC (binary mode): ``a + value + carry`` -> ``(byte, carry_out)``."""
    total = (a & 0xFF) + (value & 0xFF) + (carry & 1)
    return total & 0xFF, 1 if total > 0xFF else 0


def sbc(a: int, value: int, carry: int) -> Tuple[int, int]:
    """6502 SBC (binary mode): ``a - value - (1-carry)`` -> ``(byte, carry_out)``."""
    total = (a & 0xFF) - (value & 0xFF) - (1 - (carry & 1))
    return total & 0xFF, 1 if total >= 0 else 0


class SidWriteCapturingMemory:
    """A flat 64 KiB 6502 memory that records writes into the SID register band.

    Seeded with ``image`` at ``load``. Reads and writes wrap to 16 bits; a write
    whose address falls in ``$D400..$D400+SID_REG_COUNT`` is captured as a
    ``(reg, val)`` pair (``reg = addr - SID_BASE``) for the frame's reglog.
    """

    __slots__ = ("mem", "_writes")

    def __init__(self, image: bytes, load: int):
        self.mem = bytearray(MEM_SIZE)
        load &= 0xFFFF
        self.mem[load : load + len(image)] = image
        self._writes: List[Tuple[int, int]] = []

    def r(self, addr: int) -> int:
        """Read a byte (16-bit address wrap)."""
        return self.mem[addr & 0xFFFF]

    def w(self, addr: int, val: int) -> None:
        """Write a byte; SID-band writes are also captured for the reglog."""
        addr &= 0xFFFF
        val &= 0xFF
        self.mem[addr] = val
        reg = addr - SID_BASE
        if 0 <= reg < SID_REG_COUNT:
            self._writes.append((reg, val))

    def take(self) -> List[Tuple[int, int]]:
        """Return and clear this frame's captured SID ``(reg, val)`` writes."""
        out = self._writes
        self._writes = []
        return out


def walk_until(mem, addr: int, stop_ops, budget: int = 256) -> Iterator[int]:
    """Yield instruction start addresses from ``addr`` via :data:`OP_LEN`.

    Walks linearly, yielding each instruction's start address (including the
    first opcode found in ``stop_ops``, after which it stops), for at most
    ``budget`` instructions or until it runs off the end of ``mem``.
    """
    pc = addr
    for _ in range(budget):
        if not 0 <= pc < len(mem):
            return
        op = mem[pc]
        yield pc
        if op in stop_ops:
            return
        pc += OP_LEN[op]


def disasm_len(opcode: int) -> Optional[int]:
    """The byte length of ``opcode`` (1/2/3), or ``None`` if out of range."""
    if 0 <= opcode < len(OP_LEN):
        return OP_LEN[opcode]
    return None
