"""Shared MOS 6502 primitives the ``py*`` players and readers reinvented.

Instruction lengths, opcode-class sets, the two binary-mode ALU ops, a SID
write-capturing flat memory, and a linear-disassembly walker -- the small pieces
every faithful 6502 transcription in the family needs, in one place.
"""

from __future__ import annotations

from typing import Collection, Iterator, List, Optional, Set, Tuple

from jennings.opcodes import MODE_LEN, OPCODES

from .registers import SID_BASE, SID_REG_COUNT

MEM_SIZE = 0x10000

# 6502 instruction lengths (1/2/3 bytes) indexed by opcode. Undefined/illegal
# opcodes take the legal length of their addressing-mode column, which keeps a
# linear walk aligned.
OP_LEN: Tuple[int, ...] = tuple(MODE_LEN[OPCODES[op][1]] for op in range(256))

# Conditional-branch opcodes (2-byte, PC-relative).
BRANCH_OPS = frozenset(op for op in range(256) if OPCODES[op][1] == "rel")

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

# Mnemonics that write memory (stores and read-modify-writes).
WRITE_MNEMONICS = frozenset("STA STX STY INC DEC ASL LSR ROL ROR".split())

# Zero-page (2-byte) stores and read-modify-write opcodes.
ZP_STORE_OPS = frozenset(
    op
    for op in range(256)
    if OPCODES[op][0] in WRITE_MNEMONICS and OPCODES[op][1] in ("zp", "zpx", "zpy")
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


OP_JSR = 0x20
OP_JMP = 0x4C
OP_LDX_IMM = 0xA2
OP_INX = 0xE8

# BRK, RTI, RTS, JMP (ind): control does not fall through to the next instruction.
RETURN_OPS = frozenset((0x00, 0x40, 0x60, 0x6C))


def _word(image, addr: int) -> int:
    return image.peek(addr) | (image.peek(addr + 1) << 8)


def follow_jmp(image, addr: int) -> int:
    """``addr``, or where the chain of ``JMP abs`` trampolines at it leads.

    ``image`` is anything with ``peek(addr)`` (a :class:`~pysidtracker.SidImage`).
    """
    seen: Set[int] = set()
    while image.peek(addr) == OP_JMP and addr not in seen:
        seen.add(addr)
        addr = _word(image, addr + 1)
    return addr


def walk_calls(
    image, addr: int, targets: Collection[int], seen: Optional[Set[int]] = None
) -> Optional[int]:
    """The first of ``targets`` that the code at ``addr`` reaches, depth first.

    Follows ``JMP abs`` and descends into ``JSR abs`` targets, falling through
    otherwise, until a return opcode (:data:`RETURN_OPS`) or a revisit.
    """
    seen = set() if seen is None else seen
    while addr not in seen:
        if addr in targets:
            return addr
        seen.add(addr)
        op = image.peek(addr)
        target = _word(image, addr + 1)
        if op == OP_JMP:
            addr = target
            continue
        if op == OP_JSR:
            found = walk_calls(image, target, targets, seen)
            if found is not None:
                return found
        if op in RETURN_OPS:
            return None
        addr = (addr + OP_LEN[op]) & 0xFFFF
    return None


def track_x(image, target: int, x: int, goal: int, budget: int = 8) -> Tuple[int, bool]:
    """``(x, reaches_goal)``: follow ``target``'s ``INX``/``LDX #n``/``JMP`` chain.

    ``x`` is updated as each instruction would; stops with ``True`` on reaching
    ``goal`` and ``False`` on any other instruction or after ``budget`` steps.
    """
    for _ in range(budget):
        if target == goal:
            return x, True
        op = image.peek(target)
        if op == OP_INX:
            x = (x + 1) & 0xFF
            target = (target + 1) & 0xFFFF
        elif op == OP_LDX_IMM:
            x = image.peek(target + 1)
            target = (target + 2) & 0xFFFF
        elif op == OP_JMP:
            target = _word(image, target + 1)
        else:
            return x, False
    return x, False


def called_x_values(image, start: int, goal: int, budget: int = 16) -> Tuple[int, ...]:
    """The X values at which the code from ``start`` calls ``goal``.

    Walks straight-line code with ``X = 0``; ``LDX #n`` / ``INX`` set X, each
    ``JSR``/``JMP`` resolvable to ``goal`` (through :func:`track_x`) records the
    X it arrives with, a ``JSR`` continues with that X and a ``JMP`` or any
    unrecognised instruction ends the walk.
    """
    out: List[int] = []
    x = 0
    pc = start
    for _ in range(budget):
        op = image.peek(pc)
        if op == OP_LDX_IMM:
            x = image.peek(pc + 1)
            pc = (pc + 2) & 0xFFFF
        elif op == OP_INX:
            x = (x + 1) & 0xFF
            pc += 1
        elif op in (OP_JSR, OP_JMP):
            x, hit = track_x(image, _word(image, pc + 1), x, goal)
            if not hit:
                break
            out.append(x)
            if op == OP_JMP:
                break
            pc = (pc + 3) & 0xFFFF
        else:
            break
    return tuple(out)
