"""Relocate 6502 player code: find the instructions it runs, move its operands.

:func:`walk_code` is a recursive-descent code walk; :func:`relocate_operands`
retargets the absolute operands of the walked code for a moved region;
:func:`store_stub` and :func:`init_fixup` emit the init code of a relocated image.
"""

from __future__ import annotations

from typing import Container, Dict, Iterable, List, Mapping, Optional, Tuple

from jennings.opcodes import OPCODES

from .errors import SidParseError
from .mos6502 import BRANCH_OPS, OP_JMP, OP_JSR, OP_LEN, RETURN_OPS, s8

JAM_OPS = frozenset(op for op in range(256) if OPCODES[op][0] == "JAM")
STOP_OPS = RETURN_OPS | JAM_OPS | {OP_JMP}
FORCED_BRANCHES = {0x90: 0x18, 0xB0: 0x38, 0x50: 0xB8}


def _word(image, addr: int) -> int:
    return image.peek(addr) | (image.peek(addr + 1) << 8)


def walk_code(image, starts: Iterable[int], lo: int, hi: int) -> Dict[int, int]:
    """``{address: length}`` of each instruction the code at ``starts`` reaches
    in ``[lo, hi)``.

    ``JSR``/``JMP``/branch targets are queued; a branch its flag-setting
    predecessor forces (``CLC; BCC``, ``SEC; BCS``, ``CLV; BVC``) and
    ``JMP``/``RTS``/``RTI``/``BRK``/``JMP ()``/``JAM`` end a path.
    """
    code: Dict[int, int] = {}
    todo = list(starts)
    while todo:
        pc, last = todo.pop(), None
        while lo <= pc < hi and pc not in code:
            op = image.peek(pc)
            code[pc] = OP_LEN[op]
            if op in (OP_JSR, OP_JMP):
                todo.append(_word(image, pc + 1))
            elif op in BRANCH_OPS:
                todo.append((pc + 2 + s8(image.peek(pc + 1))) & 0xFFFF)
                if last is not None and FORCED_BRANCHES.get(op) == last:
                    break
            if op in STOP_OPS:
                break
            pc, last = pc + OP_LEN[op], op
    return code


def relocate_operands(
    image,
    code: Mapping[int, int],
    moved: range,
    delta: int,
    remap: Optional[Mapping[int, int]] = None,
    forbidden: Container[int] = (),
    error: type = SidParseError,
) -> Dict[int, int]:
    """``{operand address: new target}`` of the absolute operands of ``code``
    that change.

    A target in ``moved`` shifts by ``delta``, one in ``remap`` is replaced;
    a target in ``forbidden`` that neither covers raises ``error``.
    """
    remap = remap or {}
    out: Dict[int, int] = {}
    for pc in sorted(pc for pc, size in code.items() if size == 3):
        target = _word(image, pc + 1)
        if target in moved:
            out[pc + 1] = (target + delta) & 0xFFFF
        elif target in remap:
            out[pc + 1] = remap[target] & 0xFFFF
        elif target in forbidden:
            raise error(
                f"operand at {pc + 1:#06x} reaches {target:#06x}, "
                "data the relocation does not place"
            )
    return out


def store_stub(writes: Iterable[Tuple[int, int]]) -> bytes:
    """6502 code storing each ``(addr, value)`` of ``writes`` in order, then ``RTS``.

    ``LDA #value`` is emitted only when the value changes; a zero-page address
    takes ``STA zp``, any other ``STA abs``.
    """
    code: List[int] = []
    a = None
    for addr, v in writes:
        if v != a:
            code += [0xA9, v]
            a = v
        code += [0x85, addr] if addr < 0x100 else [0x8D, addr & 0xFF, addr >> 8]
    return bytes([*code, 0x60])


def init_fixup(
    init: int,
    adds: Iterable[Tuple[int, int]],
    copies: Iterable[Tuple[int, int, int]],
    writes: Iterable[Tuple[int, int]] = (),
) -> bytes:
    """6502 code calling ``init``, then ``CLD``, then adding each ``(addr, value)`` of
    ``adds`` in place (``CLC; ADC``), then making each ``(src, dst, n)`` copy of
    ``copies`` (``LDX #n``, then ``src - 1 + X`` to ``dst - 1 + X`` down to ``X = 1``),
    then the :func:`store_stub` of ``writes``.

    A copy of ``1 <= n <= 255`` moves ``src .. src + n - 1``; ``n`` = 256 loads
    ``X = 0``, which moves ``src - 1 .. src + 254`` instead.
    """
    code = bytearray([0x20, init & 0xFF, init >> 8, 0xD8])
    for a, v in adds:
        at = [a] if a < 0x100 else [a & 0xFF, a >> 8]
        code += bytes([0xA5 if a < 0x100 else 0xAD, *at, 0x18, 0x69, v])
        code += bytes([0x85 if a < 0x100 else 0x8D, *at])
    for src, dst, n in copies:
        src, dst = src - 1, dst - 1
        code += bytes([0xA2, n & 0xFF, 0xBD, src & 0xFF, src >> 8])
        code += bytes([0x9D, dst & 0xFF, dst >> 8, 0xCA, 0xD0, 0xF7])
    return bytes(code) + store_stub(writes)


__all__ = [
    "FORCED_BRANCHES",
    "JAM_OPS",
    "STOP_OPS",
    "init_fixup",
    "relocate_operands",
    "store_stub",
    "walk_code",
]
