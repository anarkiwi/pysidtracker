"""Relocate 6502 player code: find the instructions it runs, move its operands.

:func:`walk_code` is a recursive-descent code walk; :func:`relocate_operands`
retargets the absolute operands of the walked code for a moved region.
"""

from __future__ import annotations

from typing import Container, Dict, Iterable, Mapping, Optional

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


__all__ = [
    "FORCED_BRANCHES",
    "JAM_OPS",
    "STOP_OPS",
    "relocate_operands",
    "walk_code",
]
