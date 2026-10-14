"""A two-pass 6502 assembler with labels, on jennings' single-statement assembler.

One statement per line; ``label:`` prefixes, ``;`` comments, ``.byte``/``.word``
directives and ``<label``/``>label`` low/high bytes are supported.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

_LABEL = re.compile(r"(?<![$\w])([<>]?)([A-Za-z_]\w*)")
_MNEMONIC = re.compile(r"^[A-Za-z]{3}$")


def _assembler():
    from jennings.assembler import Assembler
    from jennings.devices.mpu6502 import MPU

    return Assembler(MPU())


def _value(token: str, labels: Dict[str, int]) -> int:
    token = token.strip()
    match = _LABEL.fullmatch(token)
    if match:
        value = labels.get(match.group(2), 0xFFFF)
        return {"<": value & 0xFF, ">": value >> 8}.get(match.group(1), value)
    return int(token[1:], 16) if token.startswith("$") else int(token, 0)


def _substitute(operand: str, labels: Dict[str, int], fill: bool) -> str:
    def repl(match):
        name = match.group(2)
        if name.upper() in ("A", "X", "Y") and not match.group(1):
            return match.group(0)
        value = labels.get(name, 0xFFFF if fill else None)
        if value is None:
            raise KeyError(f"undefined label {name!r}")
        if match.group(1):
            return f"${(value >> 8 if match.group(1) == '>' else value) & 0xFF:02X}"
        return f"${value:04X}"

    return _LABEL.sub(repl, operand)


def _pass(lines: List[str], org: int, labels: Dict[str, int], final: bool):
    asm, pc, out = _assembler(), org, bytearray()
    for raw in lines:
        line = raw.split(";", 1)[0].strip()
        while ":" in line.split(" ", 1)[0]:
            name, line = line.split(":", 1)
            labels[name.strip()] = pc
            line = line.strip()
        if not line:
            continue
        head, _, rest = line.partition(" ")
        if head.lower() in (".byte", ".word"):
            size = 1 if head.lower() == ".byte" else 2
            for tok in rest.split(","):
                value = _value(tok, labels)
                out += (
                    value.to_bytes(2, "little") if size == 2 else bytes((value & 0xFF,))
                )
            pc = org + len(out)
            continue
        if not _MNEMONIC.match(head):
            raise ValueError(f"bad statement {raw!r}")
        stmt = f"{head} {_substitute(rest, labels, not final)}".strip()
        try:
            code = asm.assemble(stmt, pc)
        except SyntaxError as exc:
            raise ValueError(f"cannot assemble {raw!r}") from exc
        out += bytes(code)
        pc = org + len(out)
    return bytes(out)


def assemble(source: str, org: int) -> Tuple[bytes, Dict[str, int]]:
    """Assemble ``source`` at ``org``; return ``(code, labels)``."""
    lines = source.splitlines()
    labels: Dict[str, int] = {}
    _pass(lines, org, labels, False)
    code = _pass(lines, org, labels, True)
    return code, labels


__all__ = ["assemble"]
