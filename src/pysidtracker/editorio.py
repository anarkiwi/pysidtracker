"""Locate KERNAL file/keyboard call sites in an image and disassemble their setup.

Finds ``JSR $FFD5`` (LOAD), ``JSR $FFD8`` (SAVE) and ``JSR $FFC0`` (OPEN) and the
``SETLFS``/``SETNAM``/register loads each makes -- the offline companion to
:mod:`pysidtracker.kernalhost`, which runs the editor headlessly.
"""

from __future__ import annotations

from typing import List, Tuple

from .disasm import disassemble
from .image import SidImage

# (name, the 3-byte JSR of the KERNAL call site).
_CALL_SITES = (
    ("LOAD", b"\x20\xd5\xff"),
    ("SAVE", b"\x20\xd8\xff"),
    ("OPEN", b"\x20\xc0\xff"),
)


def editor_io(image: SidImage, setup_bytes: int = 24) -> List[Tuple[str, int, List]]:
    """Locate LOAD/SAVE/OPEN calls and return the disassembled setup code before each.

    Scans ``image`` (from its load address) for each call-site ``JSR``; ``rows`` are
    the ``disassemble`` ``(address, raw, text)`` tuples of the preceding ``setup_bytes``.
    """
    found: List[Tuple[str, int, List]] = []
    for name, needle in _CALL_SITES:
        for address in image.find_all(needle):
            start = max(0, address - setup_bytes)
            found.append((name, address, disassemble(image, start, address + 3)))
    found.sort(key=lambda item: item[1])
    return found


__all__ = ["editor_io"]
