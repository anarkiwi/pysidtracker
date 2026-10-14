"""Disassemble a ``.sid``/``.prg`` image range (the shared reverse-engineering aid).

On jennings' disassembler -- which decodes the NMOS illegal opcodes the
packed/crunched playroutines lean on as real mnemonics -- over the image's own
64 KiB memory, with the SID header's init/play entries named as labels. Every
dependent repo's per-repo disassembler script reduces to this.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .detect import resolve_entry_points, run_init
from .errors import EmulatorUnavailable
from .image import MEM_SIZE, SidImage

Row = Tuple[int, bytes, str]


def entry_labels(image: SidImage) -> Dict[str, int]:
    """The ``{"INIT": addr, "PLAY": addr}`` label map for ``image``.

    Empty for a bare ``.prg`` (no header to name an entry from); the two
    entries collapse to one label when they share an address.
    """
    if image.header is None:
        return {}
    init, play = resolve_entry_points(image.header, image.load)
    labels: Dict[str, int] = {"INIT": init}
    if play != init:
        labels["PLAY"] = play
    return labels


def disassemble(
    image: SidImage,
    start: Optional[int] = None,
    end: Optional[int] = None,
    labels: Optional[Dict[str, int]] = None,
) -> List[Row]:
    """One ``(address, rawbytes, text)`` row per instruction in ``[start, end)``.

    ``start``/``end`` default to the image's loaded region. ``labels`` (e.g.
    :func:`entry_labels`) are shown in an operand in place of the labelled
    address. Raises :class:`EmulatorUnavailable` if ``jennings`` is missing.
    """
    try:
        from jennings.devices.mpu6502 import MPU
        from jennings.disassembler import Disassembler
        from jennings.utils.addressing import AddressParser
    except ImportError as exc:  # pragma: no cover - jennings is a core dependency
        raise EmulatorUnavailable(
            "jennings is required to disassemble: pip install pysidtracker"
        ) from exc

    start = image.load if start is None else start
    end = image.end if end is None else end
    dis = Disassembler(MPU(memory=image.mem), AddressParser(labels=dict(labels or {})))
    rows: List[Row] = []
    pc = start
    while pc < end:
        length, text = dis.instruction_at(pc)
        rows.append((pc, bytes(image.mem[pc : pc + length]), text))
        pc += length
    return rows


def image_after_init(image: SidImage, *, subtune: int = 0) -> SidImage:
    """``image`` with its init routine run and ``end`` grown over all of memory.

    init may decompress or scatter data anywhere, so the loaded region no
    longer bounds the tune. A bare ``.prg`` has no init to run (raises
    :class:`~pysidtracker.errors.SidParseError`), as does a missing ``jennings``.
    """
    run_init(image, subtune=subtune)
    image.end = MEM_SIZE
    return image


__all__ = ["Row", "disassemble", "entry_labels", "image_after_init"]
