"""Normalise the many ways a caller can hand a tune to a parser.

Every parser in the family accepts a path, a ``bytes`` blob, or an open binary
file. :func:`read_bytes` is the single dispatch they share so ``read(...)``
behaves identically everywhere.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Union

Source = Union[bytes, bytearray, str, "os.PathLike[str]", io.IOBase]


def read_bytes(src: Source) -> bytes:
    """Return the raw bytes of ``src``.

    ``src`` may be ``bytes``/``bytearray``, a filesystem path (``str`` or
    :class:`os.PathLike`), or a binary file-like object with a ``read`` method.
    Raises :class:`TypeError` for anything else.
    """
    if isinstance(src, (bytes, bytearray)):
        return bytes(src)
    if isinstance(src, (str, os.PathLike)):
        return Path(src).read_bytes()
    if isinstance(src, io.IOBase) or hasattr(src, "read"):
        return src.read()
    raise TypeError(f"cannot read a tune from {type(src).__name__}")
