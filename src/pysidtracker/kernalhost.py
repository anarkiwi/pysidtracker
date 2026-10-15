"""Run a whole C64 program (editor/packer) headlessly, KERNAL file I/O on the host.

The host is :class:`vagrant_moses.kernal.KernalHost`, servicing the real KERNAL
jump table (:data:`JUMP_TABLE`) and register ABI; this module adds
:class:`~pysidtracker.image.SidImage` input and SidError-based exceptions.
"""

from __future__ import annotations

import functools

from vagrant_moses import kernal as _kernal
from vagrant_moses.kernal import JUMP_TABLE

from .errors import KernalHostError
from .image import SidImage


def _sid_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (_kernal.KernalHostError, ValueError) as exc:
            raise KernalHostError(*exc.args) from exc

    return wrapper


@_sid_errors
def sys_addr(mem, base: int = 0x0801) -> int:
    """The first numeric ``SYS`` operand in the BASIC program at ``base``.

    ``mem`` is a 64 KiB address space or a :class:`SidImage`.

    Raises:
      KernalHostError: no line in the chain holds a numeric ``SYS``.
    """
    return _kernal.sys_addr(mem.mem if isinstance(mem, SidImage) else mem, base)


class KernalHost(_kernal.KernalHost):
    """:class:`vagrant_moses.kernal.KernalHost` raising :class:`KernalHostError`."""

    run = _sid_errors(_kernal.KernalHost.run)
    run_to_unpacked = _sid_errors(_kernal.KernalHost.run_to_unpacked)


__all__ = ["JUMP_TABLE", "KernalHost", "sys_addr"]
