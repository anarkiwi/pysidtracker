"""Check the streams a spec reads against the ``ByteGrammar`` rules it names.

A stream's ``grammar "module:attribute:rule"`` names a
:class:`~pysidtracker.bytegrammar.ByteGrammar`; execution always reads memory,
and the grammar is a static view used only to report malformed data.
"""

from __future__ import annotations

import importlib
from typing import Callable, Dict, Optional, Sequence, Tuple

from ..bytegrammar import REACH
from ..errors import SidError
from .ir import Program


def resolve(ref: str):
    """``(ByteGrammar, rule)`` of a ``module:attribute:rule`` reference."""
    module, attribute, rule = ref.split(":")
    return getattr(importlib.import_module(module), attribute), rule


class StreamValidator:
    """Parses each distinct ``(grammar, base)`` stream once; :attr:`results` holds
    ``None`` (well-formed) or the parse error, by ``(reference, base)``."""

    def __init__(self, strict: bool = False):
        self.strict = strict
        self.results: Dict[Tuple[str, int], Optional[str]] = {}
        self._grammars: Dict[str, tuple] = {}

    def check(
        self,
        ref: str,
        base: int,
        read: Callable[[int], int],
        array: Optional[Tuple[str, Sequence[int]]] = None,
    ) -> Optional[str]:
        """Parse the stream at ``base`` as ``ref``: bytes from ``read``, or from index
        ``base`` of ``array`` (its name and items), where a short tail fails to parse.
        """
        where = f"${base:04x}" if array is None else f"{array[0]}[{base}]"
        key = (ref if array is None else f"{ref} in {array[0]}", base)
        if key in self.results:
            return self.results[key]
        if ref not in self._grammars:
            self._grammars[ref] = resolve(ref)
        grammar, rule = self._grammars[ref]
        if array is None:
            data = bytes(read((base + i) & 0xFFFF) for i in range(REACH))
        else:
            data = bytes(array[1][base : base + REACH])
        try:
            grammar.parse_prefix(data, rule)
            message = None
        except SidError as exc:
            message = str(exc)
        self.results[key] = message
        if message is not None and self.strict:
            raise SidError(f"stream at {where} is not {ref}: {message}")
        return message

    def hook(
        self, program: Program, read: Callable[[int], int]
    ) -> Callable[[int, int], None]:
        """The ``G(index, base)`` callback a compiled frame calls as a stream opens."""
        refs = [ref for ref, _line in program.grammars]

        def _hook(index: int, base: int) -> None:
            self.check(refs[index], base, read, program.grammar_arrays.get(index))

        return _hook

    def failures(self) -> Dict[Tuple[str, int], str]:
        """The streams that did not parse."""
        return {k: v for k, v in self.results.items() if v is not None}


__all__ = ["StreamValidator", "resolve"]
