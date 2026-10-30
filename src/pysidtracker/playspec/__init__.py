"""playspec: SID playroutines as synchronous state machines over memory cells.

Parse (:func:`parse`/:func:`load`), check and lower (:func:`elaborate`), run
(:class:`SpecPlayer`, :func:`compile_program`) and prove (:mod:`.smt`, :mod:`.tv`).
"""

from .ir import Cell, K, Op, Program, Rd, V
from .lower import elaborate
from .player import SpecPlayer
from .pyexec import SpecRuntimeError, compile_program, run
from .syntax import Spec, SpecError, load, parse
from .unnative import equivalent, unnative
from .validate import StreamValidator

__all__ = [
    "Cell",
    "K",
    "Op",
    "Program",
    "Rd",
    "Spec",
    "SpecError",
    "SpecPlayer",
    "SpecRuntimeError",
    "StreamValidator",
    "V",
    "compile_program",
    "elaborate",
    "equivalent",
    "load",
    "parse",
    "run",
    "unnative",
]
