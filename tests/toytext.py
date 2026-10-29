"""A toy typed model for the text-grammar runtime tests."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Stop:
    """End the program."""


@dataclass(frozen=True)
class Jump:
    """Continue at ``target``, a 16-bit address."""

    target: int

    def __post_init__(self):
        if not -0x8000 <= self.target <= 0xFFFF:
            raise ValueError(f"jump target {self.target} out of range")


@dataclass(frozen=True)
class WaveNote:
    """A note, absolute or ``relative`` to the played one."""

    note: int
    relative: bool = False


@dataclass(frozen=True)
class Step:
    """An optional jump and an optional note."""

    jump: Optional[Jump]
    note: Optional[WaveNote]
    seen: int = field(default=0, init=False, compare=False)


@dataclass(frozen=True)
class Named:
    """A label."""

    name: str
