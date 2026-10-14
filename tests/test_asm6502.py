"""The two-pass assembler used to build the PSID driver and synthetic tunes."""

import pytest

from pysidtracker.asm6502 import assemble


def test_labels_resolve_forwards_and_backwards():
    code, labels = assemble(
        """
start: LDA #<data   ; low byte
  LDX #>data
loop: DEX
  BNE loop
  JMP (vector)
  LDA data,X
  ASL A
  STA $01
data: .byte 1, $FF
vector: .word start
""",
        0x1000,
    )
    assert labels == {"start": 0x1000, "loop": 0x1004, "data": 0x1010, "vector": 0x1012}
    assert code == bytes.fromhex("a910a210cad0fd6c1210bd10100a850101ff0010")


@pytest.mark.parametrize(
    "src, exc",
    [("FOO $10", ValueError), ("LDA", ValueError), ("JMP nowhere", KeyError)],
)
def test_bad_statements_are_rejected(src, exc):
    with pytest.raises(exc):
        assemble(src, 0)


def test_unassemblable_operand_is_rejected():
    with pytest.raises(ValueError, match="cannot assemble"):
        assemble("LDX $1000,X", 0)
