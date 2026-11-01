from pysidtracker.asm6502 import assemble
from pysidtracker.reference import (
    Reference,
    best_reference,
    find_reference,
    match_at,
    reference_pattern,
)

SRC = """
start: lda #$00
       sta $fb
       lda tbl,x
       lda #$07
       sta $d400
       jmp start
       .byte $10,$20
tbl:   .byte $01,$02,$03
"""


def build(org, data_byte=0x10):
    code, labels = assemble(SRC, org)
    code = bytearray(code)
    code[labels["tbl"] - org - 2] = data_byte
    data = ((labels["tbl"] - org - 2, len(code)),)
    return Reference("p", bytes(code), org, labels, data)


def place(ref, addr):
    mem = bytearray(0x10000)
    mem[addr : addr + len(ref.code)] = ref.code
    return mem


def relocated(ref, to):
    code, labels = assemble(SRC, to)
    code = bytearray(code)
    code[ref.data[0][0] :] = ref.code[ref.data[0][0] :]
    return Reference(ref.name, bytes(code), to, labels, ref.data)


def test_self_modified_operand_wild():
    ref = Reference("sm", bytes.fromhex("8D051060A907"), 0x1000)
    assert ref.self_modified() == [5]
    assert reference_pattern(ref).spec == "8D ?? ?? 60 A9 ??"


def test_relocated_match_and_bases():
    ref = build(0x1000)
    mem = place(relocated(ref, 0x2345), 0x2345)
    match = match_at(ref, mem, 0x2345)
    assert match is not None
    assert match.bases["tbl"] == 0x2345 + ref.labels["tbl"] - ref.org
    assert match_at(ref, mem, 0x2346) is None


def test_bad_relocation_rejected():
    ref = build(0x1000)
    mem = place(relocated(ref, 0x2000), 0x2000)
    mem[0x2000 + 6] ^= 0x01
    assert match_at(ref, mem, 0x2000) is None
    assert match_at(ref, mem[:0x2003], 0x2000) is None


def test_tie_break_by_data():
    a = build(0x1000, 0x10)
    b = Reference("q", build(0x1000, 0x99).code, 0x1000, a.labels, a.data)
    mem = place(relocated(b, 0x3000), 0x3000)
    found = find_reference(mem, [a, b])
    assert [m.name for m in found] == ["q", "p"]
    assert best_reference(mem, [a, b]).name == "q"
    assert best_reference(bytes(0x100), [a]) is None
