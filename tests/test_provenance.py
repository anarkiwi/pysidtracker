"""Tests for provenance: per-write load chains and the per-frame RAM watch."""

import csv
import hashlib
import struct

import numpy as np
import pytest

from pysidtracker import provenance as prov
from pysidtracker.asm6502 import assemble
from pysidtracker.errors import SidParseError
from pysidtracker.image import SidImage
from pysidtracker import survey

ORG = 0x1000

TOY_TALLY = (
    "init:\n  SEC\n  LDX #0\n  RTS\n"
    "play:\n  LDA tbl,X\n  ADC cell\n  STA $D405\n  INC cell\n  RTS\n"
    "tbl:\n  .byte $10\ncell:\n  .byte $00\n"
)


def _psid(org, init, play, code):
    off = 0xC0
    h = bytearray(off)
    h[0:4] = b"PSID"
    struct.pack_into(">H", h, 0x04, 2)
    struct.pack_into(">H", h, 0x06, off)
    struct.pack_into(">H", h, 0x08, org)
    struct.pack_into(">H", h, 0x0A, init)
    struct.pack_into(">H", h, 0x0C, play)
    struct.pack_into(">H", h, 0x0E, 1)
    struct.pack_into(">H", h, 0x10, 1)
    return bytes(h) + code


def _toy(src):
    code, labels = assemble(src, ORG)
    sid = _psid(ORG, labels["init"], labels["play"], code)
    return SidImage.from_sid(sid), labels, sid


def _decode(ops, x=0, y=0, pc=0x10, seed=()):
    from jennings.devices.mpu6502 import MPU

    mem = bytearray(0x100)
    mem[pc : pc + len(ops)] = ops
    for off, val in seed:
        mem[off] = val
    mpu = MPU(memory=mem)
    mpu.x, mpu.y, mpu.pc = x, y, pc
    return prov.decode_insn(pc, mpu, mem)


# --- addressing decode ------------------------------------------------------


@pytest.mark.parametrize(
    "ops,x,y,base,eff,idx",
    [
        (bytes((0xB5, 0x12)), 5, 0, 0x12, 0x17, "X"),  # LDA zp,X
        (bytes((0xB6, 0x12)), 0, 7, 0x12, 0x19, "Y"),  # LDX zp,Y
        (bytes((0xBD, 0x34, 0x12)), 0x64, 0, 0x1234, 0x1298, "X"),  # LDA abs,X
        (bytes((0xBE, 0x34, 0x12)), 0, 5, 0x1234, 0x1239, "Y"),  # LDX abs,Y
    ],
)
def test_decode_indexed(ops, x, y, base, eff, idx):
    ins = _decode(ops, x=x, y=y)
    assert ins.base == base
    assert ins.effective == eff
    assert ins.index == idx
    assert ins.index_value == (x if idx == "X" else y)


def test_decode_indirect_indexed():
    ins = _decode(
        bytes((0xA1, 0x20)), x=2, seed=((0x22, 0x11), (0x23, 0x22))
    )  # LDA ($20),X
    assert ins.mnemonic == "LDA" and ins.index == "X" and ins.effective == 0x2211
    ins = _decode(
        bytes((0xB1, 0x20)), y=5, seed=((0x20, 0x00), (0x21, 0x12))
    )  # LDA ($20),Y
    assert ins.mnemonic == "LDA" and ins.index == "Y" and ins.effective == 0x1205


def test_written_regs_classification():
    assert _decode(bytes((0xA5, 0x00))).writes == frozenset({"A"})  # LDA zp
    assert _decode(bytes((0xA6, 0x00))).writes == frozenset({"X"})  # LDX zp
    assert _decode(bytes((0xA4, 0x00))).writes == frozenset({"Y"})  # LDY zp
    assert _decode(bytes((0xAF, 0x00, 0x10))).writes == frozenset({"A", "X"})  # LAX abs
    assert _decode(bytes((0xE8,))).writes == frozenset({"X"})  # INX
    asl_acc = _decode(bytes((0x0A,)))  # ASL a
    asl_zp = _decode(bytes((0x06, 0x00)))
    assert asl_acc.writes == frozenset({"A"}) and "A" not in asl_zp.writes


# --- load chains ------------------------------------------------------------


def test_chain_entry_alive_from_call():
    img, _, _ = _toy("init:\n  RTS\nplay:\n  STA $D405\n  RTS\n")
    pro = prov.trace_frames(img, 1, labels={})
    w = prov.explain(pro, 0, reg=5)[0]
    assert w.value == 0 and [s.kind for s in w.chain] == ["entry"]
    assert "<A on call entry>" in w.chain[0].mnemonic


def test_chain_register_transfer():
    img, labels, _ = _toy(
        "init:\n  RTS\nplay:\n  LDA cell\n  TAX\n  STX $D405\n  RTS\ncell:\n  .byte $33\n"
    )
    pro = prov.trace_frames(img, 1, labels=labels)
    w = prov.explain(pro, 0, reg=5)[0]
    assert w.value == 0x33 and [s.kind for s in w.chain] == ["xfer", "mem"]
    assert w.chain[1].source == labels["cell"]


def test_chain_inplace_arithmetic():
    img, _, _ = _toy(
        "init:\n  RTS\nplay:\n  LDA cell\n  ASL A\n  STA $D405\n  RTS\ncell:\n  .byte $33\n"
    )
    pro = prov.trace_frames(img, 1, labels={})
    w = prov.explain(pro, 0, reg=5)[0]
    assert w.value == 0x66 and [s.kind for s in w.chain] == ["inplace", "mem"]


def test_explain_reg_filter_and_bounds():
    img, labels, _ = _toy(TOY_TALLY)
    pro = prov.trace_frames(img, 2, labels=labels)
    assert [w.register for w in prov.explain(pro, 1)] == [5]
    assert prov.explain(pro, 1, reg=4) == []
    assert prov.explain(pro, 1, reg=5)
    with pytest.raises(SidParseError):
        prov.explain(pro, 99)
    with pytest.raises(SidParseError):
        prov.explain(pro, -1)


def test_trace_requires_header():
    prg = SidImage.from_prg(bytes((0x00, 0x10)) + bytes((0xEA, 0x60)))
    with pytest.raises(SidParseError):
        prov.trace_frames(prg, 1)


def test_deep_chain_bounded_by_depth():
    # A value folded through more ALU hops than the depth cap stops back-tracking.
    img, _, _ = _toy(
        "init:\n  SEC\n  RTS\n"
        "play:\n  LDA #1\n  ADC #1\n  ADC #1\n  ADC #1\n  ADC #1\n  ADC #1\n"
        "  STA $D405\n  RTS\n"
    )
    pro = prov.trace_frames(img, 1, labels={})
    deep = prov.explain(pro, 0, reg=5, max_depth=2)[0]
    assert all(s.kind in ("alu", "imm", "entry") for s in deep.chain)
    assert len(deep.chain) <= 3


# --- watch grid + formatting ------------------------------------------------


# --- survey explain CLI -----------------------------------------------------


def render_toy(data, song, nframes):
    assert data.startswith(b"PSID") and int(song) >= 1
    grid = np.zeros((nframes, 25), np.int16)
    grid[:, 5] = 0x20 + np.arange(nframes)
    grid[:, 24] = 15
    return grid, {"variant": "toy", "size": len(data)}


def _survey_tree(tmp_path, sid):
    hvsc = tmp_path / "hvsc"
    hvsc.mkdir()
    (hvsc / "t.sid").write_bytes(sid)
    md5 = hashlib.md5(sid).hexdigest()
    tunes = tmp_path / "tunes.csv"
    with open(tunes, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, survey.TUNE_FIELDS)
        w.writeheader()
        w.writerow(
            dict(
                path="t.sid", md5=md5, player="toy", songs="1", clock="PAL", lengths="5"
            )
        )
    return hvsc, tunes, f"{md5}_1"


def test_survey_explain_cli(tmp_path, capsys):
    _, labels, sid = _toy(TOY_TALLY)
    hvsc, tunes, key = _survey_tree(tmp_path, sid)
    r = survey.main(
        [
            "explain",
            str(hvsc),
            str(tunes),
            key,
            "--work",
            str(tmp_path / "work"),
            "--frame",
            "1",
            "--reg",
            "5",
            "--watch",
            "$%04X" % labels["cell"],
            "--render",
            f"{__name__}:render_toy",
        ],
    )
    assert r == 0
    out = capsys.readouterr().out
    assert "STA $d405" in out and "LDA $1011,X" in out and "ADC $1012" in out
    assert "value $11" in out
    assert "library (frame 1, reg 5): $21" in out and "real $11" in out
    assert "000002" in out  # watched cell value at frame 1


def test_survey_explain_no_render_needed(tmp_path, capsys):
    # --render is optional; without it no library value is printed.
    _, _, sid = _toy("init:\n  RTS\nplay:\n  LDA #9\n  STA $D405\n  RTS\n")
    hvsc, tunes, key = _survey_tree(tmp_path, sid)
    pro = survey.explain(str(hvsc), tunes, key, tmp_path / "work", frame=0, reg=5)
    out = capsys.readouterr().out
    assert "STA $d405" in out and "value $09" in out and "library" not in out
    assert len(pro.frames[0]) == 3  # the LDA, STA, RTS were recorded
