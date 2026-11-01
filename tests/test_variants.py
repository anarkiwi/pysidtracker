"""Tests for the variant fingerprinting, clustering and diffing (issue #54)."""

import pytest

from pysidtracker import (
    build_signature,
    diff_signatures,
    variants,
)
from pysidtracker.asm6502 import assemble
from pysidtracker.errors import SidParseError
from pysidtracker.variants import (
    ROW_FIELDS,
    _op_class,
    variants_cli,
)

from .helpers import build_prg, build_psid_embedded_load

# Direct player: init (+ JSR setup), play (+ JSR calc), a ZP scratch cell and an in-span table.
BASE_SRC = """
init: LDA #$0F
  STA $D418
  JSR setup
  RTS
setup: LDA #$01
  STA zc
  RTS
play: INC step
  JSR calc
  RTS
calc: LDY #$01
  LDA tab,Y
  STA $D400
  RTS
step: .byte 0
zc: .byte 0
tab: .byte 1,2,3,4
"""


def _build(src: str, org: int) -> bytes:
    code, labels = assemble(src, org)
    return build_psid_embedded_load(code, org, init=labels["init"], play=labels["play"])


def _players():
    base = BASE_SRC
    return {
        "base": _build(base, 0x3000),
        "reloc": _build(base, 0x3800),
        "wrapper": _build(base + "dead: LDA #$77\n  RTS\n", 0x3000),
        "inserted": _build(
            base.replace("calc: LDY #$01", "calc: LDY #$01\n  NOP"), 0x3000
        ),
    }


def test_never_run_code_adds_nothing():
    p = _players()
    assert (
        build_signature(p["base"], nframes=4).key
        == build_signature(p["wrapper"], nframes=4).key
    )


def test_bare_prg_is_rejected():
    pytest.raises(SidParseError, build_signature, build_prg(bytes(32), 0x2000))


def test_op_class_symbols_and_relative_offsets():
    base, lo, hi = 0x3000, 0x3000, 0x3040
    assert _op_class(0x0010, base, lo, hi) == "zp"
    assert _op_class(0x3020, base, lo, hi) == "rel:32"
    assert _op_class(0xD404, base, lo, hi) == "io:S0_CTRL"
    assert _op_class(0xD405, base, lo, hi) == "io:S0_AD"
    assert _op_class(0x0314, base, lo, hi) == "vec:IRQ"
    assert _op_class(0xEA31, base, lo, hi) == f"ext:{0xEA31}"
    assert _op_class(0x1000, base, lo, hi) == "ram:4096"


def test_variants_without_refs_leaves_nearest_ref_empty():
    p = _players()
    rows, _ = variants([p["base"], p["reloc"]], [], names=["a", "b"], nframes=4)
    assert all(r["nearest_ref"] == "" and r["ndiff"] == "" for r in rows)


def test_variants_disjoint_ref_leaves_nearest_ref_empty():
    other = _build("init: LDX #$05\n  RTS\nplay: DEX\n  RTS\n", 0x3000)
    rows, _ = variants([_players()["base"]], [("OTHER", other)], names=["a"], nframes=4)
    assert len(rows) == 1
    assert rows[0]["nearest_ref"] == "" and rows[0]["ndiff"] == ""


def test_irq_path_fingerprints_and_diffs():
    import tests.irqhost_tunes as itunes

    base = itunes.build("raster_kernal")
    sig = build_signature(base, nframes=3)
    assert sig.path == "irq" and len(sig.insns) >= 8 and sig.routines == ()

    src, _ = itunes.TUNES["raster_kernal"]
    code, labels = assemble(
        src.replace("irq: INC $D019", "irq: INC $D019\n  NOP"), 0x1000
    )
    mod = build_psid_embedded_load(
        code, 0x1000, init=labels.get("init", 0x1000), play=0
    )
    modsig = build_signature(mod, nframes=3)
    assert sig.jaccard(modsig) < 1.0
    assert any("NOP" in (op[1] or op[2]) for op in diff_signatures(modsig, sig))


def test_variants_cli_writes_csv_and_prints(tmp_path, capsys):
    p = _players()
    paths = [tmp_path / f"{n}.sid" for n in ("base", "reloc", "wrapper", "inserted")]
    ref = tmp_path / "base.ref"
    for path, name in zip(paths, ("base", "reloc", "wrapper", "inserted")):
        path.write_bytes(p[name])
    ref.write_bytes(p["base"])
    out = tmp_path / "out.csv"
    rows = variants_cli(paths, [ref], out=out, nframes=4, threshold=0.8)
    assert len(rows) == 4
    assert out.read_text("utf-8").splitlines()[0].split(",") == ROW_FIELDS
    printed = capsys.readouterr().out
    assert "cluster" in printed and "nearest=base.ref" in printed


def test_decode_lengths_and_zero_page_operands():
    from pysidtracker.variants import _decode  # pylint: disable=import-outside-toplevel

    mem = bytearray(0x10000)
    mem[0x1000:0x1007] = bytes([0xB1, 0xFB, 0xA5, 0x02, 0xAD, 0x00, 0x10])
    assert _decode(mem, 0x1000, 0x1000, 0x1000, 0x1100) == (2, "LDA", "indy", "zp")
    assert _decode(mem, 0x1002, 0x1000, 0x1000, 0x1100) == (2, "LDA", "zp", "zp")
    assert _decode(mem, 0x1004, 0x1000, 0x1000, 0x1100) == (3, "LDA", "abs", "rel:0")


def test_cluster_facets_reports_only_differing_facets():
    from pysidtracker.variants import cluster_facets

    infos = [
        {"driver": 1, "driver.src": "default", "speed": 2},
        {"driver": 1, "driver.src": "detected@$1000", "speed": 3},
        {"driver": 9, "speed": 5},
    ]
    assert cluster_facets([0, 0, 1], infos) == {0: {"speed": [2, 3]}}


_DATA_SRC = """
init: RTS
play: LDX #$00
  LDA tab,X
  STA $D400
  LDA tab2,X
  STA $D401
  RTS
{pad}tab: .byte 1,2,3,4
tab2: .byte 5,6,7,8
"""


def _data_player(pad: int, edit: bool = False) -> bytes:
    src = _DATA_SRC.format(pad="  .byte 0\n" * pad)
    if edit:
        src = src.replace("play: LDX #$00", "play: LDX #$01")
    return _build(src, 0x3000)


def test_data_placement_does_not_change_signature():
    a = build_signature(_data_player(0), nframes=4)
    b = build_signature(_data_player(37), nframes=4)
    assert a.key == b.key
    assert a.jaccard(b) == 1.0
    assert diff_signatures(b, a) == []


def test_code_edit_still_differs_with_data_normalised():
    a = build_signature(_data_player(0), nframes=4)
    b = build_signature(_data_player(37, edit=True), nframes=4)
    assert a.jaccard(b) < 1.0
    ops = diff_signatures(b, a)
    assert [op[0] for op in ops] == ["changed"]


def test_op_class_data_numbered_by_first_use():
    base, lo, hi = 0x3000, 0x3000, 0x3100
    code = {0x3000, 0x3001}
    ids = {}
    assert _op_class(0x3000, base, lo, hi, code, ids) == "rel:0"
    assert _op_class(0x3050, base, lo, hi, code, ids) == "data:0"
    assert _op_class(0x3060, base, lo, hi, code, ids) == "data:1"
    assert _op_class(0x3050, base, lo, hi, code, ids) == "data:0"
    assert _op_class(0x3050, base, lo, hi, code) == "data"
