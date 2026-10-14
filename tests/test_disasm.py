"""Tests for the shared image disassembler (the ``pysidtracker disasm`` backend)."""

import pytest

from pysidtracker import (
    MEM_SIZE,
    SidImage,
    disassemble,
    entry_labels,
    image_after_init,
)
from pysidtracker import maincli
from pysidtracker.errors import SidParseError

from .helpers import build_prg, build_psid

# $1000: JMP $1010 / $1010: LDA #$07 / STA $D400 / RTS.
_CODE = bytes([0x4C, 0x10, 0x10, 0xA9, 0x07, 0x8D, 0x00, 0xD4, 0x60])
# $1000: LDA #$41 / STA $2000 / RTS (init writing past the loaded region).
_INIT = bytes([0xA9, 0x41, 0x8D, 0x00, 0x20, 0x60])


def _sid(play=0x1010, image=_CODE):
    return SidImage.from_bytes(build_psid(image, load=0x1000, init=0x1000, play=play))


def _prg(image=_CODE):
    return SidImage.from_bytes(build_prg(image, 0x1000))


def test_entry_labels_from_header():
    assert entry_labels(_sid()) == {"INIT": 0x1000, "PLAY": 0x1010}


def test_entry_labels_collapse_shared_address():
    assert entry_labels(_sid(play=0x1000)) == {"INIT": 0x1000}


def test_entry_labels_empty_for_prg():
    assert entry_labels(_prg()) == {}


def test_disassemble_rows_and_text():
    image = _sid()
    rows = disassemble(image)
    assert [(addr, raw) for addr, raw, _ in rows] == [
        (0x1000, b"\x4c\x10\x10"),
        (0x1003, b"\xa9\x07"),
        (0x1005, b"\x8d\x00\xd4"),
        (0x1008, b"\x60"),
    ]
    assert [text for _, _, text in rows] == [
        "JMP $1010",
        "LDA #$07",
        "STA $d400",
        "RTS",
    ]


def test_disassemble_labels_substitute_operand():
    rows = disassemble(_sid(), labels=entry_labels(_sid()))
    assert rows[0][2] == "JMP PLAY"


def test_disassemble_subrange():
    rows = disassemble(_sid(), 0x1003, 0x1008)
    assert [addr for addr, _, _ in rows] == [0x1003, 0x1005]


def test_disassemble_illegal_opcode_is_real_mnemonic():
    image = _prg()
    image.mem[0x1003] = 0x02  # an NMOS illegal where the operand byte was
    assert disassemble(image, 0x1003, 0x1004)[0][2] == "JAM"


def test_image_after_init_unpacks_and_grows_end():
    image = _sid(image=_INIT)
    assert image.end == 0x1006
    image = image_after_init(image)
    assert image.mem[0x2000] == 0x41
    assert image.end == MEM_SIZE


def test_image_after_init_requires_header():
    pytest.raises(SidParseError, image_after_init, _prg())


def _cli(argv):
    parser = maincli.build_parser([])
    args = parser.parse_args(argv)
    args.func(args)


def test_cli_disasm_defaults_to_loaded_region(tmp_path, capsys):
    path = tmp_path / "t.sid"
    path.write_bytes(build_psid(_CODE, load=0x1000, init=0x1000, play=0x1010))
    _cli(["disasm", str(path)])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("1000  4c 10 10  JMP PLAY")
    assert lines[-1].startswith("1008  60        RTS")


def test_cli_disasm_hex_range_args(tmp_path, capsys):
    path = tmp_path / "t.sid"
    path.write_bytes(build_psid(_CODE, load=0x1000, init=0x1000, play=0x1010))
    _cli(["disasm", str(path), "$1003", "0x1008"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("1003  a9 07     LDA #$07")


def test_cli_disasm_after_init(tmp_path, capsys):
    path = tmp_path / "t.sid"
    path.write_bytes(build_psid(_INIT, load=0x1000, init=0x1000, play=0x1000))
    _cli(["disasm", str(path), "1000", "1006", "--after-init"])
    out = capsys.readouterr().out
    assert "STA $2000" in out


def test_cli_disasm_after_init_on_prg_fails(tmp_path, capsys):
    path = tmp_path / "t.prg"
    path.write_bytes(build_prg(_CODE, 0x1000))
    rc = maincli.run_cli(
        lambda: maincli.build_parser([]),
        maincli.SidError,
        ["disasm", str(path), "--after-init"],
    )
    assert rc == 1
    assert "no SID header" in capsys.readouterr().err


def test_cli_disasm_rejects_bad_address(tmp_path):
    path = tmp_path / "t.prg"
    path.write_bytes(build_prg(_CODE, 0x1000))
    with pytest.raises(SystemExit) as exc:
        maincli.build_parser([]).parse_args(["disasm", str(path), "xyz"])
    assert exc.value.code == 2
