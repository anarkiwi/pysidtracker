"""The headless KERNAL host: real jump table and register ABI, files under a host dir.

Real 6502 programs (:func:`asm6502.assemble`) call the KERNAL entries as a C64
program does; the calls are answered in Python against ``tmp_path``.
"""

import pytest
from vagrant_moses import kernal as vm_kernal

from pysidtracker import KernalHost, KernalHostError, SidError, SidImage, sys_addr
from pysidtracker import editor_io
from pysidtracker.asm6502 import assemble

from .helpers import build_prg

DONE = 0x1701
SETUP = """
  LDA #4
  LDX #<name
  LDY #>name
  JSR $FFBD
  LDA #1
  LDX #8
  LDY #{sa}
  JSR $FFBA
"""


def _run(tmp_path, body, sa=0, name=b"WORK", **kwargs):
    """Run ``body`` after SETNAM ``name`` / SETLFS 1,8,``sa``; return the host."""
    text = SETUP.format(sa=sa) + body + "\n  INC $1701\nloop: JMP loop\nname: .byte "
    code, _ = assemble(text + ",".join(f"${b:02X}" for b in name), 0x0801)
    host = KernalHost(tmp_path, **kwargs)
    host.load_prg(build_prg(bytes(code), 0x0801))
    return host.run(until=lambda h: h.ram[DONE], max_steps=20000)


def test_save_and_load_round_trip_through_the_host_dir(tmp_path):
    payload = bytes(range(256))
    program = """
  LDA #$00
  STA $FB
  LDA #$10
  STA $FC
  LDA #$FB
  LDX #$00
  LDY #$11
  JSR $FFD8
  LDA #0
  LDX #$00
  LDY #$30
  JSR $FFD5
  STX $1702
  STY $1703
"""
    text = SETUP.format(sa=0) + program + "\n  INC $1701\nloop: JMP loop\nname: .byte "
    code, _ = assemble(text + "$57,$4F,$52,$4B", 0x0801)
    host = KernalHost(tmp_path)
    host.ram[0x1000:0x1100] = payload
    host.load_prg(build_prg(bytes(code), 0x0801))
    host.run(until=lambda h: h.ram[DONE], max_steps=20000)
    assert (tmp_path / "8" / "WORK").read_bytes() == b"\x00\x10" + payload
    assert host.ram[0x3000:0x3100] == payload
    assert host.ram[0x1702:0x1704] == b"\x00\x31"  # X/Y: the end address


def test_open_chkout_chrout_close_then_chkin_chrin_reads_back(tmp_path):
    _run(
        tmp_path,
        """
  JSR $FFC0
  LDX #1
  JSR $FFC9
  LDA #$41
  JSR $FFD2
  LDA #$42
  JSR $FFD2
  JSR $FFCC
  LDA #1
  JSR $FFC3
""",
        sa=1,
    )
    assert (tmp_path / "8" / "WORK").read_bytes() == b"AB"
    host = _run(
        tmp_path,
        """
  JSR $FFC0
  LDX #1
  JSR $FFC6
  JSR $FFCF
  STA $1702
  JSR $FFCF
  STA $1703
  JSR $FFB7
  STA $1704
  JSR $FFE7
""",
    )
    assert host.ram[0x1702:0x1704] == b"AB"
    assert host.ram[0x1704] & vm_kernal.ST_EOF


def test_getin_reads_the_tapped_keys(tmp_path):
    host = KernalHost(tmp_path)
    assert host.tap_key(ord("A")) is host
    code, _ = assemble("JSR $FFE4\nSTA $1702\nINC $1701\nloop: JMP loop", 0x0801)
    host.load_prg(build_prg(bytes(code), 0x0801))
    host.run(until=lambda h: h.ram[DONE], max_steps=1000)
    assert host.ram[0x1702] == ord("A")
    assert len(host.screen()) == 1024


def test_a_missing_file_sets_carry_and_its_error(tmp_path):
    host = _run(tmp_path, "JSR $FFC0\nBCC ok\nSTA $1702\nok:", name=b"NONE")
    assert host.ram[0x1702] == vm_kernal.FILE_NOT_FOUND


def test_untrapped_jsr_and_a_banked_out_kernal_run_on_the_cpu(tmp_path):
    code, _ = assemble(
        "JSR $1000\nLDA #$35\nSTA $01\nJSR $FFE4\nINC $1701\nloop: JMP loop", 0x0801
    )
    host = KernalHost(tmp_path).tap_key(1)
    host.ram[0x1000] = 0x60  # RTS
    host.ram[0xFFE4] = 0x60  # the RAM under the KERNAL entry
    host.load_prg(build_prg(bytes(code), 0x0801))
    host.run(until=lambda h: h.ram[DONE], max_steps=1000)
    assert list(host.keys) == [1]  # GETIN was not trapped


def test_given_roms_are_banked_in(tmp_path):
    host = KernalHost(
        tmp_path, roms={0xE000: b"\xc0" * 0x2000, 0xA000: b"\x11" * 0x2000}
    )
    assert (host.bus[0xE000], host.bus[0xA000]) == (0xC0, 0x11)


def test_run_to_unpacked_returns_the_entry_and_the_written_run(tmp_path):
    code, _ = assemble(
        """
  LDX #0
copy: TXA
  STA $2000,X
  INX
  BNE copy
  LDA #$5A
  STA $3000
  JMP $2000
""",
        0x0801,
    )
    entry, data = KernalHost(tmp_path).run_to_unpacked(build_prg(bytes(code), 0x0801))
    assert entry == 0x2000
    assert data[:256] == bytes(range(256))
    assert data[0x1000] == 0x5A and len(data) == 0x2000


def test_step_caps_raise_a_sid_error(tmp_path):
    host = KernalHost(tmp_path)
    spin = build_prg(b"\x4c\x01\x08", 0x0801)
    with pytest.raises(KernalHostError) as exc:
        host.run_to_unpacked(spin, max_steps=500)
    assert isinstance(exc.value, SidError)
    assert isinstance(exc.value, vm_kernal.KernalHostError)
    host.load_prg(spin)
    with pytest.raises(KernalHostError):
        host.run(until=lambda h: False, max_steps=100)


def _line(next_ptr: int, line_no: int, tokens: bytes) -> bytes:
    """A tokenised BASIC line: next-line pointer, line number, tokens, ``$00``."""
    return (
        next_ptr.to_bytes(2, "little") + line_no.to_bytes(2, "little") + tokens + b"\0"
    )


def _basic(*lines):
    mem = bytearray(0x10000)
    program = b"".join(lines)
    mem[0x0801 : 0x0801 + len(program)] = program
    return mem


@pytest.mark.parametrize("tokens", [b"\x9e2085", b"\x9e 2085"])
def test_sys_addr_reads_a_tokenised_sys_stub(tokens):
    assert sys_addr(_basic(_line(0x0900, 10, tokens))) == 2085


def test_sys_addr_scans_a_sid_image():
    stub = _line(0x0900, 10, b"\x9e2085")
    assert sys_addr(SidImage.from_bytes(build_prg(stub, 0x0801))) == 2085


def test_sys_addr_walks_the_next_line_link():
    first = _line(0x0809, 10, b"\x9510")  # 10 POKE10
    assert sys_addr(_basic(first, _line(0x0900, 20, b"\x9e2085"))) == 2085


@pytest.mark.parametrize("mem", [_basic(_line(0x0900, 0x6400, b"\x90")), _basic()])
def test_sys_addr_without_a_sys_raises(mem):
    with pytest.raises(KernalHostError):
        sys_addr(mem)


def test_editor_io_lists_the_transfer_call_sites():
    code, _ = assemble(
        "LDA #$01\nJSR $FFD5\nLDA #$02\nJSR $FFD8\nJSR $FFC0\nloop: JMP loop", 0x0801
    )
    image = SidImage.from_bytes(build_prg(bytes(code), 0x0801))
    sites = {name: (addr, rows) for name, addr, rows in editor_io(image)}
    assert set(sites) == {"LOAD", "SAVE", "OPEN"}
    assert all(rows for _, rows in sites.values())
