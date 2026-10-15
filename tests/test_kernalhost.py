"""Offline tests for the headless KERNAL host (file and keyboard traps to a host dir).

These run real 6502 programs (:func:`asm6502.assemble`) on :class:`KernalHost` with
the KERNAL I/O answered in Python against ``tmp_path`` -- no VICE, no network.
"""

import pytest

from jennings.devices.mpu6502 import MPU

from pysidtracker import KernalHost, KernalHostError, SidImage, editor_io, sys_addr
from pysidtracker import kernalhost
from pysidtracker.asm6502 import assemble

from .helpers import build_prg


def _prg(text, org=0x0801):
    """``text`` assembled at ``org`` as a bare PRG (2-byte load + code)."""
    code, _ = assemble(text, org)
    return build_prg(bytes(code), org)


def test_save_then_getin_round_trips_through_the_host_dir(tmp_path):
    host = KernalHost(tmp_path)
    ram = host.ram()
    ram[0x1000:0x1100] = bytes(range(256))  # the payload to be saved
    ram[0x1300] = 4
    ram[0x1301:0x1305] = b"WORK"  # SETNAM target: name "WORK", SEQ
    ram[0x02], ram[0x03] = 0x00, 0x10  # SAVE start word -> $1000
    host.tap_key(ord("A"))
    host.load_prg(
        _prg(
            "SEI\n"
            "LDA #$13\nLDX #$00\nLDY #$01\nJSR $FFD2\n"  # SETNAM $1300 (SEQ)
            "LDA #$01\nLDX #$08\nLDY #$08\nJSR $FFCF\n"  # SETLFS 1,8,8
            "LDA #$02\nLDX #$00\nLDY #$11\nJSR $FFD8\n"  # SAVE $1000:$1100
            "JSR $FFCC\nSTA $1400\nloop: JMP loop\n"  # GETIN -> $1400
        )
    )
    host.run(until=lambda h: h.ram()[0x1400] != 0, max_steps=4096)
    assert (tmp_path / "8" / "WORK.seq").read_bytes() == bytes(range(256))
    assert ram[0x1400] == ord("A")


def test_tap_key_chains_and_screen_is_one_kilobyte(tmp_path):
    host = KernalHost(tmp_path)
    assert host.tap_key(ord("A")) is host
    assert host.tap_key(ord("B")) is host
    assert len(host.screen()) == 1024


def test_run_to_unpacked_lands_on_the_freshly_written_entry(tmp_path):
    payload = bytes(range(256))  # -> $2000..$20FF (page 2)
    marker = 0x5A  # -> $3000 (page 3, contiguous, extends the written run)
    code = bytearray()
    for i, b in enumerate(payload):
        code += (
            b"\xa9"
            + bytes([b])
            + b"\x8d"
            + bytes([(0x2000 + i) & 0xFF, (0x2000 + i) >> 8])
        )
    code += b"\xa9" + bytes([marker]) + b"\x8d\x00\x30"  # LDA #marker / STA $3000
    code += b"\x4c\x00\x20"  # JMP $2000
    host = KernalHost(tmp_path)
    image = host.run_to_unpacked(build_prg(bytes(code), 0x0801))
    assert isinstance(image, SidImage)
    assert image.load == 0x2000
    assert bytes(image.mem[0x2000 : 0x2000 + 256]) == payload
    assert image.mem[0x3000] == marker
    assert image.end >= 0x4000  # the written run extended over the contiguous page 3


def test_run_to_unpacked_raises_when_never_expanding(tmp_path):
    host = KernalHost(tmp_path)
    with pytest.raises(KernalHostError):
        host.run_to_unpacked(build_prg(b"\x4c\x01\x08", 0x0801), max_steps=500)


def test_run_raises_on_the_step_cap(tmp_path):
    host = KernalHost(tmp_path)
    host.load_prg(build_prg(b"\x4c\x01\x08", 0x0801))
    with pytest.raises(KernalHostError):
        host.run(max_steps=100)


def test_sys_addr_reads_the_sys_operand_from_ram(tmp_path):
    host = KernalHost(tmp_path)
    host.ram()[0x0801:0x0809] = bytes([0x00, 0x0A, 0xFF, 0xFF, 0x64, 0x00, 0x20, 0x0D])
    assert sys_addr(host.ram(), base=0x0801) == 0x2000


def test_sys_addr_scans_a_sid_image():
    stub = bytes([0x00, 0x0A, 0xFF, 0xFF, 0x64, 0x00, 0x20, 0x0D])
    assert sys_addr(SidImage.from_bytes(build_prg(stub, 0x0801)), base=0x0801) == 0x2000


def test_sys_addr_raises_when_no_token(tmp_path):
    with pytest.raises(KernalHostError):
        sys_addr(KernalHost(tmp_path).ram(), base=0x0801)


def test_editor_io_lists_the_transfer_call_sites():
    prg = _prg(
        "LDA #$01\nLDX #$00\nJSR $FFD5\n"  # LOAD
        "LDA #$02\nJSR $FFD8\n"  # SAVE
        "LDA #$03\nJSR $FFC0\n"  # OPEN
        "loop: JMP loop\n"
    )
    image = SidImage.from_bytes(prg)
    sites = {name: (addr, rows) for name, addr, rows in editor_io(image)}
    assert set(sites) == {"LOAD", "SAVE", "OPEN"}
    byte = {"LOAD": b"\x20\xd5\xff", "SAVE": b"\x20\xd8\xff", "OPEN": b"\x20\xc0\xff"}
    for name, (addr, rows) in sites.items():
        assert bytes(image.mem[addr : addr + 3]) == byte[name]
        assert rows


def test_open_chout_close_writes_a_work_file(tmp_path):
    host = KernalHost(tmp_path)
    ram = host.ram()
    ram[0x1480] = 2
    ram[0x1481:0x1483] = b"CH"  # SETNAM target: name "CH", SEQ @ $1480
    host.load_prg(
        _prg(
            "SEI\n"
            "LDA #$14\nLDX #$80\nLDY #$01\nJSR $FFD2\n"  # SETNAM "CH" SEQ
            "LDA #$01\nLDX #$08\nLDY #$08\nJSR $FFCF\n"  # SETLFS
            "JSR $FFC0\n"  # OPEN (write channel)
            "LDA #$41\nJSR $FFC6\nLDA #$42\nJSR $FFC6\n"  # CHROUT 'A','B'
            "LDA #$01\nJSR $FFC3\n"  # CLOSE 1
            "STA $1700\nLDA #$01\nSTA $1701\nloop: JMP loop\n"
        )
    )
    host.run(until=lambda h: h.ram()[0x1701] != 0, max_steps=4096)
    assert (tmp_path / "8" / "CH.seq").read_bytes() == b"AB"


def test_chrin_reads_back_from_an_opened_work_file(tmp_path):
    drive = tmp_path / "8"
    drive.mkdir(parents=True, exist_ok=True)
    (drive / "RD.seq").write_bytes(b"READ")
    host = KernalHost(tmp_path)
    ram = host.ram()
    ram[0x14C0] = 2
    ram[0x14C1:0x14C3] = b"RD"  # SETNAM target: name "RD", SEQ @ $14C0
    host.load_prg(
        _prg(
            "SEI\n"
            "LDA #$14\nLDX #$C0\nLDY #$01\nJSR $FFD2\n"  # SETNAM "RD" SEQ
            "LDA #$01\nLDX #$08\nLDY #$08\nJSR $FFCF\n"  # SETLFS
            "JSR $FFC0\n"  # OPEN (existing -> read channel)
            "JSR $FFC9\nSTA $1702\n"  # CHRIN -> 'R'
            "JSR $FFC9\nSTA $1703\n"  # CHRIN -> 'E'
            "LDA #$01\nSTA $1701\nloop: JMP loop\n"
        )
    )
    host.run(until=lambda h: h.ram()[0x1701] != 0, max_steps=4096)
    assert ram[0x1702] == 0x52 and ram[0x1703] == 0x45  # 'R', 'E'


def test_save_load_round_trip_and_verify(tmp_path):
    host = KernalHost(tmp_path)
    ram = host.ram()
    payload = bytes(range(256))
    ram[0x1000:0x1100] = payload
    ram[0x1500] = 1
    ram[0x1501] = 0x56  # name "V", SEQ @ $1500
    ram[0x02], ram[0x03] = 0x00, 0x10  # SAVE start word -> $1000
    host.load_prg(
        _prg(
            "SEI\n"
            "LDA #$15\nLDX #$00\nLDY #$01\nJSR $FFD2\n"  # SETNAM "V" SEQ
            "LDA #$01\nLDX #$08\nLDY #$08\nJSR $FFCF\n"  # SETLFS
            "LDA #$02\nLDX #$00\nLDY #$11\nJSR $FFD8\n"  # SAVE $1000:$1100
            "LDA #$00\nLDX #$00\nLDY #$30\nJSR $FFD5\n"  # LOAD -> $3000
            "LDA #$01\nLDX #$00\nLDY #$30\nJSR $FFD5\n"  # VERIFY $3000
            "STA $1704\nLDA #$01\nSTA $1701\nloop: JMP loop\n"
        )
    )
    host.run(until=lambda h: h.ram()[0x1701] != 0, max_steps=4096)
    assert (tmp_path / "8" / "V.seq").read_bytes() == payload
    assert ram[0x3000:0x3100] == payload
    assert ram[0x1704] == 0  # verify: equal -> A=0


def test_explicit_roms_and_ntsc_bypass_resolution(tmp_path):
    host = KernalHost(
        tmp_path, ntsc=True, kernal_rom=b"\xc0" * 0x2000, basic_rom=b"\x11" * 0x2000
    )
    assert host.ntsc is True
    assert host.mem[0xE000] == 0xC0 and host.mem[0xA000] == 0x11


def test_resolved_roms_are_banked_in_when_available(tmp_path, monkeypatch):
    kernal = tmp_path / "k.bin"
    kernal.write_bytes(b"\xe4" * 0x2000)
    basic = tmp_path / "b.bin"
    basic.write_bytes(b"\x5a" * 0x2000)
    monkeypatch.setattr(
        kernalhost.roms,
        "resolve_rom",
        lambda name, **_kw: {"kernal": kernal, "basic": basic}[name],
    )
    host = KernalHost(tmp_path / "h")
    assert host.ram()[1] == 0x37  # power-on bank selects the ROMs
    assert host.mem[0xE000] == 0xE4  # real KERNAL, not the stub
    assert host.mem[0xA000] == 0x5A  # real BASIC, not zero


def test_verb_error_paths_set_carry(tmp_path):
    host = KernalHost(tmp_path)
    mpu = host.mpu
    mpu.a = mpu.x = mpu.y = 0
    host._getin()
    assert mpu.a == 0 and mpu.p & MPU.CARRY  # empty keyboard queue
    mpu.a = 0x41
    host._chout()
    assert mpu.a != 0 and mpu.p & MPU.CARRY  # no write channel open
    host._chrin()
    assert mpu.a != 0 and mpu.p & MPU.CARRY  # no input channel open
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    host._lfn = 1
    host._channels[1] = {"fh": open(empty, "rb"), "mode": "r", "path": empty}
    host._chrin()
    assert mpu.a != 0 and mpu.p & MPU.CARRY  # input channel at EOF
    host._channels[1]["fh"].close()
    host._name, host._filetype = "MISSING", 1
    host._load()
    assert mpu.a != 0 and mpu.p & MPU.CARRY  # file not found
    host._host_dir = tmp_path / "a-file" / "8"
    (tmp_path / "a-file").write_bytes(b"x")  # the drive dir is a file
    host._name, host._filetype = "X", 1
    host._open()
    assert mpu.p & MPU.CARRY  # OSError -> carry set
    mpu.a = 1
    host._close()
    assert mpu.a == 0 and not (mpu.p & MPU.CARRY)  # closing is not an error


def test_load_prg_accepts_a_sidimage(tmp_path):
    host = KernalHost(tmp_path)
    image = SidImage.from_bytes(build_prg(b"\xa9\x01\x8d\x00\x14\x60", 0x1000))
    assert host.load_prg(image) == 0x1000
    assert host.ram()[0x1000] == 0xA9


def test_untrapped_jsr_runs_on_the_cpu(tmp_path):
    host = KernalHost(tmp_path)
    host.ram()[0x1000] = 0x60  # an RTS living in the (untrapped) RAM behind the call
    host.load_prg(_prg("JSR $1000\nLDA #$01\nSTA $1701\nloop: JMP loop\n"))
    host.run(until=lambda h: h.ram()[0x1701] != 0, max_steps=4096)
    assert host.ram()[0x1701] == 1
