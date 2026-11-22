"""Tests for the helpers moved in from the RE tools: per-call write divergence, HVSC
capture lookup, relocation init stubs, header replace, register-change offsets and the
6502 lockstep."""

import hashlib

import numpy as np
import pytest

from pysidtracker.emu import live_ram, run_to_rts, wire_mpu
from pysidtracker.gridcmp import change_offsets
from pysidtracker.header import parse_sid_header, write_psid
from pysidtracker.oracle import SidtraceRow, register_changes
from pysidtracker.reglog import first_divergence
from pysidtracker.relocate import init_fixup, store_stub
from pysidtracker.survey import hvsc_capture
from pysidtracker.whydiff import LOCKSTEP_WATCH, lockstep


def test_first_divergence():
    assert first_divergence([[1, 2], [3]], [[1, 2], [3], [4]]) is None
    assert first_divergence([[1, 2], [3, 4]], [[1, 2], [3, 5]]) == (1, 1, 4, 5)
    assert first_divergence([[1], [3, 4]], [[1], [3]]) == (1, 1, 4, None)
    assert first_divergence([[9], [3]], [[9], [3, 4]]) == (1, 1, None, 4)


def _hvsc(tmp_path, data, eol="\n"):
    md5 = hashlib.md5(data).hexdigest()
    docs = tmp_path / "hvsc" / "DOCUMENTS"
    docs.mkdir(parents=True)
    text = f"[Database]{eol}; /MUSICIANS/A/Other.sid{eol}{'0' * 32}=0:10{eol}"
    text += f"; /MUSICIANS/A/Tune.sid{eol}{md5}=1:00 0:30{eol}"
    (docs / "Songlengths.md5").write_bytes(text.encode("latin-1"))
    return tmp_path / "hvsc"


def test_hvsc_capture(tmp_path):
    data = b"PSID-synthetic"
    hvsc = _hvsc(tmp_path, data, "\r\n")
    root = tmp_path / "caps"
    want = root / "MUSICIANS" / "A" / "Tune.2.csv.zst"
    want.parent.mkdir(parents=True)
    want.write_bytes(b"")
    assert hvsc_capture(data, 1, hvsc, root) == want
    assert hvsc_capture(data, 0, hvsc, root) is None
    assert hvsc_capture(b"unlisted", 1, hvsc, root) is None


def _run(code, at=0xC000, ram=None, flags=None):
    ram = bytearray(0x10000) if ram is None else ram
    ram[at : at + len(code)] = code
    mpu, mem = wire_mpu(ram)
    if flags is not None:
        mpu.p = flags
    run_to_rts(mpu, mem, at, 0, 100_000)
    return np.frombuffer(bytes(live_ram(mpu, ram)), np.uint8), mpu


def test_init_fixup_runs():
    ram = bytearray(0x10000)
    init = 0x1000
    ram[init : init + 6] = bytes(
        [0xF8, 0xA9, 0x09, 0x85, 0x20, 0x60]
    )  # SED; LDA; STA $20; RTS
    ram[0x0300] = 0x79
    ram[0x3FFF] = 0x5A
    src = np.arange(0x4000, 0x4100)
    ram[0x4000:0x4100] = (src * 7 & 0xFF).astype(np.uint8).tobytes()
    ram[0x5000:0x5003] = b"abc"
    adds = [(0x20, 0x01), (0x0300, 0x01)]
    copies = [(0x4000, 0x6000, 256), (0x5000, 0x5100, 3)]
    stores = [(0x30, 0x42), (0x1234, 0x42), (0x31, 7), (0x100, 7), (0x30, 9)]
    code = init_fixup(init, adds, copies, stores)
    assert code.endswith(store_stub(stores)) and store_stub(stores).count(0xA9) == 3
    assert code[:4] == bytes([0x20, 0x00, 0x10, 0xD8])
    mem, mpu = _run(code, 0xC000, ram)
    assert mem[0x20] == 0x0A and mem[0x300] == 0x7A
    assert bytes(mem[0x5FFF:0x60FF]) == bytes(ram[0x3FFF:0x40FF]) and mem[0x60FF] == 0
    assert bytes(mem[0x5100:0x5103]) == b"abc" and mem[0x5103] == 0
    assert [mem[0x30], mem[0x1234], mem[0x31], mem[0x100]] == [9, 0x42, 7, 7]
    assert not mpu.p & 0x08


def test_header_replace():
    data = write_psid(load=0x1000, init=0x1000, play=0x1003, image=b"\x60" * 4, songs=3,
                      name="n", speed=5)  # fmt: skip
    head = parse_sid_header(data)
    new = head.replace(init_address=0x2000, songs=1)
    assert (new.init_address, new.songs, head.init_address) == (0x2000, 1, 0x1000)
    assert new.replace(init_address=0x1000, songs=3) == head
    with pytest.raises(TypeError):
        head.replace(nope=1)


def test_register_changes_and_offsets():
    rows = [
        SidtraceRow(10, None, None, None, 0, 4, 0),
        SidtraceRow(12, None, 1, None, None, None, "vid"),
        SidtraceRow(20, None, None, None, 0, 4, 0x21),
        SidtraceRow(21, None, None, None, 1, 4, 0x41),
        SidtraceRow(30, None, None, None, 0, 4, 0x21),
        SidtraceRow(40, None, None, None, 0, 24, 0x0F),
    ]
    ref = register_changes(rows)
    assert ref == [(20, 4, 0x21), (40, 24, 0x0F)]
    assert register_changes(rows, chip=1) == [(21, 4, 0x41)]
    got = register_changes([(18, 4, 0x21), (18, 4, 0x21), (37, 24, 0x0F), (50, 1, 1)])
    assert got == [(18, 4, 0x21), (37, 24, 0x0F), (50, 1, 1)]
    assert change_offsets(got, ref) == (2, {-3: 1, -2: 1})
    assert change_offsets(got, [(20, 4, 0x21), (40, 24, 0x0E)]) == (1, {-2: 1})
    assert change_offsets([], ref) == (0, {})


PLAY = 0x1000


def _tune(table=0x2000, value=0x33, branch=0xD0):
    """Play: X = $10 counter; LDA table,X; STA $D400; STA $3000; INC $10; LDA $10;
    CMP #2; (branch) +1; NOP; RTS."""
    ram = bytearray(0x10000)
    code = [0xA6, 0x10, 0xBD, table & 0xFF, table >> 8, 0x8D, 0x00, 0xD4,
            0x8D, 0x00, 0x30, 0xE6, 0x10, 0xA5, 0x10, 0xC9, 0x02, branch, 0x01,
            0xEA, 0x60]  # fmt: skip
    ram[PLAY : PLAY + len(code)] = bytes(code)
    ram[table : table + 2] = bytes([0x11, 0x22])
    ram[table + 2] = value
    return ram


def test_lockstep_write_data_and_trace():
    assert lockstep(_tune(), _tune(), PLAY, PLAY, 4, progress=False) is None
    cells = (np.array([0x3000]), np.array([0x3000]))
    sid = np.zeros(0x10000, bool)
    sid[0xD400] = True
    why = lockstep(_tune(), _tune(value=0x44), PLAY, PLAY, 4, cells=cells,
                   writes=LOCKSTEP_WATCH | sid, progress=False)  # fmt: skip
    assert (why.frame, why.trace, why.write, why.read) == (2, None, 0, 0)
    assert why.data.tolist() == [0]
    assert why.a.writes(sid) == [(PLAY + 5, 0xD400, 0x33)]
    assert why.b.writes(sid) == [(PLAY + 5, 0xD400, 0x44)]
    assert why.a.reads(LOCKSTEP_WATCH)[0] == (PLAY + 2, 0x33)
    assert why.a.ram[0x3000] == 0x33 and why.b.ram[0x3000] == 0x44
    why = lockstep(_tune(), _tune(branch=0xF0), PLAY, PLAY, 4, progress=True)
    assert why.frame == 0 and why.trace == len(why.a.pcs) - 1
    assert why.a.pcs[why.trace] == PLAY + 20 and why.b.pcs[why.trace] == PLAY + 19


def test_lockstep_remap():
    a, b = _tune(value=0x99), _tune(table=0x2100, value=0x22)
    cells = (np.array([0x2000, 0x2001]), np.array([0x2100, 0x2101]))
    plain = lockstep(a, b, PLAY, PLAY, 4, cells=cells, progress=False)
    assert (plain.frame, plain.trace, plain.write, plain.read) == (2, None, 0, 0)
    assert plain.data.size == 0
    table = np.full(0x10000, -1, np.int64)
    table[0x2002] = 0x2001
    table[0x3000] = 0x3000
    remap = {PLAY + 2: table, PLAY + 8: table}
    assert lockstep(a, b, PLAY, PLAY, 4, remap, progress=False) is None
    kept = (np.array([0x3000]),) * 2
    dropped = lockstep(a, b, PLAY, PLAY, 4, remap, cells=kept, progress=False)
    assert (dropped.frame, dropped.write, dropped.data.tolist()) == (0, None, [0])
    assert dropped.a.ram[0x3000] == 0 and dropped.b.ram[0x3000] == 0x11


def test_lockstep_remap_keeps_pointer_bytes():
    def tune(value):
        ram = bytearray(0x10000)
        code = [0xA0, 0x00, 0xB1, 0x40, 0x8D, 0x00, 0x30, 0x60]
        ram[PLAY : PLAY + len(code)] = bytes(code)
        ram[0x40:0x42] = b"\x00\x20"
        ram[0x2000] = value
        return ram

    table = np.full(0x10000, -1, np.int64)
    table[[0x40, 0x41, 0x2000]] = -2
    assert (
        lockstep(tune(7), tune(0), PLAY, PLAY, 2, {PLAY + 2: table}, progress=False)
        is None
    )
    why = lockstep(tune(7), tune(0), PLAY, PLAY, 2, progress=False)
    assert why.a.writes(LOCKSTEP_WATCH) == [(PLAY + 4, 0x3000, 7)]
