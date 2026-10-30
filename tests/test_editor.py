"""Tests for running an editor's own code on its RAM (pysidtracker.editor)."""

import pytest

from pysidtracker import SidImage
from pysidtracker.errors import SidError
from pysidtracker.asm6502 import assemble
from pysidtracker.editor import RAM_SIZE, editor_replay, ram_image, run_editor
from pysidtracker.oracle import register_grid
from pysidtracker.testing import whole_song

AT, DECODED = 0x2000, 0x3000
DECODE, BUMP, PLAYER = 0xC000, 0xC100, 0x1000

DECODER = """
      ldx #0
copy: cpx $ae
      beq done
      lda $2000,x
      sta $3000,x
      inx
      bne copy
done: stx $3100
      lda $01
      sta $3101
      rts
"""
BUMPER = """
      ldx $3100
loop: dex
      bmi out
      inc $3000,x
      jmp loop
out:  rts
"""
LIVE = """
init: lda #0
      sta $1100
      rts
play: ldx $1100
      lda $3000,x
      sta $d400
      inc $1100
      rts
"""


def _ram(bank=0x34, volume=0x05):
    ram = bytearray(RAM_SIZE)
    for source, org in ((DECODER, DECODE), (BUMPER, BUMP), (LIVE, PLAYER)):
        code, _ = assemble(source, org)
        ram[org : org + len(code)] = code
    ram[0x01], ram[0xD418] = bank, volume
    return bytes(ram)


def test_ram_image_boots_at_init():
    image = ram_image(_ram(), DECODE)
    assert (image.load, image.end, len(image.image)) == (0, RAM_SIZE, RAM_SIZE)
    assert (image.header.init_address, image.header.play_address) == (DECODE, DECODE)
    assert ram_image(_ram(), DECODE, PLAYER + 3, flags=0x20).header.flags == 0x20


def test_run_editor_loads_decodes_and_runs_further_entries():
    payload = bytes((5, 9, 0x10))
    mem = run_editor(_ram(), payload, AT, (DECODE, BUMP))
    assert mem[0xAE] | mem[0xAF] << 8 == AT + 3
    assert bytes(mem[DECODED : DECODED + 4]) == bytes((6, 10, 0x11, 0))
    assert (mem[0x3100], mem[0x3101], mem[0xD418]) == (3, 0x34, 0x05)
    decoded = run_editor(_ram(), payload, AT, (DECODE,))
    assert bytes(decoded[DECODED : DECODED + 3]) == payload


def test_run_editor_options_keep_the_psid_driver_state():
    mem = run_editor(_ram(), b"\x01", AT, (DECODE,), keep_bank=False, keep_volume=False)
    assert mem[0x3101] != 0x34
    assert mem[0xD418] == 0x0F
    unset = run_editor(_ram(), b"\x01\x02", AT, (DECODE,), load_end=(0xFB,))
    assert unset[0xFB] | unset[0xFC] << 8 == AT + 2
    assert unset[0x3100] == 0


@pytest.mark.parametrize(
    "ram, payload, entries, match",
    [
        (bytes(10), b"\x01", (DECODE,), "65536 bytes"),
        (None, b"", (DECODE,), "payload is empty"),
        (None, b"\x01", (), "no entry points"),
    ],
)
def test_run_editor_rejects(ram, payload, entries, match):
    with pytest.raises(ValueError, match=match):
        run_editor(_ram() if ram is None else ram, payload, AT, entries)


def test_editor_replay_renders_the_live_player():
    payload = bytes((0x20, 0x30, 0x40))
    psid = editor_replay(
        _ram(),
        payload,
        AT,
        (DECODE, BUMP),
        (PLAYER, 0x3200),
        PLAYER,
        assemble(LIVE, PLAYER)[1]["play"],
    )
    image = SidImage.from_bytes(psid)
    assert (image.load, image.header.init_address) == (PLAYER, PLAYER)
    grid = register_grid(image, whole_song(3))
    assert [row[0] for row in grid] == [0x21, 0x31, 0x41]


STRAIGHT = """
      lda #1
      sta $2000
      lda #2
      sta $2001
spin: jmp spin
"""
RETURNS = """
      lda #1
      sta $2000
      rts
"""
RTS_ONLY, STRAIGHT_AT, RETURNS_AT = 0xC200, 0x1200, 0x1300


def _stop_ram():
    ram = bytearray(_ram())
    for source, org in (
        (STRAIGHT, STRAIGHT_AT),
        (RETURNS, RETURNS_AT),
        ("rts", RTS_ONLY),
    ):
        code, _ = assemble(source, org)
        ram[org : org + len(code)] = code
    return bytes(ram)


def test_run_editor_runs_an_entry_to_a_stop_pc():
    for stop in (STRAIGHT_AT + 5, {STRAIGHT_AT + 5, 0x1234}):
        mem = run_editor(
            _stop_ram(), b"\x00", AT, (RTS_ONLY, (STRAIGHT_AT, stop), RTS_ONLY)
        )
        assert (mem[0x2000], mem[0x2001]) == (1, 0)


def test_run_editor_stop_entry_restores_sp(monkeypatch):
    seen = []
    real = run_editor.__globals__["run_to_rts"]

    def spy(mpu, mem, entry, acc, cycles):
        seen.append(mpu.sp)
        return real(mpu, mem, entry, acc, cycles)

    monkeypatch.setitem(run_editor.__globals__, "run_to_rts", spy)
    run_editor(
        _stop_ram(),
        b"\x00",
        AT,
        (RTS_ONLY, RTS_ONLY, (STRAIGHT_AT, STRAIGHT_AT + 5), RTS_ONLY),
    )
    assert len(seen) == 2 and seen[0] == seen[1]


def test_run_editor_stop_entry_raises():
    with pytest.raises(SidError, match="never reached"):
        run_editor(
            _stop_ram(),
            b"\x00",
            AT,
            (RTS_ONLY, (STRAIGHT_AT, 0x1234)),
            max_cycles=10_000,
        )
    with pytest.raises(SidError, match="returned before"):
        run_editor(_stop_ram(), b"\x00", AT, (RTS_ONLY, (RETURNS_AT, 0x1234)))
