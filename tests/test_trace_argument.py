"""Tests for subtune start tracing (pysidtracker.trace.trace_init_argument)."""

import pytest

from pysidtracker import SidImage, SidParseError
from pysidtracker.asm6502 import assemble
from pysidtracker.header import write_psid
from pysidtracker.trace import init_arguments, trace_init_argument

WRAPPER = """
init: tax
      lda table,x
      jmp base
base: sta $c000
      rts
loop: jmp loop
table: .byte $07, $2a, $13
"""


def _image(songs=3, org=0x1000):
    code, labels = assemble(WRAPPER, org)
    data = write_psid(load=org, init=org, play=labels["base"], image=code, songs=songs)
    return SidImage.from_bytes(data), labels


def test_argument_at_entry_is_the_table_entry():
    image, labels = _image()
    assert [trace_init_argument(image, None, labels["base"], k) for k in range(3)] == [
        0x07,
        0x2A,
        0x13,
    ]
    assert trace_init_argument(image, labels["init"], labels["base"], 1) == 0x2A


def test_entry_never_reached_is_none():
    image, labels = _image()
    assert trace_init_argument(image, None, 0x2000, 0) is None
    assert trace_init_argument(image, labels["loop"], 0x2000, 0, max_cycles=500) is None


def test_init_arguments_cover_every_song():
    image, labels = _image()
    assert init_arguments(image, labels["base"]) == (0x07, 0x2A, 0x13)
    single, labels = _image(songs=0)
    assert init_arguments(single, labels["base"]) == (0x07,)


def test_prg_needs_an_init_address():
    code, labels = assemble(WRAPPER, 0x1000)
    prg = SidImage.from_prg(b"\x00\x10" + code)
    with pytest.raises(SidParseError, match="no SID header"):
        trace_init_argument(prg, None, labels["base"])
    assert init_arguments(prg, labels["base"], init=0x1000) == (0x07,)
