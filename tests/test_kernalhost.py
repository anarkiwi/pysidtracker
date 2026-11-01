"""The headless KERNAL host: real jump table and register ABI, files under a host dir.

Real 6502 programs (:func:`asm6502.assemble`) call the KERNAL entries as a C64
program does; the calls are answered in Python against ``tmp_path``.
"""

import pytest

from pysidtracker import KernalHost, KernalHostError, sys_addr
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


@pytest.mark.strain_exempt(reason="exercises code outside the measured package")
def test_given_roms_are_banked_in(tmp_path):
    host = KernalHost(
        tmp_path, roms={0xE000: b"\xc0" * 0x2000, 0xA000: b"\x11" * 0x2000}
    )
    assert (host.bus[0xE000], host.bus[0xA000]) == (0xC0, 0x11)


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


@pytest.mark.parametrize("mem", [_basic(_line(0x0900, 0x6400, b"\x90")), _basic()])
def test_sys_addr_without_a_sys_raises(mem):
    with pytest.raises(KernalHostError):
        sys_addr(mem)
