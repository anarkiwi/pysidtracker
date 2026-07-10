"""Tests for resolve_entry_points and is_jmp_vector."""

from pysidtracker import SidImage, is_jmp_vector, resolve_entry_points

from .helpers import build_prg, build_psid


def test_resolve_entry_points_header_zero_fallback():
    sid = build_psid(b"\x00", load=0x1000, init=0, play=0x1006)
    header = SidImage.from_sid(sid).header
    init, play = resolve_entry_points(header, 0x1000)
    assert init == 0x1000  # init == 0 -> load
    assert play == 0x1006


def test_resolve_entry_points_bare_prg_defaults():
    init, play = resolve_entry_points(None, 0x1800, 0x1800, 0x1806)
    assert (init, play) == (0x1800, 0x1806)
    init, play = resolve_entry_points(None, 0x1800)
    assert (init, play) == (0x1800, 0x1800)


def test_is_jmp_vector():
    # JMP $1006 at $1000, target inside the image.
    body = bytes([0x4C, 0x06, 0x10]) + b"\x00" * 4
    image = SidImage.from_prg(build_prg(body, load=0x1000))
    assert is_jmp_vector(image, 0x1000)
    # Not a JMP opcode.
    assert not is_jmp_vector(image, 0x1003)
    # Address outside the image.
    assert not is_jmp_vector(image, 0x2000)


def test_is_jmp_vector_target_outside():
    body = bytes([0x4C, 0x00, 0x90])  # JMP $9000, outside the loaded image
    image = SidImage.from_prg(build_prg(body, load=0x1000))
    assert not is_jmp_vector(image, 0x1000)
