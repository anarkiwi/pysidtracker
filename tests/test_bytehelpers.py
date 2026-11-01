"""Tests for the image, error and CLI helpers byte-stream readers share."""

import argparse

import pytest

from pysidtracker import (
    SidImage,
    SidParseError,
    address,
    make_package_errors,
    require_ram,
    reraise_as,
)


def test_require_ram_rejects_the_io_area():
    require_ram(0xC000, 0x1000, "song")
    require_ram(0xE000, 0x100, "song")
    with pytest.raises(SidParseError, match=r"song \[0xcfff,0xd001\) reaches"):
        require_ram(0xCFFF, 2, "song")

    class Invalid(Exception):
        pass

    with pytest.raises(Invalid):
        require_ram(0xDFFF, 1, "image", Invalid)


def test_put_ptr_splits_lo_and_hi():
    img = SidImage.from_prg(b"\x00\x10" + bytes(16))
    img.put_ptr(0x1000, 0x1008, 3, 0xABCD)
    assert img.mem[0x1003] == 0xCD and img.mem[0x100B] == 0xAB
    assert img.ptr(0x1000, 0x1008, 3) == 0xABCD
    with pytest.raises(SidParseError):
        img.put_ptr(0x0F00, 0x1008, 0, 0x1234)


def test_reraise_as_maps_parse_errors_only():
    root, parse_error, _ = make_package_errors("Wr")
    invalid = type("WrValidationError", (root,), {})
    with pytest.raises(invalid, match="bad") as info:
        with reraise_as(invalid, parse_error):
            raise parse_error("bad")
    assert isinstance(info.value.__cause__, parse_error)
    with pytest.raises(invalid):
        with reraise_as(invalid):
            raise SidParseError("base")
    with pytest.raises(KeyError):
        with reraise_as(invalid):
            raise KeyError("other")


@pytest.mark.parametrize(
    "text, value", [("$C000", 0xC000), ("0xc000", 0xC000), ("4096", 4096), ("0", 0)]
)
def test_address(text, value):
    assert address(text) == value


@pytest.mark.parametrize("text", ["C000", "$", "$10000", "-1", ""])
def test_address_rejects(text):
    with pytest.raises(argparse.ArgumentTypeError):
        address(text)
