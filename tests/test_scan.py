"""Tests for the low-level scan helpers, both numpy and pure-python paths."""

import pytest

from pysidtracker import _scan


@pytest.fixture(params=["numpy", "python"])
def scan_backend(request, monkeypatch):
    if request.param == "python":
        monkeypatch.setattr(_scan, "_np", None)
    elif _scan._np is None:  # pragma: no cover - numpy present in dev deps
        pytest.skip("numpy not installed")
    return request.param


def test_find_all_and_first():
    mem = b"xxSIGxxSIGx"
    assert _scan.find_all(mem, b"SIG") == [2, 7]
    assert _scan.find_first(mem, b"SIG") == 2
    assert _scan.find_first(mem, b"SIG", 3) == 7
    assert _scan.find_all(mem, b"") == []
    assert _scan.find_first(mem, b"zz") == -1


def test_find_split_table(scan_backend):  # pylint: disable=unused-argument
    lo = bytes(range(0x40, 0x50))
    hi = bytes(range(0x90, 0xA0))
    mem = bytearray(0x200)
    mem[0x100 : 0x100 + len(lo)] = lo
    mem[0x100 + len(lo) : 0x100 + 2 * len(lo)] = hi
    addr, first, length = _scan.find_split_table(mem, lo, hi, min_length=8, limit=0x200)
    assert addr == 0x100
    assert first == 0
    assert length == 16


def test_find_split_table_partial_slice(
    scan_backend,
):  # pylint: disable=unused-argument
    # Only lo[4:12] / hi[4:12] are present in memory (a contiguous sub-slice).
    lo = bytes(range(0x00, 0x10))
    hi = bytes(range(0x80, 0x90))
    seg_lo = lo[4:12]
    seg_hi = hi[4:12]
    mem = bytearray(0x100)
    mem[0x20 : 0x20 + 8] = seg_lo
    mem[0x28 : 0x28 + 8] = seg_hi
    addr, first, length = _scan.find_split_table(mem, lo, hi, min_length=8, limit=0x100)
    assert addr == 0x20
    assert first == 4
    assert length == 8


def test_find_split_table_none(scan_backend):  # pylint: disable=unused-argument
    assert _scan.find_split_table(bytearray(0x40), b"\x01" * 8, b"\x02" * 8) is None


def test_find_split_table_top_of_memory(
    scan_backend,
):  # pylint: disable=unused-argument
    # A table whose hi column ends exactly at the memory limit must still match
    # (regression: the pure-python scan used to stop one start address short).
    lo = bytes(range(0x40, 0x48))
    hi = bytes(range(0x90, 0x98))
    mem = bytearray(0x10000)
    mem[0xFFF0:0xFFF8] = lo
    mem[0xFFF8:0x10000] = hi
    addr, first, length = _scan.find_split_table(mem, lo, hi, min_length=8)
    assert (addr, first, length) == (0xFFF0, 0, 8)


def test_find_split_table_bad_args():
    assert _scan.find_split_table(bytearray(4), b"", b"") is None
    assert _scan.find_split_table(bytearray(4), b"\x01\x02", b"\x01") is None
