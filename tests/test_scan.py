"""Tests for the low-level scan helpers, both numpy and pure-python paths."""

from pysidtracker import _scan


def test_find_all_and_first():
    mem = b"xxSIGxxSIGx"
    assert _scan.find_all(mem, b"SIG") == [2, 7]
    assert _scan.find_first(mem, b"SIG") == 2
    assert _scan.find_first(mem, b"SIG", 3) == 7
    assert _scan.find_all(mem, b"") == []
    assert _scan.find_first(mem, b"zz") == -1


def test_find_split_table_bad_args():
    assert _scan.find_split_table(bytearray(4), b"", b"") is None
    assert _scan.find_split_table(bytearray(4), b"\x01\x02", b"\x01") is None
