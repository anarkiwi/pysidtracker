"""Tests for BaseSidParser and the source dispatch."""

import io

import pytest

from pysidtracker import BaseSidParser, PlayroutineKind, SidParseError, read_bytes

from .helpers import build_psid, store_sig


class _Parser(BaseSidParser):
    error_class = SidParseError

    def parse(self, data, **kwargs):
        image = self.load_image(data)
        pos = image.find(b"SIG")
        if pos < 0:
            raise self.error_class("no SIG")
        return {"anchor": pos, **kwargs}

    def recognize(self, image):
        pos = image.find(b"SIG")
        return pos if pos >= 0 else None


class _NoRecognize(BaseSidParser):
    def parse(self, data, **kwargs):
        return len(data)


def test_read_from_bytes_path_and_file(tmp_path):
    sid = build_psid(b"..SIG..", load=0x1000)
    p = tmp_path / "t.sid"
    p.write_bytes(sid)
    parser = _Parser()
    assert parser.read(sid)["anchor"] == 0x1002
    assert parser.read(str(p))["anchor"] == 0x1002
    assert parser.read(p)["anchor"] == 0x1002
    assert parser.read(io.BytesIO(sid))["anchor"] == 0x1002


def test_parse_kwargs_pass_through():
    sid = build_psid(b"..SIG..", load=0x1000)
    assert _Parser().read(sid, subtune=2)["subtune"] == 2


def test_parse_failure():
    with pytest.raises(SidParseError, match="no SIG"):
        _Parser().read(build_psid(b"nope", load=0x1000))


def test_detect_direct():
    sid = build_psid(b"..SIG..", load=0x1000)
    det = _Parser().detect(sid)
    assert det.kind is PlayroutineKind.DIRECT


def test_detect_packed_through_base():
    sid = build_psid(store_sig(b"SIG", 0x9000), load=0x1000)
    det = _Parser().detect(sid)
    assert det.kind is PlayroutineKind.PACKED


def test_default_recognize_is_unknown():
    sid = build_psid(b"whatever", load=0x1000)
    det = _NoRecognize().detect(sid, init=False)
    assert det.kind is PlayroutineKind.UNKNOWN


def test_read_bytes_type_error():
    with pytest.raises(TypeError, match="cannot read a tune"):
        read_bytes(1234)


def test_read_bytes_accepts_bytearray():
    assert read_bytes(bytearray(b"abc")) == b"abc"
