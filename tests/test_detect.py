"""Tests for the packed/relocating playroutine detector."""

import pytest

from pysidtracker import (
    Detection,
    PlayroutineKind,
    SidImage,
    SidParseError,
    detect_playroutine,
    run_init,
)

from .helpers import RTS_ONLY, build_prg, build_psid, store_sig

SIG = b"SIG"


def _sig_recognizer(image):
    pos = image.find(SIG)
    return pos if pos >= 0 else None


def test_direct_recognition_no_init():
    img = SidImage.from_sid(build_psid(b"..SIG..", load=0x1000))
    det = detect_playroutine(img, _sig_recognizer)
    assert det.kind is PlayroutineKind.DIRECT
    assert not det.ran_init
    assert det.anchor == 0x1002
    assert det.recognised
    assert det.trustworthy_header


def test_unknown_without_init():
    img = SidImage.from_sid(build_psid(b"nothing", load=0x1000))
    det = detect_playroutine(img, _sig_recognizer, init=False)
    assert det.kind is PlayroutineKind.UNKNOWN
    assert not det.ran_init
    assert not det.recognised
    assert not det.trustworthy_header


def test_packed_via_emulated_init():
    # Init writes SIG to $9000, far outside the loaded region: decompression.
    code = store_sig(SIG, 0x9000)
    img = SidImage.from_sid(build_psid(code, load=0x1000))
    assert img.find(SIG) == -1  # not present statically
    det = detect_playroutine(img, _sig_recognizer)
    assert det.kind is PlayroutineKind.PACKED
    assert det.ran_init
    assert det.anchor == 0x9000
    assert det.written_outside >= 3
    assert det.changed_inside == 0


def test_relocated_via_emulated_init():
    # Init writes SIG within the loaded region (padded), no outside writes.
    code = bytearray(store_sig(SIG, 0x1040))
    image = bytearray(0x80)
    image[: len(code)] = code
    img = SidImage.from_sid(build_psid(bytes(image), load=0x1000))
    det = detect_playroutine(img, _sig_recognizer)
    assert det.kind is PlayroutineKind.RELOCATED
    assert det.ran_init
    assert det.written_outside == 0
    assert det.changed_inside >= 3


def test_unknown_after_init():
    img = SidImage.from_sid(build_psid(RTS_ONLY, load=0x1000))
    det = detect_playroutine(img, _sig_recognizer)
    assert det.kind is PlayroutineKind.UNKNOWN
    assert det.ran_init
    assert not det.recognised


def test_run_init_requires_header():
    img = SidImage.from_prg(build_prg(RTS_ONLY, load=0x1000))
    with pytest.raises(SidParseError, match="no SID header"):
        run_init(img)


def test_run_init_falls_back_to_load_when_init_zero():
    code = store_sig(SIG, 0x9000)
    img = SidImage.from_sid(build_psid(code, load=0x1000, init=0))
    run_init(img)
    assert bytes(img.mem[0x9000:0x9003]) == SIG


def test_detection_dataclass_defaults():
    det = Detection(PlayroutineKind.DIRECT, ran_init=False)
    assert det.anchor is None
    assert det.changed_inside == 0
    assert det.written_outside == 0
