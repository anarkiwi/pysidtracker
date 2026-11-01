"""Tests for the packed/relocating playroutine detector."""

import pytest

from pysidtracker import (
    Detection,
    PlayroutineKind,
    SidImage,
    detect_playroutine,
)

from .helpers import build_psid

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


@pytest.mark.strain_exempt(reason="exercises code outside the measured package")
def test_detection_dataclass_defaults():
    det = Detection(PlayroutineKind.DIRECT, ran_init=False)
    assert det.anchor is None
    assert det.changed_inside == 0
    assert det.written_outside == 0
