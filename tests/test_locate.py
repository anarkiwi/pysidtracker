"""Tests for ordered layout strategies (locate)."""

import pytest

from pysidtracker import (
    PlayroutineKind,
    SidImage,
    SidParseError,
    Strategy,
    locate,
    variant_facet,
)
from pysidtracker.locate import AFTER_INIT, STATIC

from .helpers import build_psid, store_sig

SIG = b"SIG"


def _find_sig(image, _candidate):
    pos = image.find(SIG)
    if pos < 0:
        raise SidParseError("no SIG")
    return pos


def test_static_wins_without_init():
    img = SidImage.from_sid(build_psid(b"..SIG..", load=0x1000))
    got = locate(img, [Strategy("direct", STATIC, _find_sig)])
    assert got.ok and got.layout == 0x1002 and got.strategy == "direct"
    assert got.detection.kind is PlayroutineKind.DIRECT
    assert got.variant.value == "direct"


def test_after_init_logs_static_rejection():
    img = SidImage.from_sid(build_psid(store_sig(SIG, 0x9000), load=0x1000))
    strategies = [
        Strategy("later", AFTER_INIT, _find_sig),
        Strategy("direct", STATIC, _find_sig),
    ]
    got = locate(img, strategies)
    assert got.strategy == "later" and got.layout == 0x9000
    assert [r.strategy for r in got.rejections] == ["direct"]
    assert got.detection.kind is PlayroutineKind.PACKED
    assert got.image is not img


def test_failure_keeps_full_log_and_candidates():
    img = SidImage.from_sid(build_psid(b"nothing", load=0x1000))

    def decode(_image, candidate):
        raise SidParseError(f"bad {candidate}")

    strategies = [
        Strategy("a", STATIC, decode, lambda _i: [1, 2]),
        Strategy("b", STATIC, lambda _i, _c: None),
    ]
    got = locate(img, strategies, init=False)
    assert not got.ok and got.variant is None
    assert [(r.strategy, r.candidate) for r in got.rejections] == [
        ("a", 1),
        ("a", 2),
        ("b", None),
    ]
    assert got.detection.kind is PlayroutineKind.UNKNOWN
    with pytest.raises(SidParseError, match="bad 2"):
        got.require()


def test_init_false_skips_emulated_stages():
    img = SidImage.from_sid(build_psid(store_sig(SIG, 0x9000), load=0x1000))
    got = locate(img, [Strategy("later", AFTER_INIT, _find_sig)], init=False)
    assert not got.ok
    assert got.rejections[0].reason == "init disabled"


def test_variant_facet_domain():
    facet = variant_facet([Strategy("a", STATIC, _find_sig)])
    assert facet.name == "variant" and facet.domain == ("a",)


def test_bad_stage():
    with pytest.raises(ValueError):
        Strategy("x", "never", _find_sig)
