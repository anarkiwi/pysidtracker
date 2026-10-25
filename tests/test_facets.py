"""Tests for the facet registry and per-value provenance."""

import dataclasses

import pytest

from pysidtracker import SidImage
from pysidtracker.facets import (
    DEFAULT_SOURCE,
    Facet,
    FacetRegistry,
    Finding,
    Source,
    Variant,
    detected,
    inferred,
    on_off_detector,
    pattern_detector,
    probe_detector,
    reference_detector,
)


def _image(payload, load=0x1000):
    return SidImage.from_prg(bytes([load & 0xFF, load >> 8]) + bytes(payload))


def test_source_str():
    assert str(detected(0x1234)) == "detected@$1234"
    assert str(Source("detected")) == "detected"
    assert str(DEFAULT_SOURCE) == "default"
    assert str(inferred("sig")) == "inferred(sig)"


def test_facet_validation():
    with pytest.raises(ValueError):
        Facet("not-ident", "build")
    with pytest.raises(ValueError):
        Facet("x", "bogus")
    facet = Facet("x", "build", domain=[1, 2])
    assert facet.domain == (1, 2)
    assert facet.check(1) == 1
    with pytest.raises(ValueError):
        facet.check(3)
    assert Facet("y", "init").check("anything") == "anything"


def test_default_per_family():
    facet = Facet("drv", "build", default={"a": 1, "b": 2}, per_family=True)
    assert facet.default_for("a") == 1
    assert facet.default_for("zz") is None
    assert Facet("z", "init", default=7).default_for("a") == 7


def test_pattern_detector_value_and_capture():
    img = _image([0xA9, 0x2A, 0x8D, 0x05, 0xD4])
    det = pattern_detector("A9 {imm} 8D 05 D4", capture="imm")
    found = det(img)
    assert found == Finding(0x2A, detected(0x1000))
    assert pattern_detector("A9 ?? 8D", "yes")(img).value == "yes"
    assert pattern_detector("EA EA")(img) is None


def test_on_off_detector():
    on = _image([0x20, 0x00, 0x20])
    off = _image([0xEA, 0xEA, 0xEA])
    none = _image([0x00, 0x00, 0x00])
    det = on_off_detector("20 00 20", "EA EA EA")
    assert det(on).value is True
    assert det(off).value is False
    assert det(off).source == detected(0x1000)
    assert det(none) is None


def test_reference_and_probe_detectors():
    det = reference_detector(lambda img: img.load, {0x1000: "v1"}, label="load")
    found = det(_image([0]))
    assert found == Finding("v1", inferred("load 4096"))
    assert det(_image([0], load=0x2000)) is None
    assert reference_detector(lambda img: None, {})(_image([0])) is None
    probe = probe_detector(lambda img: (5, "ran"))
    assert probe(None) == Finding(5, inferred("ran"))
    assert probe_detector(lambda img: None)(None) is None


def _registry():
    return FacetRegistry(
        [
            Facet(
                "vib",
                "layout",
                domain=(True, False),
                default=False,
                detector=on_off_detector("20 00 20", "EA EA EA"),
            ),
            Facet(
                "drv",
                "build",
                default={"gt2": 0x0F},
                per_family=True,
                detector=pattern_detector("A9 {b}", capture="b"),
            ),
            Facet("rel", "release", default="r1"),
        ]
    )


def test_registry_detect_and_provenance():
    reg = _registry()
    assert len(reg) == 3
    assert "vib" in reg and reg["rel"].kind == "release"
    assert [f.name for f in reg.of_kind("build")] == ["drv"]
    var = reg.detect(_image([0x20, 0x00, 0x20, 0xA9, 0x07]), "gt2")
    assert var["vib"] is True
    assert var.source("vib") == detected(0x1000)
    assert var["drv"] == 0x07
    assert var.source("drv") == detected(0x1003)
    assert var["rel"] == "r1" and var.source("rel") == DEFAULT_SOURCE
    assert list(var) == ["vib", "drv", "rel"] and "rel" in var


def test_defaults_when_nothing_detected():
    var = _registry().detect(_image([0x00] * 4), "gt2")
    assert var["vib"] is False
    assert var["drv"] == 0x0F
    assert var.source("drv") == DEFAULT_SOURCE


def test_detected_value_outside_domain_rejected():
    facet = Facet("x", "init", domain=(1,), detector=lambda img: Finding(2, detected()))
    with pytest.raises(ValueError):
        facet.detect(None)


def test_resolve_frozen_dataclass():
    var = _registry().detect(_image([0x20, 0x00, 0x20, 0xA9, 0x07]), "gt2")
    plain = var.resolve("Build")
    assert type(plain).__name__ == "Build"
    assert (plain.vib, plain.drv, plain.rel) == (True, 7, "r1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        plain.vib = False


def test_as_info():
    var = _registry().detect(_image([0x20, 0x00, 0x20]), "gt2")
    info = var.as_info()
    assert info["vib"] is True
    assert info["vib.src"] == "detected@$1000"
    assert info["drv"] == 0x0F and info["drv.src"] == "default"
    assert info["rel.src"] == "default"


def test_replace_override():
    var = _registry().detect(_image([0x00] * 4), "gt2")
    new = var.replace("rel", "r2")
    assert new["rel"] == "r2"
    assert new.source("rel") == inferred("override")
    assert var["rel"] == "r1"
    assert new.replace("rel", "r3", detected(0x10)).source("rel") == detected(0x10)
    with pytest.raises(KeyError):
        var.replace("nope", 1)


def test_registry_duplicates_and_merge():
    reg = _registry()
    with pytest.raises(ValueError):
        reg.add(Facet("rel", "release"))
    merged = reg.merge(FacetRegistry([Facet("extra", "init", default=1)]))
    assert [f.name for f in merged] == ["vib", "drv", "rel", "extra"]
    assert len(reg) == 3
    with pytest.raises(ValueError):
        reg.merge(reg)


def test_variant_is_plain_value():
    assert isinstance(FacetRegistry().detect(None), Variant)
    assert FacetRegistry().detect(None).as_info() == {}
