"""Tests for native (pydexomizer) exomizer decrunch and its detect wiring."""

import os
import shutil
import subprocess

import pytest

from pysidtracker import (
    PlayroutineKind,
    SidImage,
    detect_playroutine,
    native_decrunch,
)
from pysidtracker import decrunch as decrunch_mod

from .helpers import build_psid, store_sig

SIG = b"SIG"


def _sig_recognizer(image):
    pos = image.find(SIG)
    return pos if pos >= 0 else None


class _FakeResult:
    def __init__(self, start, data, entry=0):
        self.start = start
        self.data = data
        self.entry = entry
        self.cycles = 0


class _FakePdx:
    """Stand-in for pydexomizer with a scripted sfx outcome."""

    def __init__(self, result=None):
        self._result = result
        self.sfx_calls = 0

    def decrunch_sfx(self, prg, entry=None, max_steps=None):
        # pylint: disable=unused-argument
        self.sfx_calls += 1
        if self._result is None:
            raise RuntimeError("not exomizer-packed")
        return self._result

    def decrunch_mem_auto(self, prg, proto=None):
        # pylint: disable=unused-argument
        raise RuntimeError("not a mem stream")


def test_native_decrunch_sfx_places_unpacked_image(monkeypatch):
    unpacked = b"..SIG.. payload"
    monkeypatch.setattr(decrunch_mod, "_pdx", _FakePdx(_FakeResult(0x2000, unpacked)))
    packed = build_psid(b"\x01\x02\x03packed", load=0x1000)
    img = native_decrunch(packed)
    assert isinstance(img, SidImage)
    assert img.load == 0x2000
    assert bytes(img.mem[0x2000 : 0x2000 + len(unpacked)]) == unpacked
    assert img.header is not None  # original header carried through
    assert img.find(SIG) == 0x2000 + 2


def test_native_decrunch_returns_none_when_not_packed(monkeypatch):
    monkeypatch.setattr(decrunch_mod, "_pdx", _FakePdx(None))
    assert native_decrunch(build_psid(b"whatever", load=0x1000)) is None


def test_native_decrunch_returns_none_on_empty_data(monkeypatch):
    monkeypatch.setattr(decrunch_mod, "_pdx", _FakePdx(_FakeResult(0x2000, b"")))
    assert native_decrunch(build_psid(b"x", load=0x1000)) is None


def test_native_decrunch_accepts_sidimage(monkeypatch):
    monkeypatch.setattr(decrunch_mod, "_pdx", _FakePdx(_FakeResult(0x0900, b"SIGdata")))
    img = SidImage.from_bytes(build_psid(b"packed", load=0x1000))
    out = native_decrunch(img)
    assert out is not None and out.load == 0x0900


# --- detect wiring (opt-in, non-regressing) --------------------------------


def test_detect_native_opt_in_reports_packed(monkeypatch):
    def fake_native(_image):
        return SidImage.from_bytes(build_psid(b"..SIG..", load=0x2000))

    monkeypatch.setattr(decrunch_mod, "native_decrunch", fake_native)
    img = SidImage.from_sid(build_psid(b"opaque", load=0x1000))
    det = detect_playroutine(img, _sig_recognizer, native=True)
    assert det.kind is PlayroutineKind.PACKED
    assert det.ran_init is False
    assert det.anchor is not None


def test_detect_native_falls_back_to_init(monkeypatch):
    # native decrunch yields nothing -> emulated-init path still runs.
    monkeypatch.setattr(decrunch_mod, "native_decrunch", lambda _img: None)
    code = store_sig(SIG, 0x9000)
    img = SidImage.from_sid(build_psid(code, load=0x1000))
    det = detect_playroutine(img, _sig_recognizer, native=True)
    assert det.kind is PlayroutineKind.PACKED
    assert det.ran_init is True


def test_detect_default_ignores_native(monkeypatch):
    # Without native=True, native_decrunch must never be consulted.
    def boom(_image):  # pragma: no cover - must not be called
        raise AssertionError("native_decrunch should not run by default")

    monkeypatch.setattr(decrunch_mod, "native_decrunch", boom)
    code = store_sig(SIG, 0x9000)
    img = SidImage.from_sid(build_psid(code, load=0x1000))
    det = detect_playroutine(img, _sig_recognizer)
    assert det.kind is PlayroutineKind.PACKED
    assert det.ran_init is True


# --- real exomizer round-trip (needs the exomizer cruncher) ----------------
# Marked ``exomizer``: excluded from the default run (like ``oracle``) and run in
# a dedicated CI job that builds the reference exomizer, so it is executed for
# real -- never skipped. ``$EXOMIZER`` overrides the binary location.

_EXOMIZER = os.environ.get("EXOMIZER") or shutil.which("exomizer") or "exomizer"


@pytest.mark.exomizer
def test_native_decrunch_real_exomizer_sfx(tmp_path):
    """End-to-end: crunch a payload with the real exomizer, then decrunch it.

    Wraps a genuine ``exomizer sfx`` self-extracting PRG in a SID container and
    asserts :func:`native_decrunch` recovers the original payload byte-for-byte
    and that the signature -- absent at the load address while crunched -- is
    present after unpacking. Exercises the real pydexomizer path, not a mock.
    """
    payload = (bytes(range(256)) * 3 + b"..SIG.." + bytes(range(256)) * 3)[:1500]
    sig_off = payload.index(SIG)
    src = tmp_path / "p.bin"
    src.write_bytes(payload)
    sfx = tmp_path / "p.sfx"
    subprocess.run(
        [_EXOMIZER, "sfx", "0x2000", "-q", "-o", str(sfx), f"{src}@0x2000"],
        check=True,
        capture_output=True,
    )
    sfx_prg = sfx.read_bytes()
    load = sfx_prg[0] | (sfx_prg[1] << 8)
    sid = build_psid(sfx_prg[2:], load=load, init=load, play=0)
    packed = SidImage.from_bytes(sid)
    # Packing genuinely transforms the payload: the plaintext is not already
    # sitting at the target load address in the crunched image.
    assert bytes(packed.mem[0x2000 : 0x2000 + len(payload)]) != payload
    unpacked = native_decrunch(sid)
    assert unpacked is not None
    assert unpacked.load == 0x2000
    assert bytes(unpacked.mem[0x2000 : 0x2000 + len(payload)]) == payload
    assert _sig_recognizer(unpacked) == 0x2000 + sig_off  # SIG recoverable
