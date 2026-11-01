"""Pure-Python SIDId: cfg parsing, first-match semantics, bundled cfg and CLI."""

import re

import pytest

from pysidtracker import sidid

from .helpers import build_psid

CFG = """Other_Player\r
A9 00 8D END\r
GoatTracker_V2.x\r
BD ?? ?? D0 AND C9 FF END\r
EE 20 D0 end\r
(GT_Editor)\r
4C ?? 10 END\r
Empty_Player\r
"""
GT_CODE = b"\xbd\x12\x34\xd0\x00\x00\xc9\xff"


@pytest.fixture(name="cfg")
def _cfg(tmp_path):
    cfg = tmp_path / "sidid.cfg"
    cfg.write_bytes(CFG.encode())
    return cfg


def _matching(sig):
    """Bytes matching a compiled signature: wildcards as zero, skips as nothing."""
    pat = sig.pattern.replace(b".*?", b"").replace(b".", b"\x00")
    return re.sub(rb"\\(.)", rb"\1", pat, flags=re.S)


def test_bundled_cfg():
    sigs = sidid.default_signatures()
    names = [n for n, _ in sigs]
    assert len(sigs) > 500 and "GoatTracker_V2.x" in names and "DMC" in " ".join(names)
    gt2 = dict(sigs)["GoatTracker_V2.x"][0]
    data = b"\x00" * 8 + _matching(gt2)
    assert gt2.search(data)
    assert "GoatTracker_V2.x" in sidid.identify_all(data)
    assert sidid.identify(data) == sidid.identify_all(data)[0]


def test_threads_scan_matches_processes(cfg, tmp_path, capsys):
    root = tmp_path / "hvsc"
    root.mkdir()
    (root / "gt.sid").write_bytes(build_psid(GT_CODE, 0x1000))
    (root / "none.sid").write_bytes(build_psid(b"\x00", 0x1000))
    out = {}
    for mode in ([], ["--threads"]):
        sidid.main([str(root), "--cfg", str(cfg), "--jobs", "2"] + mode)
        out[bool(mode)] = capsys.readouterr().out
    assert out[True] == out[False] and "GoatTracker_V2.x" in out[True]
