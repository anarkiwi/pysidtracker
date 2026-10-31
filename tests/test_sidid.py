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


def test_signatures_keep_cfg_order(cfg):
    sigs = sidid.sidid_signatures(cfg)
    assert [(n, len(r)) for n, r in sigs] == [
        ("Other_Player", 1),
        ("GoatTracker_V2.x", 2),
        ("(GT_Editor)", 1),
    ]
    gt2 = sigs[1][1][0]
    assert gt2.search(b"\x00\xbd\x12\x0a\xd0\x01\x02\xc9\xff")
    assert not gt2.search(b"\xbd\x12\x34\xd1\xc9\xff")
    assert not sigs[2][1][0].search(b"\x4c\x00\x20")


def test_identify_first_match_wins(cfg):
    sigs = sidid.sidid_signatures(cfg)
    assert sidid.identify(GT_CODE, sigs) == "GoatTracker_V2.x"
    assert sidid.identify(b"\xee\x20\xd0", sigs) == "GoatTracker_V2.x"
    assert sidid.identify(GT_CODE + b"\xa9\x00\x8d", sigs) == "Other_Player"
    assert sidid.identify(b"\x4c\x00\x10" + GT_CODE, sigs) == "GoatTracker_V2.x"
    assert sidid.identify(b"\x4c\x00\x10", sigs) == "(GT_Editor)"
    assert sidid.identify(b"\x00" * 16, sigs) is None
    assert sidid.identify_all(b"\x4c\x00\x10" + GT_CODE + b"\xa9\x00\x8d", sigs) == [
        "Other_Player",
        "GoatTracker_V2.x",
        "(GT_Editor)",
    ]


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


def test_cli(cfg, tmp_path, capsys):
    root = tmp_path / "hvsc"
    (root / "A").mkdir(parents=True)
    (root / "A" / "gt.sid").write_bytes(build_psid(GT_CODE, 0x1000))
    (root / "A" / "both.sid").write_bytes(build_psid(GT_CODE + b"\xa9\x00\x8d", 0x1000))
    (root / "none.sid").write_bytes(build_psid(b"\x00", 0x1000))
    (root / "A" / "skip.txt").write_bytes(GT_CODE)
    base = ["--cfg", str(cfg), "--jobs", "1"]
    assert sidid.main([str(root)] + base) == 0
    out = capsys.readouterr()
    assert out.out.splitlines() == [
        f"{root}/A/both.sid\tOther_Player",
        f"{root}/A/gt.sid\tGoatTracker_V2.x",
        f"{root}/none.sid\t{sidid.UNIDENTIFIED}",
    ]
    assert "1 GoatTracker_V2.x" in out.err
    sidid.main([str(root / "A" / "both.sid"), "-m", "--player", "goat"] + base)
    assert capsys.readouterr().out == f"{root}/A/both.sid\tGoatTracker_V2.x\n"


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
