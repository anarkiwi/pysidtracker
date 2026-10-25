"""Tests for the masked 6502 code-fragment scanner (numpy + pure-python)."""

import pytest

from pysidtracker import (
    CodePattern,
    SidImage,
    SitePattern,
    find_code_all,
    find_code_first,
)


def _image(payload, load=0x1000):
    return SidImage.from_prg(bytes([load & 0xFF, load >> 8]) + bytes(payload))


def test_pattern_compile_fields():
    pat = CodePattern("A9 {imm} 8D 05 D4 {addr:w}")
    assert pat.length == 7
    assert pat.literals == ((0, 0xA9), (2, 0x8D), (3, 0x05), (4, 0xD4))
    assert pat.captures == (("imm", 1, 1), ("addr", 5, 2))


def test_pattern_bad_token():
    with pytest.raises(ValueError):
        CodePattern("A9 ZZ")
    with pytest.raises(ValueError):
        CodePattern("A9 {bad:q}")
    with pytest.raises(ValueError):
        CodePattern("   ")


def test_literal_wildcard_and_captures():
    img = _image([0xA9, 0x2A, 0x8D, 0x05, 0xD4, 0xBD, 0x34, 0x12, 0x85, 0xFA])
    imm = find_code_first(img, "A9 {imm} 8D 05 D4")
    assert imm.addr == 0x1000
    assert imm.captures == {"imm": 0x2A}
    word = find_code_first(img, "BD {base:w} 85 FA")
    assert word.addr == 0x1005
    assert word.captures == {"base": 0x1234}
    # ?? wildcard matches any operand byte.
    wild = find_code_first(img, "A9 ?? 8D 05 D4")
    assert wild.addr == 0x1000
    assert wild.captures == {}


def test_multiple_matches_and_none():
    payload = [0xA9, 0x01, 0x60, 0xA9, 0x02, 0x60, 0xA9, 0x03, 0x60]
    img = _image(payload)
    hits = find_code_all(img, "A9 {v} 60")
    assert [h.addr for h in hits] == [0x1000, 0x1003, 0x1006]
    assert [h.captures["v"] for h in hits] == [1, 2, 3]
    assert find_code_first(img, "A9 FF 60") is None
    assert find_code_all(img, "A9 FF 60") == []


def test_all_wildcard_no_literals():
    img = _image([0x11, 0x22, 0x33])
    hits = find_code_all(img, "{a} {b}")
    # every start position in the searched window yields a match.
    assert hits[0].addr == 0x1000
    assert hits[0].captures == {"a": 0x11, "b": 0x22}


def test_start_end_bounds():
    img = _image([0xA9, 0x01, 0x60, 0xA9, 0x02, 0x60])
    assert find_code_first(img, "A9 {v} 60", start=0x1003).captures["v"] == 2
    assert find_code_first(img, "A9 {v} 60", end=0x1003).captures["v"] == 1
    assert find_code_all(img, "A9 {v} 60", start=0x2000) == []
    # window smaller than the pattern length yields no matches.
    assert find_code_all(img, "A9 {v} 60", start=0x1004, end=0x1006) == []


def test_image_convenience_methods():
    img = _image([0xA9, 0x2A, 0x8D, 0x05, 0xD4])
    assert img.find_code("A9 {imm} 8D 05 D4").captures == {"imm": 0x2A}
    assert len(img.find_code_all("A9 {imm} 8D 05 D4")) == 1


# --- re-expressing real parser fingerprints as CodePatterns ----------------


def test_reexpress_futurecomposer_signature():
    # pyfuturecomposer _FC_SIGNATURES[1]: the $D417 filter-store fragment. The
    # spec uses the SAME ?? wildcard tokens the parser hand-rolled as None.
    spec = "8D 17 D4 A0 06 88 88 88 88 88 88 B1 ??"
    frag = bytes.fromhex("8D17D4A006888888888888B1") + bytes([0x7C])  # zp ptr wildcard
    img = _image([0x00] * 4 + list(frag))
    match = find_code_first(img, spec)
    assert match.addr == 0x1004


def test_reexpress_musicassembler_order_sig():
    # pymusicassembler _SIG_ORDER = rb"\xbd(..)\x85\xfa\xbd(..)\x85\xfb":
    # LDA order_lo,X ; STA $fa ; LDA order_hi,X ; STA $fb, bases captured.
    frag = [0xBD, 0x00, 0x20, 0x85, 0xFA, 0xBD, 0x80, 0x20, 0x85, 0xFB]
    img = _image(frag)
    match = find_code_first(img, "BD {order_lo:w} 85 FA BD {order_hi:w} 85 FB")
    assert match.captures == {"order_lo": 0x2000, "order_hi": 0x2080}


def test_reexpress_defmon_signature():
    # pydefmon SIGNATURE tokens are already a masked skeleton; feed them verbatim.
    spec = (
        "A2 ?? A9 ?? 8E 02 D4 8D 03 D4 A2 ?? A9 ?? 8E 00 D4 8D 01 D4 "
        "A2 ?? A0 ?? A9 ?? ?? ?? 8E 06 D4 8C 05 D4 8D 04 D4 4C"
    )
    tokens = spec.split()
    frag = bytes(0x00 if t == "??" else int(t, 16) for t in tokens)
    img = _image([0x11, 0x22] + list(frag))
    assert find_code_first(img, spec).addr == 0x1002


def test_reexpress_jch_immediate_idiom():
    # pyjch _one_operand(LDA #imm, STA $D405): A9 <imm> 8D 05 D4.
    img = _image([0xA9, 0x11, 0x8D, 0x05, 0xD4])
    assert find_code_first(img, "A9 {ad} 8D 05 D4").captures == {"ad": 0x11}


def test_site_alternation_reports_label_and_captures():
    img = _image([0xA9, 0x01, 0x8D, 0x00, 0x20, 0xA9, 0x00, 0x8D, 0x00, 0x30])
    pat = SitePattern("( on: A9 01 | off: A9 00 ) 8D {dst:w}")
    found = pat.find_all(img)
    assert len(found) == 2
    on, off = found[0], found[1]
    assert (on.addr, on.alt, on.alts, on.length) == (0x1000, "on", ("on",), 5)
    assert on.captures == {"dst": 0x2000}
    assert (off.addr, off.alt, off.captures) == (0x1005, "off", {"dst": 0x3000})
    assert pat.any(img)


def test_site_unlabelled_alternatives_use_index():
    pat = SitePattern("(A9 | A2) {v}")
    match = pat.find_first(bytes([0x00, 0xA2, 0x07]))
    assert (match.addr, match.alt, match.captures) == (1, "1", {"v": 7})


def test_site_optional_group_and_gap():
    pat = SitePattern("A9 {v} [ 18 ] 8D ??{1,3} EA")
    assert pat.find_first(bytes([0xA9, 5, 0x18, 0x8D, 1, 2, 0xEA])).captures == {"v": 5}
    assert pat.find_first(bytes([0xA9, 5, 0x8D, 1, 0xEA])) is not None
    assert pat.find_first(bytes([0xA9, 5, 0x8D, 0xEA])) is None
    exact = SitePattern("A9 ??{2} EA")
    assert exact.any(bytes([0xA9, 0, 0, 0xEA]))
    assert not exact.any(bytes([0xA9, 0, 0xEA]))


def test_site_capture_equality_and_word_backref():
    pat = SitePattern("A5 {z} 85 {z} AD {w:w} 8D {w:w}")
    assert pat.any(bytes([0xA5, 3, 0x85, 3, 0xAD, 1, 2, 0x8D, 1, 2]))
    assert not pat.any(bytes([0xA5, 3, 0x85, 4, 0xAD, 1, 2, 0x8D, 1, 2]))
    assert not pat.any(bytes([0xA5, 3, 0x85, 3, 0xAD, 1, 2, 0x8D, 1, 3]))


def test_site_policy_first_and_span():
    img = _image([0xEA, 0xEA, 0xEA])
    assert len(SitePattern("EA EA").find_all(img)) == 2
    assert len(SitePattern("EA EA", first=True).find_all(img)) == 1
    img.mem[0x2000:0x2002] = bytes([0xEA, 0xEA])
    assert len(SitePattern("EA EA").find_all(img)) == 2
    assert len(SitePattern("EA EA", span="image").find_all(img)) == 3
    assert len(SitePattern("EA EA").find_all(img, start=0x1001, end=0x1003)) == 1


def test_site_no_alternation_has_no_alt():
    match = SitePattern("EA").find_first(bytes([0xEA]))
    assert (match.alt, match.alts) == (None, ())


def test_site_optional_group_capture_scope():
    pat = SitePattern("[ A9 {v} ] 8D")
    assert pat.find_first(bytes([0xA9, 9, 0x8D])).captures == {"v": 9}
    assert pat.find_first(bytes([0x8D])).captures == {}


@pytest.mark.parametrize(
    "spec",
    [
        "",
        "ZZ",
        "( A9",
        "A9 )",
        "[ A9",
        "( A9 | )",
        "A9 ??{3,1}",
        "{v} {v:w}",
        "A9 ]",
    ],
)
def test_site_bad_spec(spec):
    with pytest.raises(ValueError):
        SitePattern(spec)


def test_site_bad_span():
    with pytest.raises(ValueError):
        SitePattern("A9", span="rom")
