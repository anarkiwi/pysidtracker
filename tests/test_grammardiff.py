"""Tests for the structural Lark grammar comparison."""

import pytest

from pysidtracker.grammardiff import grammar_diff
from pysidtracker.maincli import main

A = r"""
start: item* tail
item: "a" INT -> a_item
    | "b" /[xy]/
tail: "end" | "stop"
old: INT
INT: /[0-9]+/
%import text (WS)
%ignore WS
"""

# A respelt: a grouped ``item*``, alternatives reordered
SAME = r"""
start: (item)* tail
item: "b" /[xy]/
    | "a" INT -> a_item
tail: "stop" | "end"
old: INT
INT: /[0-9]+/
%import text (WS)
%ignore WS
"""

B = r"""
start: item+ tail
item: "a" INT -> a_item
    | "b" /[xz]/
tail: "end"
new: INT
INT: /[0-9a-f]+/
%import text (WS)
%ignore WS
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, "utf-8")
    return path


def test_identical_spellings_compare_equal(tmp_path):
    a, same = _write(tmp_path, "a.lark", A), _write(tmp_path, "same.lark", SAME)
    assert grammar_diff(a, same) == []
    assert grammar_diff(a, a, ["start"]) == []
    assert main(["grammardiff", str(a), str(same)]) == 0


def test_differences_are_reported(tmp_path, capsys):
    a, b = _write(tmp_path, "a.lark", A), _write(tmp_path, "b.lark", B)
    assert grammar_diff(a, b) == [
        f"rule old only in {a}",
        f"rule new only in {b}",
        "rule item: - B /[xy]/",
        "rule item: + B /[xz]/",
        "rule start: - tail",
        "rule tail: - STOP",
        "terminal INT: /[0-9]+/ != /[0-9a-f]+/",
        f"terminal STOP only in {a}",
    ]
    assert main(["grammardiff", str(a), str(b), "--start", "start"]) == 1
    out = capsys.readouterr()
    assert "rule old only in" not in out.out and "difference(s)" in out.err


def test_unloadable_grammar(tmp_path):
    bad = _write(tmp_path, "bad.lark", "start: missing\n")
    with pytest.raises(Exception, match="grammar does not load"):
        grammar_diff(bad, bad)
