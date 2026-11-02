"""Words that start a declaration are still names wherever a name is meant."""

import pytest

from pysidtracker.playspec import elaborate, parse
from pysidtracker.playspec.syntax import SOFT

HEAD = "spec t; cell c: u8 @ 0x20; "
PLACES = [
    HEAD + "cell W: u8 @ 0x21; machine m { step { if W == 1 { c := 2; } W -= 1; } }",
    HEAD + "machine m { proc W() { c := 1; } step { if c == 1 { c := 2; } W(); } }",
    HEAD + "machine m { step { let W = c; if true { c := W; } c := W + 1; } }",
    HEAD + "state W: u8 = 0; machine m { step { if c == 1 { } W += 1; c := W; } }",
    HEAD + "machine m { cell W: u8 @ 0x21; step { c := W; } }",
]


@pytest.mark.strain_exempt(reason="the same coverage under each backend fixture")
@pytest.mark.parametrize("word", [*SOFT, "match"])
def test_a_declaration_keyword_is_a_cell_proc_let_or_state_name(word):
    for place in PLACES:
        text = place.replace("W", word) + " schedule { m; }"
        assert elaborate(parse(text), {}).name == "t"
