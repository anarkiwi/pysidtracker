"""State belongs to the tune: every entry of a player sees the one set of values."""

import pytest

from pysidtracker.asm6502 import assemble
from pysidtracker.playspec import SpecError, SpecPlayer, elaborate, native, parse
from pysidtracker.testing import whole_song

from .helpers import build_psid

TWO = """spec two;
layout { out: addr; }
cell o: u8 @ L.out;
state n: u8 = 0;
machine voice[x in 0..2] {
  state k: u8 = 10;
  control phase { lo = 0, hi = 1 }
  step { in phase { lo => { goto hi; } hi => { goto lo; k += 1; } } }
}
machine m { step { n += 1; o := n; } }
schedule { m; voice[0]; voice[1]; }
"""


def alternating(use_native):
    """A tune whose play routine calls entry ``pa`` then entry ``pb`` each frame."""
    code, labels = assemble(
        "init:\n  rts\npa:\n  rts\npb:\n  rts\nwrap:\n  jsr pa\n  jsr pb\n  rts\n",
        0x1000,
    )
    image = build_psid(code, 0x1000, init=labels["init"], play=labels["wrap"])[0x7C:]
    where = {labels["pa"]: 0x1300, labels["pb"]: 0x1301}
    return SpecPlayer(
        parse(TWO),
        image,
        0x1000,
        labels["init"],
        labels["wrap"],
        bind=lambda view, entry: {"out": where[entry]},
        entries=lambda view: set(where),
        native=use_native,
    )


@pytest.mark.strain_exempt(reason="asserts the two executors agree on shared state")
@pytest.mark.parametrize("fast", [False, True])
def test_entries_of_one_tune_share_their_state(fast):
    if fast and not native.available():
        pytest.skip("the native core is not built")
    play = alternating(fast)
    play.render_grid(whole_song(4))
    assert play.memory[0x1300:0x1302] == bytes([7, 8])
    assert play.state("n") == 8
    assert play.state("voice[1].k") == play.state("voice[0].k") == 14
    assert play.state("voice[0].phase") == 0
    play.set_state("n", 100)
    play.render_grid(whole_song(1))
    assert play.memory[0x1300:0x1302] == bytes([101, 102])
    with pytest.raises(KeyError, match="no state"):
        play.state("missing")


@pytest.mark.strain_exempt(reason="the same coverage under each backend fixture")
def test_output_is_a_name_wherever_it_is_not_a_declaration():
    text = """spec o;
cell c: u8 @ 0x20;
machine m {
  proc output() { c := 7; }
  step { output(); }
}
schedule { m; }
"""
    assert elaborate(parse(text), {}).name == "o"
    assert parse(text + 'output sid @ 0xD400 order "v0.freq";').outputs[0][2] == "sid"
    with pytest.raises(SpecError, match="unknown declaration"):
        parse(text + 'emit sid @ 0xD400 order "v0.freq";')
