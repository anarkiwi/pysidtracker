"""Byte-exact jennings-vs-sidtrace oracle renders (the ``oracle`` marker).

Runs need Docker and HVSC (a local ``$HVSC`` tree or cache), so the default
suite excludes them (a CI job runs ``pytest -m oracle``) and an unavailable
tune or render fails rather than skips. ``make_oracle_fixtures`` reuses the harness.
"""

import pytest

from pysidtracker import make_oracle_fixtures, register_grid

# HVSC tunes verified to render byte-exactly against the deterministic oracle
# with the base jennings renderer. NMOS illegal opcodes are enabled (harmless for
# tunes that do not use them).
TUNES = {
    "monty": "MUSICIANS/H/Hubbard_Rob/Monty_on_the_Run.sid",
    "trap": "MUSICIANS/D/Daglish_Ben/Trap.sid",
    "commando": "MUSICIANS/H/Hubbard_Rob/Commando.sid",
    "sanxion": "MUSICIANS/H/Hubbard_Rob/Sanxion.sid",
    "cybernoid": "MUSICIANS/T/Tel_Jeroen/Cybernoid.sid",
    "thing_on_a_spring": "MUSICIANS/H/Hubbard_Rob/Thing_on_a_Spring.sid",
    "into_orbit": "MUSICIANS/D/Danko_Tomas/Into_Orbit.sid",
}


def _render(data, nframes):
    return register_grid(data, nframes, illegal_opcodes=True)


tune_id, oracle_match = make_oracle_fixtures(
    TUNES,
    render=_render,
    frames=250,
)


@pytest.mark.oracle
def test_render_matches_oracle(oracle_match):  # noqa: F811
    oracle_match()
