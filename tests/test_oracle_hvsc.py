"""Byte-exact comparison of the jennings render against the sidtrace oracle.

Marked ``oracle``: these tests need Docker (the ``anarkiwi/sidtrace`` image) and
network access to HVSC, so the default suite excludes them (see ``pyproject``);
a dedicated CI job runs ``pytest -m oracle``. They are never skipped -- an
unavailable tune or a failed oracle render fails the test rather than hiding a
regression. HVSC ``.sid`` files are copyright works: they are downloaded to a
cache (or a local ``$HVSC`` tree), never committed.

A format package reuses this verbatim by calling ``make_oracle_fixtures`` with
its own tune list and its player's renderer.
"""

import os
from pathlib import Path

import pytest

from pysidtracker import make_oracle_fixtures, register_grid

# Cache under the workspace (a Docker-daemon-visible path, and what CI persists
# via actions/cache). ``$PYSIDTRACKER_ORACLE_CACHE`` overrides the location.
_CACHE = Path(os.environ.get("PYSIDTRACKER_ORACLE_CACHE", ".oracle-cache"))

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
}


def _render(data, nframes):
    return register_grid(data, nframes, illegal_opcodes=True)


tune_id, oracle_match = make_oracle_fixtures(
    TUNES,
    hvsc_cache=_CACHE / "hvsc",
    oracle_cache=_CACHE / "csv",
    render=_render,
    frames=250,
)


@pytest.mark.oracle
def test_render_matches_oracle(oracle_match):  # noqa: F811
    oracle_match()
