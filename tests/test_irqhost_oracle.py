"""The interrupt host's trace against a live sidtrace render, row for row.

Marked ``oracle`` (Docker + network, run by the CI oracle job): every synthetic
tune in :mod:`tests.irqhost_tunes` and real HVSC RSIDs whose digis run from
NMIs or main loops must log exactly the rows sidplayfp logs, over the whole song.
"""

import os

import pytest

from pysidtracker.irqhost import InterruptHost
from pysidtracker.oracle import cached_sidtrace
from pysidtracker.survey import ORACLE_ARGS
from pysidtracker.testing import default_cache, resolve_tune, song_length, whole_song

from . import irqhost_tunes as tunes

_CACHE = default_cache("csv") / "irqhost"
HVSC_RSIDS = {
    "nmi_digi": ("DEMOS/A-F/Axel-F_Remix.sid", 1),
    "main_loop_digi": ("MUSICIANS/M/Mueller_Markus/Digi_Sample_Music.sid", 2),
    "raster_digi": ("MUSICIANS/B/Bain_David/Aftermath.sid", 1),
}


def _assert_same_trace(sid, seconds, song=1, name=None):
    trace = _CACHE / f"{name or sid.stem}_{song}.csv.zst"
    user = f"{os.getuid()}:{os.getgid()}"
    want = cached_sidtrace(
        sid, trace, seconds=seconds + 1, song=song, user=user, extra_args=ORACLE_ARGS
    )
    host = InterruptHost(sid.read_bytes(), song - 1)
    end = host.start + int(seconds * host.clock_hz)
    host.run(end + 1 - host.cycle)
    got = [r for r in host.rows() if r.cycle <= end]
    want = [r for r in want if r.cycle <= end]
    first = next((i for i, (w, g) in enumerate(zip(want, got)) if w != g), None)
    assert first is None and len(got) == len(want), (first, len(want), len(got))


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(tunes.TUNES))
def test_synthetic_trace_matches_sidtrace(name):
    sid = _CACHE / f"{name}.sid"
    sid.parent.mkdir(parents=True, exist_ok=True)
    sid.write_bytes(tunes.build(name))
    _assert_same_trace(sid, whole_song(2))


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(HVSC_RSIDS))
def test_hvsc_rsid_trace_matches_sidtrace(name):
    relpath, song = HVSC_RSIDS[name]
    sid = resolve_tune(relpath)
    assert sid is not None, f"{relpath} unreachable"
    _assert_same_trace(sid, song_length(sid, song), song, name)
