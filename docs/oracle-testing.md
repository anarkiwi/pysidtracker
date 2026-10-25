# Oracle testing

`make_oracle_fixtures` is a reusable, byte-exact test: it renders a tune with a
library's player and asserts the per-frame SID register grid matches
[`sidtrace`](https://github.com/anarkiwi/sidtrace) — a patched `sidplayfp` run in
Docker — frame for frame.

## How it works

1. **Resolve** each tune from a local `$HVSC` tree, else a cache dir, else an
   HVSC mirror download (`resolve_tune`). HVSC `.sid` files are copyright works
   and are never committed.
2. **Render the oracle** with `run_sidtrace` (the `anarkiwi/sidtrace` image),
   which streams a zstd CSV of every changed SID write annotated with
   cycle/interrupt timing. The CSV is cached per tune.
3. **Frame** the CSV into a per-frame grid (`sidtrace_grid`): the cadence is the
   median raise interval of the play source (`play_source`) taken from the CSV
   itself (`sidtrace_cadence`), so no cadence guessing is needed.
4. **Compare** the library render against the oracle grid with `aligned_match`.

The oracle is deterministic (`sidtrace` pins the simulated power-on delay), so a
given tune renders identically every run.

## Interrupt rows and handler nesting

A sidtrace (>= v0.6.0) row with empty `chip`/`reg` is an interrupt event:

- **raise** (`nmi`, `video_irq`, `cia_irq`): the source asserted its line
  (`c64::interruptIRQ`/`interruptNMI`). The CPU only samples it at the end of an
  instruction: an NMI is edge-triggered and always taken next, an IRQ is
  level-triggered and waits while the I flag is set (e.g. inside another
  handler, until its `RTI`), and is never taken if acknowledged first. A raise
  is therefore not the moment its handler starts running.
- **entry** (`nmi_entry`, `video_irq_entry`, `cia_irq_entry`, `brk_entry`): the
  CPU committed to an interrupt sequence and picked its vector, so an NMI that
  hijacks an IRQ or `BRK` is an `nmi_entry`. An IRQ entry names the source that
  pulled the line low (the earliest raise still asserting it).
- **rti**: an `RTI` executed, logged before any interrupt taken straight after.

`handler_sources(rows)` replays these as a stack: each entry pushes its source
(`brk_entry` the running one), each `rti` pops (an `rti` outside any handler is
ignored), and every row belongs to the innermost running handler, or `None`
outside all of them. `play_source(rows, chip)` is the source whose handlers
drive the most distinct SID registers, then make the most writes, so a digi
NMI interrupting the music IRQ never claims the IRQ's writes. A trace without
`rti` rows (a handler that never returned, or sidtrace < v0.6.0) cannot be
nested, so `handler_sources` reports `None` for every row and `play_source`
returns `None`; the cadence and grid helpers then fall back to the most raised
source. `cached_sidtrace` and `oracle_grid` reuse such a cached trace rather than
re-rendering it (which cannot add an `rti`); staleness across sidtrace upgrades
is decided by the caller's `oracle/STAMP` (the sidtrace image id).

A trace covers the play calls before the last raise of its timing source
(`sidtrace_calls`). `cached_sidtrace(..., calls=n)` re-renders a cached trace that
covers fewer than `n` calls and raises `ShortTrace` if a fresh render still falls
short; `oracle_grid` and `survey.oracle_frames` repeat the last write through the
calls after it but raise `ShortTrace` rather than pad past the end of the trace.
Renders that need `n` calls are sized by `render_seconds(play_seconds(tune, n,
song))`: the calls at the tune's `playroutine_cadence`, rounded up, plus a second.

## Using it in a format package

```python
# tests/test_oracle.py
import pytest
from pysidtracker import make_oracle_fixtures
from mypkg import MyPlayer

TUNES = {"demo": "MUSICIANS/X/Author/Demo.sid"}

def render(data, nframes):
    return MyPlayer(data).render_grid(nframes)   # a MemPlayer subclass

tune_id, oracle_match = make_oracle_fixtures(TUNES, render=render)

@pytest.mark.oracle
def test_render_matches_oracle(oracle_match):
    oracle_match()
```

`render(tune_bytes, nframes) -> grid` defaults to `register_grid` (the base jennings
renderer) when omitted.

## Explained failures

Pass `player(tune_bytes) -> MemPlayer` instead of `render` and a mismatch fails with
the [whydiff](whydiff.md) report: the diverging cell, the Python lines and 6502
instructions its value came from side by side, and both full traced histories
(`report_dir` also saves it as `<tune>.txt` / `.json`). For a single tune:

```python
from pysidtracker.testing import assert_oracle_match

@pytest.mark.oracle
@pytest.mark.xfail(strict=True, reason="detune carry")
def test_diverging_tune():
    assert_oracle_match("GAMES/S-Z/Your_Fathers_Nightmare.sid", lambda d: MyPlayer(d),
                        report_dir=".tunecache/why")
```

`song` (1-based, default 1) selects the subtune on both sides: sidtrace plays it
(`-o<song>`, cached as `<stem>_<song>.csv.zst`) and the jennings reference runs it.
`oracle_grid(..., song=None)`, as `make_oracle_fixtures` uses, plays the header's
start song. `assert_matches(data, player, reference)` is the same check against any
grid.

## Decode fidelity

Formats whose playback emulates the tune's own code (`EmuPlayer`/`TunePlayer`) never
exercise the decoded model against the oracle: a wrong decode still renders exact.
`pysidtracker.testing.decode_fidelity(original, rebuilt)` renders both the original
image and one rebuilt from the decoded model (the format's writer, or the tracker's
own packer) and returns the `Verdict`; `assert_decode_fidelity(data, rebuild)` is the
pytest form, and `survey.fidelity_row(row, data, rebuild)` adds the `fidelity` verdict
column (`exact`, `differs`, `not_decompiled`, `rebuild_error`; `FIDELITY_FIELDS`).

## Never skipped

An unavailable tune raises `TuneFetchError` and a missing/failed oracle raises
`SidtraceUnavailable` — a broken HVSC download or Docker oracle **fails** the
test instead of silently skipping and masking a regression.

## Parallelism and caching

The tests are `pytest-xdist` safe: each render uses a private mount directory and
places its cached CSV atomically, so workers sharing one cache directory never
collide. Tunes and traces default to the shared cache root
([caching.md](caching.md)), which the `cache` action persists in CI:

```yaml
- uses: anarkiwi/pysidtracker/.github/actions/cache@v0.30.0
  with: { key: "${{ hashFiles('tests/test_oracle.py') }}", sidtrace: "true", tunes: tests }
- run: pytest -m oracle -n auto
```

`pysid-tune-fetch --cache DIR --tests PATH` pre-caches every tune a suite names:
the `*.sid` string literals of the Python files under `PATH`, or the
whitespace-separated `*.sid` relpaths of a manifest file given directly (e.g. a
committed corpus sample list).

See this repo's [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) `oracle`
job and [`tests/test_oracle_hvsc.py`](../tests/test_oracle_hvsc.py) for a complete
example (the base package validates its own `register_grid` this way).
