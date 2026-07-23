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
   median interrupt-raise interval taken from the CSV itself (`sidtrace_cadence`),
   so no cadence guessing is needed.
4. **Compare** the library render against the oracle grid with `aligned_match`.

The oracle is deterministic (`sidtrace` pins the simulated power-on delay), so a
given tune renders identically every run.

## Using it in a format package

```python
# tests/test_oracle.py
import pytest
from pysidtracker import make_oracle_fixtures
from mypkg import MyPlayer

TUNES = {"demo": "MUSICIANS/X/Author/Demo.sid"}

def render(data, nframes):
    return MyPlayer(data).render_grid(nframes)   # a MemPlayer subclass

tune_id, oracle_match = make_oracle_fixtures(
    TUNES,
    hvsc_cache=".oracle-cache/hvsc",
    oracle_cache=".oracle-cache/csv",
    render=render,
)

@pytest.mark.oracle
def test_render_matches_oracle(oracle_match):
    oracle_match()
```

`render(tune_bytes, nframes) -> grid` defaults to `register_grid` (the base jennings
renderer) when omitted.

## Never skipped

An unavailable tune raises `TuneFetchError` and a missing/failed oracle raises
`SidtraceUnavailable` — a broken HVSC download or Docker oracle **fails** the
test instead of silently skipping and masking a regression.

## Parallelism and caching

The tests are `pytest-xdist` safe: each render uses a private mount directory and
places its cached CSV atomically, so workers sharing one cache directory never
collide. Point `hvsc_cache` / `oracle_cache` at a workspace directory and cache
it in CI:

```yaml
- uses: actions/cache@v4
  with: { path: .oracle-cache/hvsc, key: hvsc-${{ hashFiles('tests/test_oracle.py') }} }
- run: docker pull anarkiwi/sidtrace:latest
- uses: actions/cache@v4
  with: { path: .oracle-cache/csv, key: oracle-csv-${{ steps.image.outputs.digest }} }
- run: pytest -m oracle -n auto
```

See this repo's [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) `oracle`
job and [`tests/test_oracle_hvsc.py`](../tests/test_oracle_hvsc.py) for a complete
example (the base package validates its own `register_grid` this way).
