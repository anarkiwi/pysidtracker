# pathdiff: the fewest fixes for a whole survey

`whydiff` explains one divergence. `pathdiff` takes every divergent (and exact)
subtune of a survey, records the code paths both players ran each frame, and
localises the divergences over the whole corpus to the smallest set of fixes, each
with a representative tune and a strict xfail case.

```bash
python -m pysidtracker.pathdiff build $HVSC tunes.csv --work .hvsc-survey \
    --left 6502 --right scripts/oracle_hvsc.py:player --jobs 28
python -m pysidtracker.pathdiff rank --work .hvsc-survey \
    --right scripts/oracle_hvsc.py:player --explain 5
python -m pysidtracker.pathdiff xfails --work .hvsc-survey \
    --right scripts/oracle_hvsc.py:player --manifest tests/pathdiff.json --top 10
```

Inputs the corpus lacks (a branch outcome no tune reaches, a smaller reproducer
of a case) come from `pathfuzz` ([pathfuzz.md](pathfuzz.md)), on one tune with the
same `--right` specs; its patches are `xfails`' `patch` field:

```bash
python -m pysidtracker.pathfuzz gaps     tune.sid --right SPEC [--song N] [--frames N]
python -m pysidtracker.pathfuzz exercise tune.sid --right SPEC [--side 6502|python] [--budget N]
python -m pysidtracker.pathfuzz minimise tune.sid --right SPEC [--patch mutants.json#k]
```

## Sides

`--left` (default `6502`) is the reference, `--right` the player under test.
A side is one of:

| spec | player | feature prefix |
| --- | --- | --- |
| `6502` | the tune on jennings (`paths.cpu_paths`) | `6502:` |
| `format:NAME` | an installed format's `SidFormat.play(data, song - 1)` | `NAME:` |
| `module:fn`, `path.py:fn` | `fn(data, song)`: a `MemPlayer`, or a `paths.PySubject` | `python:`, or the subject's `label` |

A `path.py` is imported as its stem's module with its own directory on `sys.path`.
Two sides with one label are prefixed `<label>-left:` and `<label>-right:`. A
`PySubject` plugs in any player with `play_frame`: its `cell(player, register)`
names where a register is held, `sources` maps exec'd code's pseudo-filenames to
their text, and `starts(player)` returns each voice's first-note frame (for
`--compare onset`). A unitracker lift, for example:

```python
def lift(data, song):
    lifted = backends.lifter(data, song - 1)
    lifted.run()
    text = lifted.document
    return PySubject(lambda: Document(load(text)), cell=lambda p, r: (p.regs, r),
                     label="lift", starts=lambda p: p.starts)
```

(`Document` is unitracker-player's `_Onsets` with `play_frame = step`.)

## Comparison

- `--compare lead` (default): `gridcmp.verdict_lead` up to `LEAD` frames of shift;
  frames are aligned by its lead.
- `--compare onset`: `gridcmp.verdict_onset` (carry-tolerant), each voice counted
  from its first note: the right, else the left side's `starts`, else the
  voice's first gated frame in the left grid. Frames are not shifted.

A `MemPlayer`'s grid is its register file (`regs`, its `snapshot`), any other
player's its `cell`s; pulse-width highs are masked to their nibble on both sides.
The onset frame fails, the frames before it pass, later frames are dropped. With
a `6502` side and the survey's `oracle/<key>.csv.zst` present, jennings is checked
against the sidtrace oracle: where it parts at or before the onset, the tune keeps
only the frames before that and counts as `host_mismatch`.

## `build`

Selects `results.csv` rows by `--verdict` (default `differs` and `exact`: exact tunes
are the pass evidence that pairs the two sides' paths); without a `results.csv`,
every song of `tunes.csv` is a candidate. Each subtune is recorded over its song
length up to `--max-frames` (default 3000), in `pysidtracker.jobs` workers with a
`--timeout` deadline, skipping keys already cached unless `--force`.

`work/pathdiff/<left>-<right>/` holds `meta.json` (cache `format`, sides, compare
mode, HVSC root, prefixes), `vocab.txt` (the pair's feature names, line `i` naming
id `i`) and `tunes/<key>.npz` per subtune: its distinct frames as CSR over feature
ids with their frame counts (the onset row last), the features' texts in this tune,
each feature's first-occurrence `rank` on its side in the onset frame (-1: not run
there), its `relevant` features and each feature's `decision` (below), the onset,
lead and first diverging register, each side's last store to it, and `host`. A failed subtune's file holds its error. A cache of another `format`
is refused by `build` and `rank` until rebuilt with `--force`.

## `rank`

Builds the spectrum (`faultloc.build_spectrum`, `results.csv` `info` fields as
`info:key=value` features), pairs the sides' features by their passing frames
(`correspond`), takes per failing tune its candidate causes (`candidates`):

- `first`: where either side's onset-frame execution first departs (`partings`);
- otherwise `path`: a feature active at the onset whose partners are all inactive
  (or an unpartnered one its own tune never ran before);
- `value`: the sides ran partnered paths, so the diverging register's two last
  stores are the cause;
- `info`: a build option at least as suspicious;

and ranks the minimum cover (`rank_fixes`, exact `milp`). A `first` key's labels end
with its representative's context (below). It writes `fixes.csv`
(`rank`, `id`, `kind`, `key`, `n_tunes`, `score`, `representative`, `labels`,
`tunes`) and prints the table and how many failing tunes the top 1/5/10 fixes are
credited with. Labels are the representative's disassembly and source lines.
`--explain N` writes `explain/<rank>.txt`: `whydiff.explain_divergence` of each
top fix's representative for a 6502 pair, `explain_cell` of both `python_side`s
otherwise.

### The first departure

An onset frame is mostly the first or second play call, where the sides' feature
sets differ almost everywhere: a set-based key then names the mismatch most tunes
share, a symptom. Each side's trace also ranks its features by first execution
in the call (`PathTrace.order`: the jennings step, or the Python event, that first
ran it), and `partings` takes, per side, the first partnered non-`store:` feature
of the onset frame none of whose partners the other side runs in that frame: where
that side's execution left the other's. Shared features may run in any order on
either side (voices taken in another order), and unpartnered ones (helper code the
other side need not mirror) are skipped.

### Relevant departures

The earliest departure is often in code that does not feed the diverging register
(a filter routine running first each frame while a voice frequency diverges), and
code that runs every frame has different partners in different builds of one
player, splitting one cause into several keys. So `build` also records, per failing
tune, the features the diverging register depends on (`pysidtracker.relevance`):

- each side's onset frame is sliced back from its last store to the register (or
  from `last_change`'s frame when it stored none), as whydiff does: jennings'
  `cpu_slice` over the `cpu_history` dataflow, `PyTrace`'s slice on a Python side;
- each slice step also keeps its dynamic control parent: the innermost branch whose
  region, up to its immediate post-dominator in the same call, it ran in. Post-dominators
  (`networkx.immediate_dominators` on the reversed graph) are over the static CFG
  (`StaticCode.cfg` and both arms of each branch; coverage.py's line arcs of each
  traced file, `sources` for exec'd code) and the executed edges, a call continuing at
  its return site and a return leaving to the exit. A kept branch keeps its
  condition's inputs, so the slice closes over data and control dependence;
- the edge each kept step took is a relevant feature, keyed as `paths` keys it.

Each edge feature also carries its `decision`, the branch it leaves (`dec:` and its
source block or line). A departure is relevant when its decision is that of a
relevant feature of either side, or of a partner of one: a side that never stored the
register (silent) has no slice of its own, and the other side's slice names the
decision through the correspondence. Where some departure of a tune is relevant,
`partings` takes the earliest relevant one per side (`Parting.relevant`) and a side
with none gives no `first` key; otherwise every departure counts.

Each side's departure is a candidate, `("first", left, "")` and
`("first", "", right)`, so one root cause yields one key in every tune even where
the other side's code differs by build; the cover keeps whichever explains more
tunes. Its labels end with `after:` the departing side's last feature with a
running partner and `vs:` the other side's departure. A tune where every feature
has a running partner, or a cache without `rank`, falls back to the set-based
`path` and `value` keys. Partners are paired by passing frames, so a branch arm
whose partner the other side also runs for another voice in the same frame is not
a departure.

The slice covers the onset frame only: a register a play call copies from a buffer
an earlier call computed (a ghost register file) depends on that copy alone, and its
tune falls back to every departure. `PyTrace` follows scalars, so a frame that hands
its register file to `_mem` in one bulk copy leaves only the other side's slice.

## `xfails`

Merges one case per top fix (its representative) into a JSON manifest, sorted by
`id` with sorted keys; an existing `id` is kept as it is, so regenerating changes
only what is new. Fields: `id` (a digest of the fix key), `key`, `kind`,
`relpath`, `md5`, `song`, `frames` (the fewest reference frames reproducing the
onset verdict), `left`, `right`, `compare`, `reason` (the fix's labels and
whydiff's headline) and an optional `patch`, `[file_offset, byte]` pairs applied
to the `.sid` before running (a minimised or mutated input).

`--as-left`/`--as-right` set the side specs written into each case (and so the
specs `check_case` resolves), e.g. an importable `module:fn` for a script path the
pair was built with; `--verify` and `--prune` still run the pair's own specs.

`--verify` (default) keeps a new case only when `check_case` fails it and a fresh
record of it has the case's key among its candidates; `--prune` drops existing
cases `check_case` now passes. Tunes come from the pair's HVSC root; the oracle
traces are cached in `--oracle-cache` (default `default_cache("csv")`, see
[caching.md](caching.md)).

### In a package's tests

```python
import pytest
from pysidtracker.pathdiff import check_case, xfail_params

@pytest.mark.parametrize("case", xfail_params("tests/pathdiff.json"))
def test_pathdiff(case):
    check_case(case)
```

Each case is `xfail(strict=True, raises=AssertionError)`: it fails with the
whydiff explanation until the fix lands, then passes and the strict xfail turns
the run red until the case is pruned. A 6502 pair compares the other side with
the sidtrace oracle grid (`testing.oracle_grid`, the patched tune written to a
temporary file) under the lead criterion; a Python pair compares both sides under
the case's `compare`. `check_case(..., reference=fn)` replaces the oracle with
`fn(path, song, frames)`, and `resolve=` the HVSC lookup (`testing.resolve_tune`).
Tunes (`hvsc_cache`) and oracle traces (`oracle_cache`) default to the shared
cache (`testing.default_cache`, [caching.md](caching.md)).
Side specs are resolved as `build` resolves them, so a package's manifest names
importable `module:fn` sides.
