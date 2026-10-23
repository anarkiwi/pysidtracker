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
there), the onset, lead and first diverging register, each side's last store to it,
and `host`. A failed subtune's file holds its error. A cache of another `format`
is refused by `build` and `rank` until rebuilt with `--force`.

## `rank`

Builds the spectrum (`faultloc.build_spectrum`, `results.csv` `info` fields as
`info:key=value` features), pairs the sides' features by their passing frames
(`correspond`), takes per failing tune its candidate causes (`candidates`):

- `first`: where the two executions first part in the onset frame (`partings`);
- otherwise `path`: a feature active at the onset whose partners are all inactive
  (or an unpartnered one its own tune never ran before);
- `value`: the sides ran partnered paths, so the diverging register's two last
  stores are the cause;
- `info`: a build option at least as suspicious;

and ranks the minimum cover (`rank_fixes`, exact `milp`). A `first` key's labels end
with `after:` its representative's last agreed pair. It writes `fixes.csv`
(`rank`, `id`, `kind`, `key`, `n_tunes`, `score`, `representative`, `labels`,
`tunes`) and prints the table and how many failing tunes the top 1/5/10 fixes are
credited with. Labels are the representative's disassembly and source lines.
`--explain N` writes `explain/<rank>.txt`: `whydiff.explain_divergence` of each
top fix's representative for a 6502 pair, `explain_cell` of both `python_side`s
otherwise.

### The first parting

An onset frame is mostly the first or second play call, where the sides' feature
sets differ almost everywhere: a set-based key then names the mismatch most tunes
share, a symptom. Each side's trace also ranks its features by first execution
in the call (`PathTrace.order`: the jennings step, or the Python event, that first
ran it), and `partings` walks the two onset sequences together:

1. keep the partnered features that are not `store:` sites (an unpartnered one is
   helper code the other side need not mirror) and map them to their
   correspondence class (`Correspondence.classes`, the components of the partner
   graph);
2. take both heads together while they are the same class or both are run
   somewhere by the other side in that frame (voices taken in another order),
   skipping later repeats of a taken class;
3. stop at the first head the other side never runs in the frame: the key is
   `("first", left head, right head)` (`""` for a side that has ended), with the
   last equal pair as context.

The key names features, not tunes or positions, so one root cause in any tune
yields one key. A tune whose sequences never part, or a cache without `rank`,
falls back to the set-based `path` and `value` keys. A parting whose two arms are
each run elsewhere in the frame by the other side (another voice) is taken as an
interleaving, and the walk continues to the next exclusive head.

## `xfails`

Merges one case per top fix (its representative) into a JSON manifest, sorted by
`id` with sorted keys; an existing `id` is kept as it is, so regenerating changes
only what is new. Fields: `id` (a digest of the fix key), `key`, `kind`,
`relpath`, `md5`, `song`, `frames` (the fewest reference frames reproducing the
onset verdict), `left`, `right`, `compare`, `reason` (the fix's labels and
whydiff's headline) and an optional `patch`, `[file_offset, byte]` pairs applied
to the `.sid` before running (a minimised or mutated input).

`--verify` (default) keeps a new case only when `check_case` fails it and a fresh
record of it has the case's key among its candidates; `--prune` drops existing
cases `check_case` now passes. `--hvsc-cache` and `--oracle-cache` (default
`<pair>/oracle`) are `check_case`'s.

### In a package's tests

```python
import pytest
from pysidtracker.pathdiff import check_case, xfail_params

@pytest.mark.parametrize("case", xfail_params("tests/pathdiff.json"))
def test_pathdiff(case):
    check_case(case, hvsc_cache=".hvsc-cache", oracle_cache=".oracle-cache")
```

Each case is `xfail(strict=True, raises=AssertionError)`: it fails with the
whydiff explanation until the fix lands, then passes and the strict xfail turns
the run red until the case is pruned. A 6502 pair compares the other side with
the sidtrace oracle grid (`testing.oracle_grid`, the patched tune written to a
temporary file) under the lead criterion; a Python pair compares both sides under
the case's `compare`. `check_case(..., reference=fn)` replaces the oracle with
`fn(path, song, frames)`, and `resolve=` the HVSC lookup (`testing.resolve_tune`).
Side specs are resolved as `build` resolves them, so a package's manifest names
importable `module:fn` sides.
