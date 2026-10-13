# HVSC oracle survey

`pysidtracker.survey` compares a format package's renderer with the
[`sidtrace`](https://github.com/anarkiwi/sidtrace) oracle (sidplayfp in Docker)
for every subtune of every matching tune in an HVSC tree, over each subtune's
full HVSC song length (`DOCUMENTS/Songlengths.md5`).

```bash
pip install "pysidtracker[survey]"
python -m pysidtracker.survey find $HVSC --sidid-cfg sidid.cfg --player 'goattracker|^\(gt' --out tunes.csv
python -m pysidtracker.survey run $HVSC tunes.csv --work .hvsc-survey --render mypkg.survey:render
python -m pysidtracker.survey report .hvsc-survey/results.csv
python -m pysidtracker.survey diff $HVSC tunes.csv <md5>_<song> --work .hvsc-survey --render mypkg.survey:render
```

`run` also takes `--player REGEX`, `--limit N` (first N tunes), `--jobs` and
`--default-seconds` (subtunes with no HVSC length). Player regexes are
case-insensitive.

## Renderer

A format package supplies only an importable callable

```python
def render(data: bytes, song: int, nframes: int) -> tuple[numpy.ndarray, dict]:
```

`data` is the `.sid` file, `song` 1-based, and the grid has at least
`nframes + survey.LEAD` rows of the 25 per-frame SID registers
(`$D400..$D418`, pulse-width highs masked to a nibble). The dict is recorded as
the `info` column. Raising `pysidtracker.errors.SidParseError` (or a subclass)
means the tune was not recognised; any other exception is a library error.

## `find`

Identifies every `.sid` with [SIDId](https://github.com/cadaver/sidid)
semantics: players are tried in `sidid.cfg` order and the first with any
matching signature names the tune, which is then kept if `--player` matches
that name. `tunes.csv` columns: `path` (relative to the HVSC root), `md5`,
`player`, `songs`, `clock` (`PAL`/`NTSC` from header flag bits 2-3), `lengths`
(seconds per song, space-separated).

## `run`

Renders each subtune under sidtrace (`-o<song> -f8000 --sidlite`: register
writes do not depend on the SID model or sample rate) and under the renderer,
caching both as zstd register CSVs keyed `<md5>_<song>` in `oracle/` and
`library/` under `--work`. Cached oracle traces are reused. `results.csv`:

| column | meaning |
| --- | --- |
| `key`, `path`, `player`, `song` | subtune identity |
| `seconds`, `frames` | song length and the oracle frames compared |
| `verdict` | `exact`, `differs`, `not_decompiled`, `library_error`, `oracle_error` |
| `lead` | frames the render is shifted against the oracle (negative: oracle leads) |
| `first_diff` | first differing oracle frame at that lead |
| `diff_frames`, `diff_regs` | count of differing frames and the registers involved |
| `info` | the renderer's dict, as JSON |
| `oracle_gated`, `library_gated` | whether any voice's gate is ever set |
| `error` | the failure message for error verdicts |

The lead is the shift (up to `LEAD` frames either way) with the latest first
mismatch. `diff` prints the oracle (`O`) and render (`L`) register rows around
a subtune's first difference. The work directory derives from copyright tunes:
keep it untracked.
