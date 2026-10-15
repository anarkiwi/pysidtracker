# HVSC oracle survey

`pysidtracker.survey` compares a format package's renderer with the
[`sidtrace`](https://github.com/anarkiwi/sidtrace) oracle (sidplayfp in Docker)
for every subtune of every matching tune in an HVSC tree, over each subtune's
full HVSC song length (`DOCUMENTS/Songlengths.md5`).

```bash
pip install "pysidtracker[survey]"
python -m pysidtracker.survey find $HVSC --player 'goattracker|^\(gt' --out tunes.csv
python -m pysidtracker.survey run $HVSC tunes.csv --work .hvsc-survey --render mypkg.survey:render
python -m pysidtracker.survey report .hvsc-survey/results.csv
python -m pysidtracker.survey triage .hvsc-survey
python -m pysidtracker.survey diff $HVSC tunes.csv <md5>_<song> --work .hvsc-survey --render mypkg.survey:render
python -m pysidtracker.survey leads $HVSC tunes.csv <md5>_<song> --work .hvsc-survey
python -m pysidtracker.survey leads $HVSC tunes.csv --verdict differs --work .hvsc-survey
```

`run` also takes `--player REGEX`, `--limit N` (first N tunes), `--jobs`,
`--default-seconds` (subtunes with no HVSC length) and `--timeout` (render
seconds per subtune before it is a `library_error`, default 600). Player regexes are
case-insensitive.

## Renderer

A format package supplies only an importable callable

```python
def render(data: bytes, song: int, nframes: int) -> tuple[numpy.ndarray, dict]:
```

`data` is the `.sid` file and `song` is 1-based. It returns at least `nframes`
rows of the 25 per-frame SID registers (`$D400..$D418`, pulse-width highs
masked to a nibble). `nframes` already includes the `LEAD` frames the
alignment may shift by, so a renderer needs nothing from `pysidtracker.survey`.
The dict is recorded as the `info` column. Raising `pysidtracker.errors.SidParseError` (or a subclass)
means the tune was not recognised; any other exception is a library error.

## `find`

Identifies every `.sid` with `pysidtracker.sidid` (below): players are tried
in `sidid.cfg` order and the first with any matching signature names the tune,
which is then kept if `--player` matches that name. `--sidid-cfg` overrides the
bundled `sidid.cfg`. `tunes.csv` columns: `path` (relative to the HVSC root),
`md5`, `player`, `songs`, `clock` (`PAL`/`NTSC` from header flag bits 2-3),
`lengths` (seconds per song, space-separated).

## `match`

Find the HVSC tune for a native file (e.g. a CSDB editor dump) by byte-matching
their resident images — the matcher the `py*` format packages previously each
wrote as `match_hvsc.py`, and the input for the single-file oracle compare
(#47).

```bash
python -m pysidtracker.survey match $HVSC tunes.csv native1.sid native2.sid
```

`match` runs the tunes of a `find` CSV and each native file through the `body`
hook and compares them with `survey.match_images`. The default hook is the
PSID/RSID loaded image rebased to that image's load address; `--body
module:function` (e.g. a hook in a format library's `oracle_hvsc.py`) returns
the comparable bytes where editor and packed layouts differ by more than base.
Candidates and queries are compared at every rebase offset `d` with `|d|` up
to `--max-offset` (default 0x200, in ascending `|d|` order); the candidate
with the most equal bytes over the overlap wins (smallest `|d|` on ties, then
fewest differing bytes), and a candidate whose image length differs from the
query's by more than 0x200 bytes gets a penalty that cannot be overcome, so it
loses to any candidate of comparable length. The output is TSV: `native`,
`hvsc`, `length` (the query image length), `ndiff` (differing bytes over the
overlap at the winning offset), `offset`; `--out` writes it to a file instead
of stdout.

## `run`

Renders each subtune under sidtrace (`-o<song> -f8000`: register writes do not
depend on the sample rate; `--sidlite` is not used because its osc3/env3 reads
ignore the tune's SID model; sidtrace >= v0.5.0 logs every interrupt raise, so
oracle frames are the player's actual calls, including calls that write
nothing) and under the renderer, caching both as zstd register CSVs keyed
`<md5>_<song>` in `oracle/` and
`library/` under `--work`. Cached oracle traces are reused while `oracle/STAMP`
(the sidtrace image id, arguments and logged row kinds) matches; a change of sidtrace
wipes and re-renders them. A valid trace has no `rti` rows only when its handler never
returned (a main loop that set the I flag), and that is reused, not re-rendered.
`results.csv`:

| column | meaning |
| --- | --- |
| `key`, `path`, `player`, `song` | subtune identity |
| `seconds`, `frames` | song length and the oracle frames compared |
| `verdict` | `exact`, `differs`, `not_decompiled`, `library_error`, `oracle_error` |
| `lead` | frames the render is shifted against the oracle (negative: oracle leads) |
| `first_diff` | first differing oracle frame at that lead |
| `diff_frames`, `diff_regs` | count of differing frames and the registers involved |
| `min_lead`, `min_bad` | the lead with the fewest differing frames, and that count |
| `min_first`, `min_regs` | its first differing frame and the registers differing on its bad frames |
| `info` | the renderer's dict, as JSON |
| `oracle_gated`, `library_gated` | whether any voice's gate is ever set |
| `error` | the failure message for error verdicts |

The lead is the shift (up to `LEAD` frames either way) with the latest first
mismatch (the smallest `|lead|` on ties). A render wrong only on frame 0 has its
first mismatch at 0 at the true lead, so another lead can win and `diff_frames`
then covers nearly the whole song; `min_lead`/`min_bad` expose that case.
`report` lists the near-exact `differs` subtunes (`min_bad` at most `--near`,
default 3) grouped by bad-frame count, first bad frame, lead and registers.
Results written before these columns existed still report, without that list. `diff` prints the oracle (`O`) and render (`L`) register rows around
a subtune's first difference. The work directory derives from copyright tunes:
keep it untracked.

## `triage`

Classifies each `differs` subtune's first mismatch into `triage.csv` and
prints the tallies, so a format package's failure classes come from the tool
rather than ad-hoc scripts:

| column | meaning |
| --- | --- |
| `first_diff`, `when` | first differing frame: `frame0`, `early` (< 10), `mid` (< 250), `late` |
| `kind` | `silent` (the render never gates a voice the oracle gates), `misplaced` (an oracle value appears in another differing register), `offset±N` (two or more registers off by the same delta), `missing` (render zero), `extra` (oracle zero), else `value` |
| `roles` | voice-agnostic registers involved: `freq`, `pw`, `ctrl`, `ad`, `sr`, `fc`, `res`, `vol` |
| `diffs` | `[[reg, oracle, render], ...]` at that frame |

`triage.csv` also carries `min_lead`, `min_bad`, `min_first` and `min_regs`,
computed from the cached grids (so older `results.csv` files get them too), and
the near-exact list is printed as in `report`.

It also prints the exact rate for every renderer `info` key=value (keys with at
most 20 distinct values). A renderer that reports the playroutine options it
decoded therefore shows which options the player does not model yet.

## `leads`

Tabulates every lead from `-LEAD` to `LEAD`: the first differing frame, the
number of differing frames, and the registers differing on that first frame.
Cached `oracle/` traces and `library/` grids under `--work` are reused;
`--render` is needed only for a subtune without a cached library grid (it is
then rendered and cached). With one key it prints the table, marking the
chosen lead `*` and `min_lead` `<`. With several keys, `--all` (every `exact`
or `differs` subtune in `results.csv`) or `--verdict V` (repeatable), it scans
them in `--jobs` processes into `leads.csv` (`--out` overrides), one row per
subtune and lead (`key`, `path`, `song`, `frames`, `lead`, `first_diff`, `bad`,
`regs`), and prints the subtunes whose `lead` and `min_lead` differ and the
near-exact list.

## `pysidtracker.sidid`

A pure-Python [SIDId](https://github.com/cadaver/sidid): players are tried in
`sidid.cfg` order, `??` matches any byte and `AND` skips any bytes. SIDId's
`sidid.cfg` is bundled (BSD-3, see `src/pysidtracker/data/SIDID-LICENSE`).

```bash
pysidid $HVSC                       # path<TAB>player, tallies on stderr
pysidid $HVSC --player goattracker  # only matching players
pysidid tune.sid -m                 # every matching player (SIDId -m)
```

`identify(data)` / `identify_all(data)` return the first / every matching
player name for bytes, using the bundled signatures unless `sigs` are passed.
