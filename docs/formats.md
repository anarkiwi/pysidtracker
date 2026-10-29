# Format hooks: grammar listings and semantic equivalence

A format package registers a `SidFormat` (`pysidtracker.formats`). Besides
`parser` and `player`, two optional hooks give every format the same
inspection commands in the main `pysidtracker` CLI.

| field | type | backs |
|-------|------|-------|
| `grammar` | `ByteGrammar` | `pysidtracker grammar` |
| `streams` | `model -> Iterable[(name, model, rule)]` | `pysidtracker grammar` |
| `canonical` | `model -> str` | `pysidtracker equiv`, the `canonical` snapshot probe |

```python
SidFormat(
    "mytracker", MyParser(), MyPlayer,
    grammar=GRAMMAR,
    streams=lambda song: [(f"pattern {i}", p, "pattern") for i, p in enumerate(song.patterns)],
    canonical=lambda song: str(Tune.from_song(song).canonical()),
)
```

## `grammar`

```bash
pysidtracker grammar SONG [--subtune N] [--format NAME]
```

prints `SidFormat.listing(model)`, the `ByteGrammar.listing` of the model's
streams ([bytegrammar.md](bytegrammar.md)). `--format` is an installed format
name, a `module:attr` naming a `SidFormat` (or a factory of one), or `auto`
(the format recognising the tune). A format without `grammar` and `streams`
fails with `error: format 'NAME' has no byte grammar`. A format's own
contributed command of the same name is shadowed by the shared one.

## `canonical` and `equiv`

`canonical(model)` is a text form of what the model plays, equal for two models
exactly when they play the same: unreachable data dropped, indistinguishable
table rows merged (`automaton.minimise`, [automaton.md](automaton.md)) and
everything renumbered by first use.

```bash
pysidtracker equiv A B [--subtune N] [--format NAME]
```

exits 0, silently, when both tunes' canonical texts are equal, else prints
their unified diff and exits 1. `--subtune` defaults to each tune's start song.
`SidFormat.canonical_text(model)` raises `SidFormatError` without the hook.

`SidFormat.snapshot_probes()` is `probes` plus, when the hook is set and
`probes` names no `canonical`, the UTF-8 canonical text as the `canonical`
probe, so `pysidtracker snapshot` records it for every tune.
