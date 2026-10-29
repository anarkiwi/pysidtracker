# Byte-stream grammars, pointer tables, relocation and corpus snapshots

Tracker song data is mostly byte streams (patterns, orderlists, instrument or
arpeggio programs) that a player walks with an 8-bit cursor from a split lo/hi
pointer table. These modules give every format the same machinery for reading,
validating, printing and relocating them.

## `pysidtracker.bytegrammar`

A stream is parsed as text, one latin-1 character per byte, by a Lark LALR
parser with the contextual lexer, so a terminal's byte range can depend on the
rule being parsed.

```python
from lark import Transformer, v_args
from pysidtracker import ByteGrammar, Parsed

@v_args(inline=True)
class ToModel(Transformer):
    def orderlist(self, *items):          # each start rule returns Parsed
        model = Orderlist(...)
        return Parsed(model, len(model.to_bytes()))

GRAMMAR = ByteGrammar(
    __package__, "mytracker.lark", ("pattern", "orderlist"), ToModel(), MyParseError
)
GRAMMAR.parse_prefix(data, "orderlist")   # Parsed(model, size); rest never read
GRAMMAR.parse_exact(blob, "orderlist")    # model; blob must be one whole stream
GRAMMAR.tree(data, "pattern")             # lark Tree for printing
```

The grammar file is a package resource. It takes the shared terminals with

```
%import bytes (BYTE, _TAIL)
```

from `pysidtracker/grammars/bytes.lark`: `BYTE` is any one byte, and `_TAIL`
(filtered from trees) is whatever follows a stream's terminator, so a start
rule ends `... TERMINATOR _TAIL?` and parses a whole cursor window. Sibling
grammars in the format's own package are importable too. The two Lark parsers
(transformer results, plain trees) are built once per `ByteGrammar`.

A stream that does not parse raises the grammar's `error` class (default
`SidParseError`): `<rule> has no terminator within N bytes`, or `<rule>:
unexpected byte $xx at offset k`.

Pointer-table reads, against a `SidImage`:

| call | result |
|------|--------|
| `img.window(addr, size)` | the `size` bytes from `addr`, zero past `$FFFF` |
| `GRAMMAR.read(img, addr, rule, what=None, size=REACH)` | `Parsed` stream at `addr`; errors name `what at $addr` |
| `GRAMMAR.entry(img, lo, hi, i, rule, what=None, size=REACH)` | the stream entry `i` of the `lo`/`hi` table points at |
| `GRAMMAR.stored(img, lo, hi, i, rule, region, size=REACH)` | that stream's model if the pointer lies in `region` and the bytes (up to `region.stop`) are one stream, else `None` |
| `img.put_ptr(lo, hi, i, addr)` | the write counterpart of `img.ptr(lo, hi, i)` |

`REACH = 0x100` is an 8-bit `Y` cursor's reach.

Validation and printing:

- `GRAMMAR.check(streams, ValidationError, skip_equality=())` takes
  `(name, model, rule)` triples and raises `ValidationError` unless each
  `model.to_bytes()` parses back as exactly one `rule` stream equal to `model`;
  rules in `skip_equality` (a model the transformer returns in another shape,
  such as an event list) only need to parse.
- `GRAMMAR.listing(streams)` yields `name:` and the indented `render(tree)` of
  each stream, one `TERMINAL=$xx` item per byte. `add_grammar_command(commands,
  handler, song_help=...)` adds the `grammar SONG [--subtune N]` subcommand that
  prints it.
- `select_rule(variant, ((facet, rule), ...), default)` picks the start rule of
  the first facet the build's `Variant` (or a mapping, or a resolved variant's
  attributes) sets, e.g. the orderlist rule with a build's extra commands.

## Writers' helpers

- `reraise_as(ValidationError, catch=SidParseError)` turns a parse error met
  while a writer re-reads its own output into its validation error.
- `require_ram(start, length, what, error=SidParseError)` refuses data reaching
  the C64 I/O area (`$D000-$DFFF`).
- `load_prefix(addr)` is a `.prg`'s two-byte load address.
- `address` is the argparse type for `$C000` / `0xC000` / decimal addresses.

## `pysidtracker.relocate`

- `walk_code(image, starts, lo, hi)` is `{addr: length}` of every instruction
  the code at `starts` reaches inside `[lo, hi)`: `JSR`, `JMP` and branch
  targets are followed, `CLC; BCC`, `SEC; BCS` and `CLV; BVC` are taken as
  unconditional, and `JMP`, `RTS`, `RTI`, `BRK`, `JMP ()` and `JAM` end a path.
- `relocate_operands(image, code, moved, delta, remap=None, forbidden=(),
  error=SidParseError)` is `{operand addr: new target}` of each absolute operand
  of `code` that changes: targets in `moved` shift by `delta`, targets in
  `remap` are replaced, and a target in `forbidden` that neither covers raises
  `error` (data the relocation does not place).

## `pysidtracker.snapshot`

A corpus snapshot records, per tune, the digest of the decoded model and of
every probe of it (a rebuilt image, an export), or the error each step raised.
Two snapshots taken before and after a refactor show which tunes it changed.

```bash
pysidtracker snapshot build CORPUS... -o before.json \
    [--format NAME|module:attr] [--probe NAME=module:attr ...] [--jobs N]
pysidtracker snapshot diff before.json after.json   # exit 1 on any difference
```

Probes come from the format's `SidFormat.probes` (`{name: callable(model)}`)
plus `--probe`; `--format` is an installed format name, `auto`, or a
`module:attr` naming a `SidFormat` or a factory of one. Inputs are files,
directories and `find` CSVs as for `variant census`. Models are digested by
`repr`, `bytes` results by content (BLAKE2b-128). Python API: `fingerprint(data,
fmt, probes)`, `snapshot(tunes, fmt, probes, workers)`, `diff(old, new)`,
`read_snapshot` / `write_snapshot`.
