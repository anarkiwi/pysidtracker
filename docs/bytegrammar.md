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
  prints it; a format whose `SidFormat` sets `grammar` and `streams(model)`
  (its `(name, model, rule)` triples) gets it from the main CLI, `pysidtracker
  grammar SONG [--subtune N] [--format NAME]` (see [formats.md](formats.md)).
- `select_rule(variant, ((facet, rule), ...), default)` picks the start rule of
  the first facet the build's `Variant` (or a mapping, or a resolved variant's
  attributes) sets, e.g. the orderlist rule with a build's extra commands.

## Bitmap-selected fields

Some records are a bitmap byte followed by one value per set bit, in a fixed
order (defMON's sidTAB rows: low bits 6, 7, 5, 4, 3, 2, 1 select WGh, WGl, AD,
SR, TR, AF, PW; then a high bitmap whose bits 7-4 select a byte each and bit 3
a 2-byte ACID). An LALR grammar could only enumerate one terminal class per
bitmap value. Instead, `bitmaps` names each bitmap terminal and its
`BitField(bit, tag, width=1)` items in value order:

```python
ROWS = ByteGrammar(__package__, "sidtab.lark", ("row",), ToRow(), bitmaps={
    "LOWMAP": ((6, "T_WGH", 1), (7, "T_WGL", 1), (5, "T_AD", 1), ...),
    "HIGHMAP": (..., (3, "T_ACID", 2)),
})
```

```
row: low high _TAIL?
low: LOWMAP wgh? wgl? ad? sr? tr? af? pw?
high: HIGHMAP ps? re? fv? cp? acid?
wgh: T_WGH BYTE
acid: T_ACID BYTE BYTE
LOWMAP: /[\x00-\xff]/
HIGHMAP: /[\x00-\xff]/
%declare T_WGH T_WGL T_AD T_SR T_TR T_AF T_PW T_PS T_RE T_FV T_CP T_ACID
```

`BitmapPostLex` (a Lark postlexer, usable on any Lark parser) passes the bitmap
token through, then for each set bit yields a zero-width tag token followed by
the `width` value tokens, lexed lazily in the parser's state after the tag.
Bits without a `BitField` select nothing. Tags carry no byte, so `size`, token
`end_pos` and error offsets still count raw bytes, a stream truncated inside
the values reads as having no terminator, and `render` prints a tag by name.

## An LALR pitfall: `_TAIL` after a repeated rule

When `_TAIL` can follow a repeated rule, LALR merges the lookaheads of every
repetition, so the contextual lexer accepts `_TAIL` after the first step and
swallows the rest of the stream as tail:

```
packed: step~0..31 alt _TAIL?          // _TAIL is lexed after step 1
```

Give the final position its own rule, so only it is followed by `_TAIL`:

```
packed: step~0..31 alt _TAIL?
      | step~31 last _TAIL?
last: ...                              // the same expansions as step
```

## Writers' helpers

- `reraise_as(ValidationError, catch=SidParseError)` turns a parse error met
  while a writer re-reads its own output into its validation error.
- `require_ram(start, length, what, error=SidParseError)` refuses data reaching
  the C64 I/O area (`$D000-$DFFF`).
- `load_prefix(addr)` is a `.prg`'s two-byte load address.
- `address` is the argparse type for `$C000` / `0xC000` / decimal addresses.
- `fits(value, encode, decode, what, error=SidFormatError)` raises `error`
  unless `decode(bytes(encode(value))) == value`: `"{what}: {value!r} has no
  bytes (...)"` when encoding or decoding raises one of `CODEC_ERRORS`
  (`ValueError`, `KeyError`, `TypeError`, `OverflowError` and every `SidError`,
  so a format's own errors), else `"... reads back as {back!r}"`. It checks each
  model element in the context its bytes are read in.

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

A corpus snapshot records, per tune, the digest of the decoded model, of its
`render` (`fmt.player(model)` over the start song's whole length, `song_frames`)
and of every probe of the model (a rebuilt image, an export), or the error each
step raised.
Two snapshots taken before and after a refactor show which tunes it changed.

```bash
pysidtracker snapshot build CORPUS... -o before.json \
    [--format NAME|module:attr] [--probe NAME=module:attr ...] [--jobs N] [--work SURVEY_WORK]
pysidtracker snapshot diff before.json after.json   # exit 1 on any difference
```

The built-in `render` digest means a format needs no render probe; a probe must
never render a part of a song. Probes come from the format's `SidFormat.probes` (`{name: callable(model)}`),
plus its canonical text as `canonical` when the format sets the `canonical`
hook and `probes` has no `canonical` ([formats.md](formats.md)), plus `--probe`; `--format` is an installed format name, `auto`, or a
`module:attr` naming a `SidFormat` or a factory of one. Inputs are files,
directories and `find` CSVs as for `variant census`. Models are digested by
`repr`, `bytes` results by content (BLAKE2b-128). Python API: `fingerprint(data,
fmt, probes)`, `snapshot(tunes, fmt, probes, workers)`, `diff(old, new)`,
`read_snapshot` / `write_snapshot`.

A committed snapshot also states a corpus test's expected outcomes:
`expected_errors(snap)` is `{tune: message}` of the error each model raised
(the text after `!Error: `), `None` where it decoded. In the suite,
`testing.corpus_params(tunes, expected)` is one `pytest.param(rel, message)` per
relative path, its id the path, and `testing.check_outcome(load, path, message,
error_type=SidError)` asserts `load(path)` raises `error_type` with exactly that
message, or decodes cleanly when it is `None`:

```python
EXPECTED = expected_errors(read_snapshot(SNAPSHOT))

@pytest.mark.parametrize("rel,message", corpus_params(SAMPLE, EXPECTED))
def test_corpus_tune(rel, message):
    check_outcome(decompile, resolve_tune(rel), message, MySidParseError)
```
