# Text grammars of typed models

A format's typed song model (frozen dataclasses: rows, commands, table steps) is
printed and parsed as text in a Lark grammar whose constructor rules name the
dataclasses they build. `loads(dumps(model)) == model` then shows the grammar
states everything the model holds.

## `pysidtracker.textgrammar`

```
// mytracker/song.lark
start: entry*
entry: INT ":" (jump | wave_note | stop)
jump: "Jump" "(" INT ")"
wave_note: "WaveNote" "(" INT ("," RELATIVE)? ")"
stop: "Stop"
none: "none"
RELATIVE: "relative"
%import text (INT, STRING, COMMENT, WS)
%ignore WS
%ignore COMMENT
```

```python
from pysidtracker.textgrammar import (
    ModelTransformer, TextGrammar, classes, hexint, seq, term,
)

class Build(ModelTransformer):
    classes = classes(typed)                # {"wave_note": WaveNote, ...}

    def entry(self, children):
        return tuple(children)

    def start(self, children):
        return seq(children, 1, "entry", MyParseError)

GRAMMAR = TextGrammar(__package__, "song.lark", Build, MyParseError)
GRAMMAR.loads(text)
"\n".join(f"{hexint(n)}: {term(op)}" for n, op in enumerate(ops, 1))
```

| name | role |
|------|------|
| `classes(module, *extra)` | `{rule: dataclass}` of `module`'s own dataclasses and `extra`, the rule being `rule_name(cls)`, the class name in snake_case (`SetAD`: `set_ad`) |
| `ModelTransformer` | its `__default__` builds `classes[rule]`: the non-token children fill the non-`bool` init fields in order, and a `bool` field is True when a token of type `FIELD.upper()` is a child; a rule without a class stays a `Tree`. `INT` (`$hex`, decimal, negative), `STRING` (JSON) and `none` are values |
| `term(value)` | the printer: `Name(arg, ...)` of the init fields with true `bool` fields by name, a constructor without arguments bare, ints `$XX`/`-$XX` (`hexint`), strings as JSON, `None` as `none` |
| `seq(items, base, what, error)` | the values of `(index, value)` items, raising `error` (`"{what} 3 where 2 is due"`) unless the indices count up from `base` |
| `TextGrammar(package, resource, transformer_cls, error)` | `loads(text)`: the LALR parse, transformed; a `LarkError`, or a `VisitError` from a model's own validation, raises `error` |

The shared terminals live in `pysidtracker/grammars/text.lark`: `INT`,
`STRING`, `COMMENT` (`#` to end of line) and `WS`. A format grammar imports them
and declares its own `%ignore`s.

## `pysidtracker grammardiff`

```bash
pysidtracker grammardiff A.lark B.lark [--start RULE ...]   # exit 1 on any difference
```

Both grammars are compiled as Lark compiles them (imports resolve against the
grammar's directory and pysidtracker's fragments) and normalised to
`{rule: {alternative}}` and `{terminal: pattern}`. An alternative is its symbol
names, an alias as `-> name`; Lark's EBNF helper rules (`x*`, `x+`) are inlined
by structure (`{@ x | x}`, `@` the recursion) and anonymous terminals by
pattern, so respellings of the same rules compare equal. Without `--start`
every rule a grammar defines is a start, so unreachable rules are compared too.
The report lists rules only in one grammar, the alternatives a shared rule
loses (`-`) or gains (`+`), and terminals only in one or with different
patterns. `normalise(text, path, start)`, `compare(a, b, names)` and
`grammar_diff(path_a, path_b, start)` are the Python API.
