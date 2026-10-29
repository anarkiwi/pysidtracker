# Program-table automata

Instrument, wave, pulse, filter and arpeggio tables are rows a player fetches
one after another: each row does something (its *label*: the columns it writes,
how long it holds) and names at most one successor row (`succ`, `None` for a
stop). Such a table is a functional graph, a deterministic automaton over a
one-letter alphabet. `pysidtracker.automaton` gives every format the same three
operations on it; nodes are any hashable row ids.

| call | result |
|------|--------|
| `minimise(labels, succ)` | `{node: class}`: rows no player can tell apart share a class |
| `number_chains(entries, succ, start=0)` | first-use numbers, each entry's chain consecutive |
| `layout_chains(succ, size, first=0, error=SidFormatError)` | `(slots, jumps)` placing rows in a `size`-slot table |

## `minimise`

Two rows are equivalent iff the label sequences the player reads from them,
following `succ`, are equal (a stop ends the sequence); this is the coarsest
partition stable under `labels` and `succ`, the one Moore refinement reaches.
Because each row has one successor it is computed in linear time instead of by
refinement rounds:

- every node is reached on a walk that ends at a stop, a classified node, or a
  new cycle;
- a cycle is classified by its labels' primitive period (KMP failure function)
  and least rotation (Booth's algorithm): its node at offset `i` from that
  rotation gets the class of `(rotation, i mod period)`, so equal cycles of any
  length or phase share classes;
- every other node hash-conses `(label, class of succ)`, and cycle nodes are
  entered under the same key, so a tail can merge with a cycle node.

Class ids are numbered by first appearance in `labels` order. A node missing
from `succ` is a stop; every successor must be a key of `labels`.

## `number_chains`

Numbers rows from `start` in first-use order: each entry (e.g. the rows the
patterns trigger, in play order) is followed along `succ` and numbered
consecutively until it meets a numbered row or a stop. `None` entries are
skipped. Unreached rows get no number, so the result is also the reachable set.

## `layout_chains`

Places rows so the player's "next slot" fetch reproduces `succ`. Rows nobody
jumps into (sources) are placed first, in `succ`'s order, then the rest. Each
chain fills consecutive slots; a chain that runs into a row already placed is
closed by a jump slot after its last row, `jumps[slot] = target`. Slots start
at `first`; if that layout does not fit `size` slots it is retried from `0`,
then `error` is raised. Every successor is then either the next slot or the
target of the jump slot after it.

```python
from pysidtracker.automaton import layout_chains, minimise, number_chains

labels = {0: "saw", 1: "pulse", 2: "saw", 3: "pulse"}
succ = {0: 1, 1: 2, 2: 3, 3: 0}               # saw, pulse, saw, ... forever
classes = minimise(labels, succ)              # {0: 0, 1: 1, 2: 0, 3: 1}
rows = number_chains([0], {0: 1, 1: 0})       # {0: 0, 1: 1}
slots, jumps = layout_chains({0: 1, 1: 0}, 256, first=1)
# slots {0: 1, 1: 2}, jumps {3: 0}
```
