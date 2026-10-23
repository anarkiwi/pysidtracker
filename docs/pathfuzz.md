# pathfuzz: inputs for the paths a corpus never takes

`pathdiff` localises faults from the paths the corpus runs. `pathfuzz` makes the
inputs it lacks: it lists the branch outcomes a tune never reaches on each side,
mutates the tune's data bytes until they are reached, checks every kept mutant against
the other side, and shrinks a divergent input to a small reproducer.

```
python -m pysidtracker.pathfuzz gaps     tune.sid --right SPEC [--song 1] [--frames 500] [--out gaps.json]
python -m pysidtracker.pathfuzz exercise tune.sid --right SPEC [--side 6502|python] [--budget 200] [--seed 0] [--out mutants.json]
python -m pysidtracker.pathfuzz minimise tune.sid --right SPEC [--patch mutants.json#k] [--register 0x04] [--neutral 0] [--out min.json]
```

`SPEC` is `module:fn` or `path.py:fn` (`survey.load_render`); `fn(data, song)`
returns a `MemPlayer` or a `paths.PySubject`. The left side is always the tune on
jennings (`paths.cpu_paths`).

## Patches

A patch is a JSON list of `[file_offset, byte]` pairs over the `.sid` file bytes,
shared with `pathdiff`. `apply_patch(data, patch)`, `make_patch(original, mutated)`
and `addr_to_offset(data, addr)` (the file offset of a loaded C64 address, honouring
a load address embedded in the body; None outside the image) convert.

## Coverage

```python
gaps(traces, universe)        # universe keys no trace covered
distil({tune: traces, ...})   # minimum tunes covering every covered feature (AFL cmin)
```

`traces` is a `PathTrace`, a sequence of them, or feature names; the universes are
`paths.cpu_universe` / `paths.py_universe`. `distil` is an exact set cover
(`faultloc.min_cover`). `py_arcs(roots, sources)` keys the Python universe alike and
maps each arc to its from-line.

## Targets

`baseline(data, song, make_right, frames)` traces both sides over `frames + LEAD`
and decodes both universes. `Baseline.targets(side)` are the unreached outcomes of
branches that ran: a 6502 conditional branch (`rel`) whose other edge was taken, or a
Python branch line with an unreached arc. Each target records the frames running the
branch, the frames taking each reached outcome and its first frame.

## Candidate bytes

Only data bytes are mutated: loaded from the file, never executed in `frames`
(`variants`) and not statically decoded as code (`paths`' control flow).

- **6502**: `branch_slice` takes the target frame's `whydiff.cpu_history` and
  backward-slices the branch's flags (`cpu_slice` over the steps' jennings `flow`, from
  `"PC"`) over every execution in that frame: the bytes the decision read at entry,
  and the instruction that set the flags.
- **Python**: a `MemPlayer`'s `PathTrace.reads` (its `_rd` addresses). A player that
  reads decoded structures instead of memory has none; the 6502's reads of the same
  frames stand in.

`rank_reads` orders addresses read up to the target frame: in the slice first, then
read differently by frames that took different reached outcomes, then read in the
target frame, then read more in frames running the branch than in the others. The top
`width` addresses are tried with `values(byte, ops)`: the flag setter's boundaries
(`CMP/CPX/CPY #i`: i-1, i, i+1; `AND/ORA/EOR #m`: m's bits; `BIT`: N and V), bit
flips, ±1, 0 and $FF; then every other byte value in random order.

## Exercise

```python
mutants = exercise(data, song, make_right, side="6502", budget=200, frames=500)
```

Targets take turns (earliest frame first), each trying its next single-byte
mutation of the original; a try runs both sides only up to the target's frame +
`margin` + `LEAD`. A try reaching any feature no earlier input reached, on either side,
is kept (an AFL-style queue) as a `Mutant`:

| field | meaning |
|---|---|
| `patch` | the mutation |
| `target`, `side`, `hit` | what it aimed at, and whether it reached it |
| `frames` | frames run |
| `new`, `labels` | per side the features it reached first |
| `verdict` | `gridcmp.verdict_lead` of the right grid against jennings |
| `diverges` | the right side mismatches where the original tune did not (another first frame or register), or fails on it |
| `error` | the right side's exception |

`--out` writes `{tune, song, side, frames, baseline, gain, mutants}`; `gain` sums the
new features per side.

## Minimise

```python
pred = same_onset_predicate(make_right, register, song)  # first departure at register
patch, frames = minimise(data, song, pred, patch=mutant.patch)   # (i) a mutant
patch, frames = minimise(data, song, pred, addrs=None)           # (ii) an original tune
```

Both modes are ddmin (Zeller) over changes from a baseline, memoised by subset:

- **(i)** the 1-minimal subset of a patch's entries keeping the predicate;
- **(ii)** the 1-minimal set of data bytes (`addrs`, default those the 6502 read in
  `frames`) to keep while every other one becomes `neutral` (default 0). The returned
  patch writes the neutralised bytes.

The frame count is bisected to the least `n` keeping the predicate, before ddmin (on
the starting input) and after (on the reduced one). Bisection needs a predicate
monotone in frames: one true at `n` is true for every later `n`, as "the fault
reproduces within `n` frames" is. The returned `frames` is then the onset frame + 1.
`same_onset_predicate` compares `oracle.register_grid` with the right side's grid
under `verdict_lead`; a right side that raises counts as not reproducing.

## Limits

- Mutations are single bytes of the original tune; a condition needing two bytes
  changed together is reached only if one of them already sits at its value.
- The slice is of one play call: a branch on state an earlier frame derived from the
  data (a counter, a copy into RAM) ranks the earlier reads, not the source byte.
- 6502 universe edges found only dynamically (indirect jumps) shift `#k` ranks, as
  `paths.cpu_universe` documents.
