# whydiff: why a render diverges

`gridcmp` says *which* register cell of a `MemPlayer` render differs from the
reference. `whydiff` says *why*: it traces the player's Python execution of that
frame and the tune's own 6502 play call on jennings, and puts them side by side.

```
python -m pysidtracker.whydiff tune.sid --format sidwizard --state     # an installed format
python -m pysidtracker.whydiff tune.sid mypkg.player:make --frames 250 [--reference grid.csv.zst]
```

`--format NAME` (or `auto`) plays the installed format's player on `--song`
(`SidFormat.play`); otherwise `make(tune_bytes, song) -> MemPlayer`. The reference is a `gridcmp` grid file (e.g.
a cached sidtrace grid), or the jennings grid. `--frame`/`--register` explain a
chosen cell; `--json` saves the machine-readable form. In tests, use
`assert_matches` / `assert_oracle_match` (see [oracle-testing.md](oracle-testing.md)).
From Python:

```python
why = explain_divergence(data, lambda d: MyPlayer(d), reference)  # None on agreement
print(why.text())
```

## Two Python players

The sides are pluggable: `python_side` traces any player with `play_frame` (a
`MemPlayer`, or a generic player whose registers live in a list), `cpu_side` a jennings
play call, and `explain_cell(left, right, grids, frame=, register=)` puts any two
side by side. To explain a generic player against a native transcription:

```python
generic = python_side(lambda: SongPlayer(song), tick, reg, label="generic",
                      cell=lambda p, r: (p.regs, r),          # where register r is stored
                      sources={name: source})                  # exec'd code: its text
native = python_side(lambda: SWMPlayer(swm), tick, reg, label="native")
print(explain_cell(generic, native, (ours, theirs), frame=tick, register=reg).text())
```

`cell(player, register)` names the `(container, index)` the register's value is stored
in (default the `MemPlayer` SID cell); the frame's last store to it is the slice's root.
A root that is a pseudo-filename in `sources` (code compiled from a string) is traced
with that text as its source. Given its own `grid`, a side that does not store the
register on the diverging frame is traced on `last_change`, the last frame that
changed it, and the header says so. `explain_divergence` takes the same
`label`/`cell`/`roots`/`sources`, so an exec'd player can face the 6502 directly.

## State: where it first parted

An output can diverge long after the state behind it did. `--state`
(`explain_divergence(state=True)`, always in `assert_matches`) takes the state the
left slice read from earlier frames and finds where it first stopped matching the
other side:

- Each such value is replayed frame by frame (a Python path from the player, e.g.
  `player.voices[2].ptn_gate`) beside the other side's frame state: the RAM cells its
  play call read before writing them, or the other player's paths.
- A list member's 6502 cell is fitted as a family (`fit_family`): the base and stride
  under which all of the list's members agree on the most frames, so a cell that never
  changes is still found through its siblings.
- Otherwise its partner starts equal if one does, then tracks longest (`first_break`,
  exact, or one-to-one when the representations differ), then shares its name, agrees
  most and is in the other side's slice.
- The earliest parting is explained like the output was, rooted at both sides' stores
  to that state, and chased again from there.

```
STATE: the state read from earlier frames, tracked over frames 0-263
  player.voices[2].wf_pos ~ $1048: tracks
  player.voices[2].ptn_gate ~ $1034: parts at frame 2
STATE ONSET ====
divergence: tune, reference frame 2, state player.voices[2].ptn_gate ~ $1034
  python $FE, 6502 $00
```

## The report

For a synthetic tune whose Python transcription drops a carry:

```
divergence: toy.sid, reference frame 2, register $00 v1_freq_lo
  python $10, reference $11, jennings $11
  python frame 2 (lead 0); 6502 play call 2 (lead 0), which agrees with the reference to here
  ...grid window around the cell...

WHY: the lines and instructions the diverging value came from
PYTHON                                          | 6502
      36  carry = self.cnt >= self.threshold    |    $1006 AD 19 10 LDA $1019    A=10 ... r$1019=10
          | self.cnt=2, self.threshold=3        |    $1009 48       PHA          ... w$01FD=10
          := carry=False                        |    $100A AD 1A 10 LDA $101a    A=02 ... r$101A=02
      37  self._wr(0xD400, 0x10 + carry)        |    $100D C9 02    CMP #$02     ..-B..ZC
>>   ** SID $00 v1_freq_lo = $10                |    $100F 68       PLA          A=10 ... r$01FD=10
                                                |    $1010 69 00    ADC #$00     A=11
                                                | >> $1012 8D 00 D4 STA $d400    w$D400=11

state read from earlier frames
   self.cnt = 2                                 |    SP = entry
   self.threshold = 3                           |    $1019 = $10
                                                |    $101A = $02

FULL HISTORIES, meeting at the diverging write
...
```

- **Header**: the cell, the Python, reference and jennings values, and the frame
  each side played. If the jennings grid does not match the reference up to the
  cell, the header says so (the 6502 history is then not the reference's own run).
- **WHY**: the backward dynamic slice of the diverging write on each side. These are
  the lines and instructions its value was computed from, oldest first.
- **State read from earlier frames**: the inputs of each slice that were set before
  this frame. Compare these first: a wrong value here means an earlier frame
  diverged in state without diverging in output.
- **Full histories**: every traced Python event and every 6502 instruction of the
  frame. The two columns meet at the diverging write, and `>>` marks the slice.

## Python side

`PyTrace` is installed with `sys.settrace` only while the one frame plays (earlier
frames play untraced, and nothing is hooked otherwise). It traces only the
player's own package (`player_roots`). For each executed statement it records:

- the values of the names, attributes and subscripts it reads, before it runs;
- the outcome of an `if`/`while`/`assert` condition;
- the values it assigned, after it ran;
- calls with their arguments and returns with their values;
- each `MemPlayer._wr` to a SID register.

Only side-effect-free expressions are evaluated: chains with calls, lambdas or
comprehensions are not.

An attribute named at run time is the same storage as the static one:
`setattr(o, name, v)`, `delattr(o, name)`, `vars(o)[name] = v`, `o.__dict__[name] = v`
and `del o.attr` store `o.<name>`, with `name` evaluated in the frame after the
statement, and `getattr(o, name)` reads it.

Each read is linked to the event that last stored to the same storage: a local of
that call, a global, an attribute of that exact object, or that item (or each item of
a slice) of that container, falling back to the container's own store when that store
built it (an item of a container merely bound in the frame is earlier state). Calls link to
their call site, returns to the caller's line, and statements to the branch that
guards them; a line reached through a call site counts only the returns of calls it
made before that call.
The slice follows only scalar values. A container mutated by a method call
(`list.append`) is not seen as a store.

## 6502 side

`cpu_history` boots the tune as `register_grid` does and steps the target play
call. Each `CpuStep` records the disassembly as the code stood, the registers and
flags after it, the bytes it loaded and stored (`observe_data_access`), and
`flow`: jennings' `dataflow.flow`, each location it set mapped to the registers,
flags and bytes its value came from. The flow is read off the pcode jennings
executes, so it is exact for every legal and illegal opcode. `cpu_slice` walks
these backwards from the SID store.

## Limits

- The 6502 history is the flat jennings PSID host, not sidplayfp. For tunes whose
  jennings grid differs from sidtrace, the header reports where they part.
- Code that runs inside a `sys.settrace` callback is invisible to coverage's
  tracer, so the tracer's tests drive its callbacks directly.
