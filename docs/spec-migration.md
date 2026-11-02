# Migrating a playspec to the abstractions

A transliterated playspec states what a player's 6502 code does, routine by routine. These rules
rewrite it into the playspec's own constructs, one pass at a time, each pass checked against the
spec before it. They are written to be applied by a tool: each has what to detect, what to write
and what makes it safe. The SID-Wizard spec (`pysidwizard/specs/sidwizard.play`) was migrated by
them; the other format specs are next.

## The gate

`tools/spec_trace.py record REF SPEC TUNES.csv` plays the starting spec over a corpus and keeps
each call's SID writes and registers; `check REF SPEC2 TUNES.csv` plays a rewritten spec and
compares what is audible, per call:

- the 25 registers, with the bits the chip ignores masked (the high nibble of each pulse-width
  high byte, the high five bits of the cutoff's low byte);
- per voice, whether the gate was cleared and set again within the call (the attack restarts)
  and whether TEST was set (the oscillator resets).

The order and grouping of writes within a call are not compared: by vagrant-moses
`docs/sidorder.md` they change nothing audible but those two edges. A pass is kept when no tune
differs and no refusal changes. Record once from the starting spec; never move the reference.

## Rules

| # | Detect | Rewrite | Safe when |
|---|---|---|---|
| 1 | cells of one table declared under several names (one per reading instruction) | one cell | the binder gives the names one address in every tune; a name bound to 0 is code the build lacks |
| 2 | per-routine addresses of one cell (`per_proc`, `<cell>@<proc>`) | one address per cell; a build that disagrees is refused by the binder | the gate (the corpus needed none) |
| 3 | a pointer built from lo/hi tables plus an offset, then read through (`zp := word(lo[i], hi[i]) + reloc; mem[zp + y]`) | a split pointer table `cell p: u16[n] @ (lo, hi)`, `view r via p offset L.reloc` read as `r[i].f`, `table t via p` read as `t[i][y].b`; relocation a binder constant | every read names the index chosen; where reads follow no selection (a note without an instrument), keep the last chosen index as state |
| 4 | a loop walking a table by an index cell (`y = pos; ... mem[base + y] ...; pos = y`) | a `cursor` over the table; the handler reads the row at the cursor with `cursor.c0..cN` | the cursor's step and the loop's advance agree, jumps included |
| 5 | a cell only ever written | delete it and its writes | nothing reads it (the CPU's scratch: temporaries, dispatch targets) |
| 6 | a proc with one caller, or one line, or that only calls on (an entry trampoline) | inline it | its locals do not clash with the caller's; it has no early return |
| 7 | a chain of `if v == a ... elif v == b` on one value | `match v { a => ...; b => ...; }`; a body that is only `if L.flag {}` becomes an arm guard | an arm guard only where no later arm or default would catch the value |
| 8 | lo/hi byte cells with carries between them | a `u16` cell; a register of a narrower width as `u9..u15`, packed (`u11 @ (lo : 3, hi)`) | a player variable keeps the width of its storage where its overflow reaches elsewhere |
| 9 | per-instance cells at `L.name + x * stride` | a `record` at the stride, fields placed by `@ L.name` | the fields' addresses follow the binding |
| 10 | one instance's code writing another's cells by address | `machine[i].cell := ...` | |
| 11 | a CPU flag passed between routines as parameters and returns | one per-instance signal the routines set and read | the routines' order is the player's |
| 12 | stores of one SID register's bytes, in either order or one of them | one assignment to the register's cell at its width | the gate; only the gate and TEST edges are order-sensitive |

| 13 | a spec covering several builds through `flag` layout fields | `tools/spec_specialise.py SPEC TUNE` fixes the flags the binder gives the tune, keeps the branches the build takes (`if`/`elif`, `ite`, arm and `then` guards, `&&`/`||`) and drops what nothing names any more (procs, cells, events, layout fields) | gated on the tunes of that build |

| 14 | a cell that holds the player's own working state, its address meaningful only to the 6502 (a zero-page pointer kept as an index) | `state name: type = initial;`, at no address | the gate |
| 15 | handlers and objects reacting to one event | declare them in the order the player does the work: reactions run in declaration order | the gate |

## Detecting the candidates

- Rules 1 and 3: bind every corpus tune and compare the addresses the binder gives the names of
  one table (`L.p_<table><n>`); identical or 0 everywhere is one table.
- Rule 5: a cell whose name never appears outside assignments to it.
- Rule 6: call counts of each proc; a body of one statement or one call.
- Rule 7: an `if`/`elif` chain whose conditions are all `v == constant` on one `v`.
- Rule 8: a write of `x` followed by `carry(x, ...)` feeding a write of its neighbour.
- Rules 9 and 10: `+ x * k` in a place, `mem[L.f + v * k]` in a body.
- Rule 11: a `bool` parameter named for the carry and a `-> bool` return passed on.

## What the rules do not reach

- Routine shape: the rules keep the player's control flow. Writing the spec from the format
  (data model, counters, events, decoders) and gating it against the transliterated spec is
  the step after them.
