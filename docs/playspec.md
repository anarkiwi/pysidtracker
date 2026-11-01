# playspec: playroutines as synchronous state machines

`pysidtracker.playspec` is a textual DSL that states a SID playroutine
losslessly as communicating extended finite state machines over memory-mapped
cells, plus its toolchain:

| Stage | API | CLI (`pysidtracker playspec ...`) |
|---|---|---|
| Parse (Lark, `playspec.lark`), derive variants | `parse(text, resolve)`, `load(path)`, `derive` | |
| Static checks + bind + lower to IR | `elaborate(spec, binding)` | `check SPEC [--binding JSON]` |
| Concrete execution (compiled Python, or [native](native-spec.md)) | `compile_program`, `run`, `SpecPlayer` | `python`, `run TUNE --binder mod:attr` |
| SMT transition relation, bounded checking | `smt.transition`, `smt.bmc` | `bmc SPEC --binding JSON --frames N` |
| Translation validation against 6502 code | `tv.validate`, `tv.replay` | `tv TUNE --binder ... --routine ... --pc ...` |
| Replace `native` hooks with spec statements | `unnative`, `equivalent` | `unnative SPEC TUNE --binder ... -o OUT` |
| Reschedule SID writes into a canonical order, emit and prove | `outorder.reorder`, `emit.emit`, `outproof.prove` | `reorder TUNES... [--order ORDER] [--prove MS] [--emit OUT]` |

pysidtracker ships the language and tooling only. Specs of particular players
belong to their format packages, which bind a spec to a build's layout and run it
with `SpecPlayer`. A build that differs from a spec in a few statements is a
[variant](#variants) that `extends` it, not a copy.

## Model

A play call is one synchronous instant. The `schedule` fixes the order in which
machine instances step within it. A machine sees memory as the instances before
it in the schedule left it: voice 0's stores are visible to voice 1 in the same
call. Signals are frame-local values with a default, used for instant
communication. A signal read by an instance other than its owner must follow the
owner in the schedule; the checker enforces this causality rule.

```
machine clock {
  signal row: bool = false;
  step { tempo -= 1; if neg(tempo) { tempo := tempo_reload; row := true; } }
}
machine voice[x in 0..3] {
  cell dur: u8 @ L.dur + x;                                   // private per instance
  control phase on note_ctl mask 0x40 { fresh = 0x00, held = 0x40 }
  step { if clock.row { dur -= 1; if neg(dur) { parse(); return; } } output(); }
  ...
}
schedule {
  clock;
  if clock.row { voice[0]; voice[1]; voice[2]; } else { for v in L.output_voices { voice[v]; } }
}
```

### Declarations

- `layout { names: addr|byte|flag|list; }` declares the binding a spec is
  elaborated against, written `L.name`: typically a format package's discovered
  layout of one player build.
- `cell n: T @ place [attrs];`:
  - `T` is `u8`, `u16`, `u8[n]` or `ptr[n]`.
  - `place` is an address expression, or `(lo, hi)` for split word tables.
  - Top-level cells are shared; cells inside a machine are private per instance.
  - Attributes:
    - `ro`: a song table; any store into it is an error.
    - `out`: SID registers.
    - `operand`: the cell lives in an instruction operand (self-modified code).
    - `scratch`: excluded from translation-validation comparison.
    - `latched`: a reader must not follow another instance's store in the same call.
    - `volatile`: an input register whose value comes from outside the spec (voice 3's
      `$D41B`/`$D41C`, a raster poll). Each read is a separate read of memory that is
      never folded, shared or reordered; a store to it is an error. The spec says
      nothing about *when* in the call the read happens (see "Volatile inputs").
    - `per_proc`: for a partly relocated player whose routines address the cell at different places. A proc `P` accesses the cell at its place with `L.<cell>` read as the binding's `<cell>@P` when that field exists (the rest of the place expression is unchanged), else at the cell's own place. Without such fields the program is unchanged.
    - `notes(lo|hi|word, K)`: the `ro` table holds the native note table; index `i` means standard note `i + K` (`K` a static, possibly negative). `lo`/`hi` mark a `u8[n]` as the low/high byte of the note word, `word` a `u16[n]`/`ptr[n]` as the whole word. Cells aliasing one table may declare different `K`.
- `machine m [x in a..b] { cell/signal/control/proc/step }` declares a machine
  and its instances; `x` is the static instance index.
- `control c [on cell mask M] { s = v, ... }` declares control states. With
  `on cell mask M` they are bits of a memory cell (`in c { s => {...} }`,
  `goto s`); without it they are a hidden persistent variable.
- `proc f(a: u8, s: stream) -> u8 { ... }` declares a procedure. Procedures are
  not recursive, `return` is allowed, and streams pass by reference.
- `schedule { ... }` lists the intra-frame order: instances, `if` on statics or
  signals, and `for` over a range or an `L.list`.
- `invariant name: expr;` and `assume expr;` state properties for bounded
  checking. `old(e)` reads frame-start memory.

### Statements and expressions

- **Statements:**
  - `:=` and compound assignment (`+= -= &= |= ^= <<= >>=`).
  - `let`.
  - `if`/`elif`/`else`.
  - `while c bound N`: running past the bound is a runtime error and a proof obligation.
  - Static `for`, `in`, `goto`, `assert`.
  - `native addr(a = e, x = e, y = e) resumes pc;` runs the tune's own 6502 code (a documented hook a rip patches) from `addr` with those registers (each optional). If the code reaches `pc` the spec carries on with the registers restored; if the call returns first, the CPU has finished it and the spec's call ends. It needs a `SpecPlayer` (`run` raises) and is opaque to `check` and `tv`, which refuse it.
  - `name: stmt` labels a statement and `{ ... }` groups statements without opening a scope; both are places a [variant](#variants) can `patch`.
  - `finish;` ends the whole play call: the rest of the schedule is skipped and the player returns to its caller.
  - `stream s = base @ cursor [grammar "mod:attr:rule" [if cond] {, "mod:attr:rule" [if cond]}]`: the first rule whose static `cond` (e.g. `L.extended`) holds, an arm without `if` always holding, so a layout can pick the rule per build; read with `let v = next s;`, `s[k]` (relative to the cursor) and `s.pos`.
- **Arithmetic:**
  - Exact 8/16-bit wraparound. Untyped literals take the other operand's width; u8 widens to u16 implicitly.
  - Narrowing is a width error (use `lo()`/`hi()`).
  - Comparisons are unsigned. Precedence is Python's: comparisons bind below bitwise operators.
  - `/` and `%` take static operands only.
- **Builtins:**
  - Bytes and words: `lo hi word u8 u16 sx zp`.
  - Flags and carries: `neg` (bit 7), `carry(a, b[, cin])`, `borrow(a, b[, cin])`.
  - Signed compares: `slt sle sgt sge`.
  - Bits and rotates: `bit rol ror`.
  - `ite`, `ind(zp, y)` (`(zp),Y` with zero-page pointer wrap and 8-bit index).
  - Statics: `len`, `member`.
  - Memory: `mem[a]`, `mem16[a]`.
  - Entry registers: `cpu.a/x/y/c/z/n/v/d/i`.

### Memory semantics

Every cell read and write goes to memory at its bound address. Aliasing in rips,
operands the player rewrites, and the power-on/post-init RAM all behave as on the
C64. An indexed cell `t[i]` is `mem[base + zext(i)]` with 16-bit wrap; stream
cursors and `zp()` wrap at 8 bits. Streams read live memory, so the grammar named
on a stream is a static view only: `SpecPlayer(..., validate=True)` parses each
opened stream with that `ByteGrammar` and reports malformed data in
`player.validator.failures()`.

### Static checks

`elaborate` raises one `SpecError` listing every diagnostic:
- undefined names (cells, layout fields, procs, states, machines);
- width errors;
- stores to `ro` cells;
- causality: a foreign signal read before its owner runs, a signal written by
  another machine, or a `latched` cell read after another instance's store;
- recursion, unreachable code, `old()` outside an invariant, and misplaced
  value-returning calls.

With a binding it also rejects private cells shared by two instances, and lists
writable cells aliasing read-only tables in `Program.notes`.

## Events and objects

A player is mostly counters that wrap, tables walked a row at a time and streams read a row at
a time. A spec declares those as **objects**, each stating what it does when an **event**
happens, instead of writing the cursor, counter and carry arithmetic in procs. Objects stay
typed through the IR, both executors, the simplifier and the solver
([In the toolchain](#objects-in-the-toolchain)).

### Events

`frame` is the play call. Under it are `tick`, `row`, and below `row` `note`, `legato` and
`instrument`. `event name: parent;` adds one below an existing event.

| Statement | Effect |
|---|---|
| `fire e;` | runs the reactions to `e` of every visible object and handler, in declaration order: the top-level ones, then those of the machine instance that fires |
| `fire e { ... }` | states the reactions by hand: the block runs and nothing else reacts |
| `react obj on e(operands) [-> flag];` | one reaction of `obj`, with its operands given; `flag` (a `bool` variable) receives whether it wrapped |
| `handler name on e, f { ... }` | statements that run, in declaration order among the objects, when `e` or `f` fires; the glue between objects |

A reaction names an event exactly: a firing of `note` reaches `on note`, not `on row`; a
reaction lists several events as `on note, legato`. The hierarchy bounds the cascade: a counter
that wraps fires an event strictly below every event it counts, so a firing always ends. Operand
expressions and handlers are lowered where they run, in the firing instance, so `x` is the index
of the voice that fires.

### counter

```
counter name: cell {
  on frame -= 1;          // or += k; the step k is a positive constant
  on note reset;          // loads the reload value
  reload e;               // the value a wrap and a reset load (default 0)
  wrap [at e] [-> event]; // up: wraps when the new value exceeds e; down: when it borrows
}
```

`cell` is a `u8` or `u16`. `+= k` stores `v + k` and wraps when that exceeds the limit (default:
the cell's maximum); `-= k` stores `v - k` and wraps when `v < k`. A wrap loads the reload value
instead and fires the `wrap` event.

The tempo countdown of the first example, `tempo -= 1; if neg(tempo) { tempo := reload; row :=
true; }`, is `examples/events/tempo.play`:

```
counter countdown: tempo { on frame -= 1; reload speed; wrap -> row; }
counter played: rows { on row += 1; }
machine clock { step { fire frame; } }
```

### walker

A table walker advances through a table of fixed-width rows. The first byte of a row (`c0`) picks
the arm by range; the arm reads the row's bytes `c0`, `c1`, ... as operands.

```
walker name {
  table @ base [, base ...] [width w];  // interleaved rows (one base) or columns (several)
  pos cell; count cell; values v0, v1;  // position, sweep counter, the values it exposes
  start e;                              // the position `reset` loads (default 0)
  on frame step; on note reset;
  row lo[..hi]: sweep(into=v, count=op, step=op [, frac=n]) [then v = op, ...];
  row lo[..hi]: delay(count=op) [then ...];
  row lo[..hi]: set(v = op, v.hi = op, v.lo = op) [then ...];
  row lo[..hi]: jump(op);
  row lo[..hi]: stop;
}
```

- **Layout.** `table @ b width w` has rows of `w` bytes at `b`, `b+1`, ...: `ci` is the byte at
  `pos + i`. `table @ b0, b1, b2` reads `ci` at `bi + pos` and advances by `width` (default 1).
  The bases are expressions evaluated at each step, so a table can move.
- **Operands.** `ci`, a constant, `& mask` and one shift: `c0 & 0x7f`, `c0 << 4`, `(c1 & 0x0f) >> 2`.
- **Step.** The row at `pos` is the first arm whose range holds `c0`, else a `stop`.
  - `sweep` ends when `cnt == count`: `cnt := 0`, `pos` advances by the width, the `then`
    assignments run. Otherwise `cnt += 1` and the signed `step` is added to the value `into`.
    A word value takes the carry; with `frac = n` the low `n` bits of the sum are held in the
    low byte of a word (a fixed-point value).
  - `delay` is a `sweep` that adds nothing; `set` assigns and advances.
  - `jump` loads `pos` and decodes the row there in the same step, and a jump to the current
    row holds. A walker that jumps 256 times in a step raises `SpecRuntimeError`.
- **Values.** `values` are cells the rest of the spec reads; `v.hi` and `v.lo` assign one byte
  of a word.

The pulse and filter tables of `examples/events/pulse-filter.play`:

```
walker pulse {
  table @ L.pulse_table width 3;
  pos pwpos; count pwcnt; values pw; start start_pulse;
  on frame step; on note reset;
  row 0..0x7f:    sweep(into=pw, count=c0, step=c1);
  row 0x80..0xfd: set(pw.hi = c0 & 0x7f, pw.lo = c1);
  row 0xfe:       jump(c1);
  row 0xff:       stop;
}
```

replaces a proc that decodes the row, compares the sweep counter, adds the sign-extended step to
the low byte, carries into the high byte by hand, follows jumps and writes the position back.
`tests/test_playspec_event_examples.py` checks it against that proc on random tables.

### cursor

```
cursor name @ base pos cell [grammar "module:attr:rule"] {
  on row next; on pattern_end reset;
  run lo..hi base b -> cell;                // a byte in lo..hi starts a run of (byte - b) empty rows
  field name [when cond] [-> cell = operand];
}
```

The cursor reads the stream `base` at the 8-bit position in `pos`, as `stream` does, and
`grammar` names the same `ByteGrammar` view. `next` reads one row: a pending run count makes it
empty (and counts down); else a first byte in the `run` range loads the count (and is the row);
else the row is a first field and each following field whose `when` holds of the previous field's
byte: `neg(prev)`, `bit(prev, n)`, `(prev & m) == v`, `!=`, `!`. An empty row reads its first
field as 0. Each field with `->` stores its byte, through the operand, in the cell; `pos`
advances over the bytes read. `reset` loads `pos` from `start` (default 0) and clears the run.

The orderlist and pattern of `examples/events/pattern-cursor.play` replace a proc of `stream`s,
`next`s and position write-backs:

```
cursor orderlist @ L.orderlist pos ordpos { on pattern_end next; field pattern -> curptn = pattern; }
cursor pattern @ word(ptn_lo[curptn], ptn_hi[curptn]) pos ptnpos {
  on row next; on pattern_end reset;
  run 0xF0..0xF7 base 0xF0 -> gap;
  field note -> curnote = note & 0x7F;
  field inst when neg(note) -> curinst = inst & 0x7F;
  field fx when neg(inst) -> curfx = fx;
}
handler last_row on row { if curnote == 0x7F { fire pattern_end; } }
```

### owner

`owner name: cell { id e; none e; on note claim [when e]; on legato release [when e]; }` is a cell
holding which instance owns a resource. `claim` stores `id` (default the machine instance index)
when `when` holds; `release` stores `none` (default `0xFF`) if the cell holds `id`. `name.mine`
tests that the cell holds this instance's `id`, `name.free` that it holds `none`; `owns(a, b)` is
the same test on two values.

### record

`record name @ base stride s { f: u8; g: u16 [@ offset]; }` in a machine gives each instance the
cells `f`, `g` at `base + x * s + offset` (fields follow one another unless `@` places them). It
replaces cells declared as `L.base + 7 * x + k` and is checked for overlaps and the stride.

### u16 cells

A `u16` cell is a word at adjacent bytes or at `(lo, hi)` addresses; reads and stores combine the
bytes, so `pw += sx(step)` carries into the high byte. A walker's `sweep` and `v.hi`/`v.lo` use the
same cells.

### Objects in the toolchain

- **Elaboration.** Parsing and `elaborate` check the declarations (events, cells, ranges, operand
  forms) and bind them to `Program.objects` (`ir.Obj`: kind, byte addresses of each cell, the
  static structure, the reactions) and `Program.events`. `fire` becomes `("fire", event, body)`,
  a transparent block, whose body holds one `("obj", name, event, flag, operands, reads, writes,
  dynamic)` statement per reaction; a wrap becomes an `if` on `flag` around the cascaded `fire`.
  `ir.py` states the statements.
- **Execution.** `pyexec` calls the reaction functions of `objects.py`, the oracle. The native
  core (`crates/specvm-core/src/objects.rs`) steps the same reactions in Rust: the encoder pushes the
  operands and emits an `Obj` instruction naming a flat description; the stores, device reads and
  failures behave as those of any other statement. [native-spec.md](native-spec.md) has the checks.
- **Simplifier.** `outorder.reorder` flattens `fire` and leaves every `obj` statement as it is,
  with its operands simplified: its `reads` and `writes` keep the cells it touches out of
  promotion and fixing, a table at a computed address (`dynamic`) keeps every cell in memory, and
  an object that writes the SID is refused. `emit` writes each object back as a declaration with
  `react` statements; the emitted spec elaborates to the same program (`isomorphic` compares the
  objects too).
- **Solver.** `smt` treats a reaction as an uninterpreted, deterministic function of its
  operands and the bytes it may read: every byte it writes and its wrap flag are one function
  application each, the same for every call. `outproof.prove` therefore proves the code around
  objects; what an object does is not examined, and `bmc` properties over bytes an object writes
  can fail on counterexamples the object's real behaviour excludes.
- **Limits.** Rows carry at most 16 bytes, and operands are the forms above. An object cannot write the SID (state those stores);
  a walker's step does not return the carry the 6502 leaves, and a cursor's position is the offset
  of the next byte, not of the last one read.

## Variants

`spec v extends base;` makes `v` a variant: it starts as `base` and states only what
differs. `parse` derives it, so `elaborate`, `smt`, `tv` and `SpecPlayer` see one flat
spec. Every edit must apply exactly, or `parse` raises a `SpecError` listing each one
that does not.

| Item | Effect | Refused when |
|---|---|---|
| a plain declaration (`layout` field, `const`, `cell`, `proc`, `machine`, `invariant`, `assume`) | added | the base already has that name (a layout field of the same kind is allowed) |
| `override const/cell/proc/machine/invariant/schedule ...` | replaces the base's, in its place | the base has none of that kind and name |
| `extend machine m { ... }` | merges members into `m`: plain members are added, `override cell/signal/control/proc/step` replaces one, `patch` and `drop` act on its members | `m` does not exist |
| `patch p { replace l {...} before l {...} after l {...} }` | edits proc `p` (top-level, or a member or `step` inside `extend machine`) at the statement labelled `l` | `p` has no statement labelled `l`, or more than one |
| `drop n, ...;` | removes the declarations (or members) named | none or several have that name |

A base marks the places its variants differ with labels: `name: stmt`, at any depth
in a proc. A group `{ ... }` is a statement sequence that opens no scope, so a
replacement can bind `let`s that later statements use, and a label on a group covers
several statements. `replace` keeps the label on the new statements, so a variant of a
variant can edit the same place again; `before`/`after` insert next to it.

```
// player.play: the orderlist's end byte restarts the voice
spec player;
...
machine voice[x in 0..3] {
  proc orderwalk() {
    ...
    order_end: if ord[0] == L.order_end {
      ord.pos := L.order_restart;
    }
    ...
  }
}

// player_stop.play: a build whose end byte stops the voice instead
spec player_stop extends player;
cell stopped: u8 @ L.stopped;
layout { stopped: addr; }
extend machine voice {
  patch orderwalk {
    replace order_end {
      if ord[0] == L.order_end {
        stopped := 1;
        finish;
      }
    }
  }
}
```

`load(path)` reads a base from `NAME.play` in the same directory; `parse(text,
resolve)` takes a mapping or function from base name to `Spec`. Diagnostics give each
line as `spec:line` (`line player_stop:9: ...`). `Spec.sources` holds every spec in the
lineage by name.

**Choosing a variant per build.** `SpecPlayer`'s `bind(view, entry)` may return
`(variant, binding)` instead of a binding, so the code-shape probes that discover a
build's layout also choose which variant runs it. A format registers `bind` with its
spec through [`SidFormat.spec`](formats.md#spec).

What goes where:
- **The layout:** what a build moves. Addresses, operands, table sizes, and static flags that pick between expressions.
- **A variant:** what a build does differently. Other statements, procs or schedule, rather than an `L.flag` guarding both code paths in one spec.
- **`native`:** code the spec has not stated yet. [`unnative`](#removing-native-hooks) states it as a variant.

## SID writes

`SpecPlayer.writes` lists the SID writes `(register, value)` the spec's stores made in the last frame, in order: every play call of the frame, every write (one of the value the register holds too). A `native` block's and the tune's own code's writes are not among them.

## Volatile inputs

A `volatile` cell is an input of the call. The k-th read of volatile cells in a call
returns the k-th input, whatever the cycle:
- `SpecPlayer` reads it through the bus as any other cell. With `inputs=f`, `f(address)`
  supplies the value instead.
- `EmuPlayer(record_reads=addresses)` records `(address, value)` for each read its
  6502 code makes of those addresses in the last call (`reads`). Feeding those values to
  a `SpecPlayer`'s `inputs` compares the two players given the same inputs, so
  agreement does not depend on the cycle either reads at.
- Translation validation treats each read as a fresh input. The obligation is that the
  spec and the 6502 code read volatile addresses the same number of times, at the same
  addresses in the same order; there are no cycle terms. Reads on a path that is not
  taken do not count.

## Execution

The lowered IR is one structured program per call:
- each proc is lowered once per instance, stream arguments and set of instances
  already run, and called (`pysidtracker playspec python` emits one function per
  proc, named after it; the symbolic backends execute the body at each call);
- returns become guard flags;
- static choices and addresses fold per instance.

`pyexec` compiles it to a single Python function. `SpecPlayer` subclasses
`EmuPlayer`:
- It boots the tune's own init on the jennings host (power-on RAM, PAL flag, `$01` banking).
- Every play call then runs the spec on that memory. Reads go through the CPU's banked bus; stores go to RAM, with `$00`/`$01` also reaching the bus.
- When the play address is not the modelled entry, the tune's 6502 play routine runs and each call into an entry that `entries(view)` reports is trapped and run as the spec. One binding is kept per entry and banking, which handles play wrappers and several player instances.
- `play_frame`/`render_grid` produce the same forward-filled grid as every other player.

## Standard notes

A tracker's native frequency table differs from the equal-tempered one. A cell
declared `notes(part, K)` says which note each of its indexes means, so a player can
answer every read of it with the standard word instead, e.g.
`cell flo: u8[96] @ L.freq_lo ro notes(lo, -24);` for a table whose index 24 is C0.

`pysidtracker.notes.frequency(note, clock_hz)` is the word: `round(440 * 2 ** ((note -
57) / 12) * 2**24 / clock_hz)`, 0 where that rounds to 0 or exceeds 16 bits. Note 0 is
C0, 24 is C2, 57 is A4.

`SpecPlayer(..., notes=clock_hz)` (or `elaborate(..., standard=clock_hz)`) lowers each
read of a declared cell to a lookup op that `pyexec` compiles to a precomputed table
index. Any index is answered, past the native table's end included, and an index below C0
answers its note's word too. Memory is never written, so the data overrun indexes reach
is intact. Without `notes` the declaration changes nothing: `elaborate`, `pyexec`,
`smt`, `tv` and `unnative` see native memory. `smt` refuses a program lowered with
`standard`. `native` blocks and the tune's own 6502 code still read the native table,
as do `mem[]` reads.

A variant's `override cell` restates the declaration, so a variant that omits `notes`
leaves that cell native.

## Removing native hooks

`unnative(spec, binding, image, name=...)` returns a [variant](#variants) of `spec`
(default name `<spec>_stated`) that overrides each proc holding a `native` statement with
the statement replaced. `pysidtracker playspec unnative SPEC TUNE --binder ... -o OUT`
writes it, named after `OUT`'s stem, after checking it against the original. Keep it
beside `SPEC` so `load` finds the base.
- **Lift.** The hook's 6502 runs symbolically (`tv.Machine6502`) from `addr` with the
  `native` arguments as inputs, until it reaches `pc` (the spec carries on) or returns
  (the call ends).
- **Statements.** Each exit's memory is stated as `mem[a] := v` stores, with a `finish;`
  on the returning exit and an `if` between exits. Everything the hook reads is read
  into `let`s first, so the stores cannot change what the hook saw. The text replaces
  the `native` statement in its proc's source, so the proc's comments survive. A hook
  inside statements a `patch` put into a base's proc is refused.
- **Check.** `equivalent` plays both specs on the tune and compares every frame's SID
  registers and every byte outside the stack page; the CLI does this (`--frames`).
- **Refused.** A hook that reads a register or flag its arguments do not give it, shifts
  by a variable, stores into its own code, loops without a bound, or is never reached by
  the schedule. Registers a hook leaves for its caller and stack-page bytes are not
  modelled, as the spec models neither.

## Output order

`outorder.reorder(program, order, memory=...)` rewrites a bound program so each call
writes only the SID writes an order keeps, in that order. An order lists slots: `name` is
a register's last write in the call, `voice(...)` repeats a list per voice and
`interleave(...)` runs it register by register. A voice's `sr`, `ad` and `ctrl` also
have per-segment slots `name:k`, the one of the `k`-th last gate segment, and
`segments(sr,ad,ctrl):N` lists `N` segments of the three, oldest first. The default is
`fc_lo,fc_hi,res_filt,mode_vol,voice(segments(sr,ad,ctrl):4,freq_lo,freq_hi,pw_lo,pw_hi)`:
per voice, `sr,ad,ctrl` once per gate segment, then the rest.

A gate segment ends at a ctrl write that changes the gate or test bit from the call's
previous ctrl write (the call's first ctrl write always ends one). Its group is the
`sr` and `ad` written since the previous segment ended, then that ctrl write; a register
not written since is left out. What follows the last segment is the final group: the
`sr` and `ad` written since, then the last ctrl write if one followed the last segment.
The oldest segment is flushed first; the segment slots of `name:k` hold the last `N - 1`
segments and `name` the final group. A voice whose order has no segments writes each
register's last value.

The SID envelope generator stores the AD and SR nibbles on a write and reads them at a
rate-counter match or a state change, so the order of a frame's ctrl, AD and SR writes
changes neither the envelope nor the generator state (an exhaustive emulator test over
every state, level, AD and SR value, gate and test change and ordering of the three
writes found no difference). A frame commands of a voice's envelope its gate edges and the
AD and SR in force at each edge and at the frame's end; the writes between do not matter.
The test bit does not reach the envelope but resets the oscillator, so it ends a
segment as the gate does.

`outorder.kept(writes, slots, native)` is the definition above on the writes of one call.
A voice whose ctrl a path may store as many times as the order has segments (a loop,
sample playback by test-bit toggling) cannot be bounded: its `sr`, `ad` and `ctrl` stay
in place in the program's order (`Reordered.native`), before every other flush.

The transform:
1. **Inline** the procs and **nest** each lowered `return` guard into the arm of the
   branch that does not return.
2. **Fix and unroll**: forward values (step 7) with each fixed-address read of a declared
   byte replaced by its byte in the bound `memory`, and each loop whose condition is then
   known unrolled, until no bounded store of the result reaches a byte so replaced (a
   relocated operand addressing the SID, a channel list walked from a table, become
   constants). A non-`ro` byte in an `observed` interval is never fixed.
3. **Promote** frame-local cells to variables: a constant-address cell no path reads
   before storing it in the call, outside the SID, volatile cells, `observed` intervals and
   bounded indexed accesses. Staging buffers written and copied out in the same call (a
   shadow register block, zero-page temporaries, self-modified operands used as scratch)
   become variables.
4. **Split** each bounded non-constant SID access (a voice offset held in a variable) into
   one constant-address access per register it may reach, and the access as before
   where it may reach none.
5. **Instrument** every SID store as an assignment of the register's output variable and
   written flag, and every SID read as the output variable where written. A store to a
   segmented voice's `sr` or `ad` assigns the final group's variable. A store to its
   `ctrl` whose gate or test bit differs from the last ctrl value (or the first) closes
   the segment: the closed groups shift back one, the final group's `sr` and `ad` become
   the newest closed group with that ctrl value, and the final group is cleared; otherwise
   it assigns the final group's ctrl. The last value stored and a stored flag serve reads.
   The `native` voices' stores are kept in place and set a flag.
6. **Place** each slot's flush, `if written: store(register, output)`, at the earliest
   statement boundary after which no store follows that assigns its variables (a ctrl
   store assigns every segment's) and the slots before it are flushed, inside an arm of
   a branch where that falls before the arm's end, and before each `finish`. The native
   stores come first: no slot flushes before the last one. Every path flushes every slot
   once, in order (`flush_order`).
7. **Forward** values: constants, copies and single-use expressions propagate to their
   uses under the conditions known to hold in each arm; at a join, a variable assigned in
   either arm becomes an `ite` on a snapshot of the condition.
8. **Eliminate** dead assignments and empty branches, and **declare**: each variable is
   zero-initialised at the head of its innermost enclosing block outside any loop body
   unless its first statement there assigns it.

An output variable left after step 8 is a buffer the order forces: its value depends on
state the call changes before the slot can be flushed, because a register earlier in the
order is stored later in the call, or the register is stored inside a loop the transform
cannot unroll (`Reordered.staged`). GoatTracker 2's AD and SR are the common case: the
new note's instrument writes them, a pattern row read later in the same call may change
the instrument before a hard restart rewrites them, and the AD slot must follow that
rewrite. `OrderError` refuses what cannot be rescheduled: `native` code and a store
through an unbounded pointer.

`emit.emit(program)` writes a reordered program as playspec source: one machine whose
step is the call, the cells it names declared at their bound addresses (owner-qualified
names, `ro`, `volatile` and `notes(...)` kept), SID stores as `sid[V0_AD]` with a
`const` per register, and each variable `let`-declared at its first assignment (original
names without their lowering suffix, `out_<register>`/`wrote_<register>` for an output
and its flag, `cond` for a condition snapshot). Elaborating the text gives the program
back up to variable names (`emit.isomorphic`), with the binding or without.

`SpecPlayer(..., transform=f)` runs `f(program, view)` on each bound program, so a
format's player plays the reordered spec. `outorder.compare(make, order, frames)` plays
a tune both ways, the reordered side as its emitted spec elaborated again, and compares
the register grid and each call's writes against the kept writes of the original's;
`pysidtracker playspec reorder TUNES... [--prove MS] [--emit OUT]` does it over tunes
(`.sid` files or a `.csv` with a `path` column) with the installed formats, proves each
call (`--prove`, a per-query timeout) and writes the emitted specs.
`outorder.census(result)` lists what a reordered program still uses, the [object reactions](#events-and-objects) by kind among it.

`outorder.check(original, result)` states the claim as one solver query: from any
memory and control state, the reordered call writes the original's kept writes in order
(both logged into variables, native stores included, which a path bounds by its loops) and leaves the same memory and controls outside the SID and
the promoted cells. It closes for small programs only; `outproof.prove` proves the same
claim compositionally.

### Proving a reordered call

Write `P0` for the lowered call and `Pk` for the program after step `k`; `R` for `P0`
instrumented (step 5) with every slot flushed at its end and before each `finish`, which
defines the kept writes in canonical order. The theorem: from any memory and control
state satisfying the premises below, `P8` writes exactly `R`'s logged writes in order
and ends with the same memory outside the SID and the promoted cells, and the same
controls. `R`'s writes are `kept` of `P0`'s: for each voice, in order, the gate or test
edges of `P0`'s ctrl writes (a segment per edge, the last `N - 1` kept, at most that
many by the native rule) with the `sr` and `ad` in force at each and at the call's end,
and every other register's last value; so `P8` and `P0` command the same sequence of
gate edges with the same AD and SR at each edge and at the call's end, and leave the
same final value in every other register. A voice with native stores writes them as
`P0` does, in order. Premises: the fixed bytes hold their bound values, and no read through an
unbounded pointer reads a promoted cell or the SID.

The passes compose by transitivity; each is justified as follows.

- **Inline, nest, split, instrument, declare, hook removal and `mk` normalisation** are
  equivalences by the IR semantics. Procs share their variables, so a call is its body.
  `nest` moves `if f {S}` into the arm of a preceding branch after which `f` is unknown,
  out of the arm that assigns `f` a constant making it false (its last assignment on
  every path of that arm). `split` replaces a store at `a` by
  `if a == c1 {store c1} elif ... else {store a}`. Instrumentation assigns each register's
  last value and flag; `R` writes them, which is the definition of the kept writes.
  `declare` assigns 0 before every occurrence of the variable, where its value is the
  call's initial 0.
- **Promotion** holds by the definite-store analysis `frame_local` computes: every read of
  a promoted cell follows a store to it on every path, so the variable holds what memory
  would, and the cell is dead at the call's end, so its final memory is not part of the
  claim. Bounded indexed accesses that may reach a cell exclude it; the pointer premise
  covers unbounded reads.
- **Placement** holds by two structural checks `prove` repeats on `P6`. `flush_order`:
  every path flushes every slot exactly once, in canonical order (paths through both
  arms of every branch, a superset of the feasible ones). `flushed_last`: no statement
  after a slot's flush assigns its output or flag on any path that reaches the end
  without `finish`. A flush therefore writes the values `R` writes, in `R`'s order, and
  flushes write only the SID.
- **Forwarding** (steps 2 and 7) holds by local obligations the pass records, each one
  solver query over a fresh state (free memory, free variables) at one program point.
  The pass keeps a set of facts at every point: `v == e` for a forwarded variable, and
  `c == true/false` for a condition known in an arm. The obligations:
  - `step`: assuming the facts before a statement, the original statement and its
    rewritten form leave equal memory and variables, and the facts after it hold.
  - `cond`: assuming the facts, a branch or loop condition equals its rewritten form, or
    the constant that drops the branch or unrolls the loop.
  - `entry`: the facts and the condition's truth imply the arm's first facts.
  - `join`: an arm's last facts, with the condition snapshot's truth in that arm, imply
    the facts after the branch. A loop's facts after the loop are its invariant: they
    hold on entry and after the body.

  By induction over the statements (Hoare-style sequencing, the conditional rule and the
  invariant rule) the facts hold at every point of every execution from the empty
  facts at the start, so each rewritten statement computes what the original did. The
  premises of every query also include the fixed bytes (step 2, which stores never
  reach, so they hold everywhere) and each variable's value range from `ranges`, which
  each `step` obligation re-proves for the variable it assigns (with 0, the initial
  value, in every range).
- **Elimination** removes an assignment only where its variable is dead (backward
  liveness, with controls live at the end) and an `if` only where both arms are empty and
  its condition reads no volatile input; neither changes memory, controls or the writes.

`prove` recomputes the passes, checks that they reproduce the result, runs the
structural checks and discharges every obligation, reporting what stays open (a refuted
or unknown obligation, with its kind and program point, or those left when its time
budget ran out), the time and peak memory. A fact carried unchanged past a statement
that assigns none of its names and, if it reads memory, stores nothing, needs no query
(the frame rule); a query's premises are the facts its names reach and the known
conditions. Each query is small, so a whole call needs no more memory than its program.

Each query runs on a fresh solver through a simplifying tactic: a solver kept across
queries slows with every query it answers. The z3 term of a pure expression is built
once per memory and the values of the variables it reads, and the free symbols, ranges
and fixed bytes once per pass. Obligations are independent, so `--proof-jobs N` (or
`prove(jobs=N)`) discharges contiguous blocks of them in `N` worker processes, each with
its own z3 context; `tools/proofprofile.py` splits a proof's time between term
construction and `check`, and `tools/proofbench.py` times whole tunes.

## Proof tooling

- **Transition relation (`smt.transition`).** The IR executes over z3: memory is
  an array of 8-bit values indexed by 16-bit addresses. Execution is guarded,
  so each store and assignment is an if-then-else on its path condition and one
  call is one term, with no path enumeration. Loops unroll to their bound.
- **Bounded checking (`smt.bmc`).** It checks invariants, asserts and loop bounds
  over N calls. Initial memory is either fully symbolic (constrained by
  `assume`) or concrete post-init memory with chosen bytes left symbolic
  (`smt.concrete_memory(data, symbolic=...)`). It returns the failing frame and
  property plus the cells' initial bytes from the counterexample.
- **Translation validation (`tv.validate`).**
  - **Inputs:** a spec routine (e.g. `voice[0].parse`) and the 6502 code at `pc`.
  - **The 6502 side** executes deity-informant's p-code lift (the semantics jennings runs) over the same z3 memory. Paths fork on branches and merge at equal `(pc, call stack)`. JSR/RTS are tracked as calls.
  - **Constants:** lifted code bytes are constrained to the image. Bytes of the spec's `operand` cells stay symbolic and are read live from memory when the instruction executes.
  - **Stretches:** `stop=ADDR` (`tv ... --stop ADDR`) runs the code only to that address, for a spec routine that states the first part of a longer one; every path must reach it, and the comparison and `preserve` check happen there.
  - **The claim:** no byte either side wrote, outside the stack page and `scratch` cells, can differ afterwards; `preserve` registers survive; and the spec's own obligations hold.
  - **Assumption:** no data read aliases a return address the code itself pushed. The spec does not model the stack.
  - **Refusal:** a constant store into lifted code is refused.
  - **Counterexample:** z3 rewrites reads through stores before solving. A counterexample carries the differing address and both bytes, the initial registers, every cell's initial bytes and the full 64 KiB model. `tv.replay` re-runs it concretely on jennings and the compiled spec to confirm it.

## Coverage matrix

Every mechanism the Music Assembler (pymusicassembler), defMON (pydefmon) and
GoatTracker 1/2 (pygoattracker) players use, and the construct that expresses it.
The tests exercise each construct with small generic specs on synthetic memory.

| Mechanism | Players | Construct |
|---|---|---|
| Play-call divider / tempo reload from an operand | MA | `clock` machine, `operand` cell, `row` signal |
| Per-row tick counter, tempo and funktempo table | GT2, GT1 | private tick/tempo cells, `ro` table, toggle |
| Multi-speed (several calls per frame) | GT2, defMON | schedule `for` over `L.calls`; host cadence (`playroutine_cadence`) per call |
| Every-NMI vs main-tick work | defMON | `clock` signal gating work in the schedule |
| Pattern row streams, packed rests, durations, command bytes | all | `stream` + `next`/`s[k]`, live memory, `grammar` view |
| Orderlists: repeat, transpose, loop/restart, jump, tempo commands | all | stream over the orderlist, bounded `while` for command chains |
| Arranger with counted/forever jumps, one shared loop counter | defMON | shared cells, guarded row cursor |
| ALT step end for all voices, acted on next tick | defMON | memory flag written by voices, read by an earlier machine next call; `latched` |
| Wavetable programs (delay, wave, command, jump; relative/absolute note) | GT2, GT1, MA | 1-based `ro` tables, pointer cells |
| Pulse tables / sweeps (12-bit, signed speed, clamp, direction flip) | GT2, GT1, defMON, MA | `u16` split cells, `sx`, `carry`/`borrow`, masks |
| Filter tables / cutoff sweeps shared per chip, run before channels | GT2, GT1, MA, defMON | machine ordered first in `schedule`, shared cells |
| sidTAB cascade with jump rows and bitmap-selected columns | defMON | per voice-layer instances, stream whose bitmap bytes select the following values |
| Arpeggio phase counters | GT1, MA | private counters, static `ite` |
| Portamento, toneporta (snap), vibrato ramp | GT2, GT1, MA | 16-bit cell arithmetic, compares |
| Detune without carry, slide accumulator | defMON | 8-bit add into the low byte only |
| Hard restart, gate timer look-ahead fetch, legato | GT2, GT1 | conditional SID stores from `operand` ADSR |
| Ghost registers (output one call late) | GT2 | cells written after the SID store |
| Gate on/off, trigger latch | all | memory-mapped `control` states (`in`/`goto`) |
| Transpose | all | cell arithmetic on note indices |
| Self-modified operands (immediates, abs,X bases) | MA, GT1, GT2 | `operand` cells; `mem[op + x]` |
| Self-modified opcode (ADC/SBC choice) | defMON | `operand` cell holding the opcode byte |
| Zero-page pointers and temporaries, `(zp),Y` | all | zero-page cells, `ind`, stream bases |
| Multiple SIDs / stereo voices | GT2, defMON | instances routed to SID bases by static index |
| Linked-list channel walk (variable count) | GT1 | runtime index into array cells, bounded `while` |
| Carry flag entering the play call | GT1 | `cpu.c` entry register |
| Several player instances in one image | MA, GT2 | `SpecPlayer` binding per trapped entry |
| Play wrappers / driver code calling the player | MA | `SpecPlayer` runs the 6502 play routine, trapping entries |
| `$01` banking (ROM-banked tables, banking by the play code) | MA | reads through the CPU's banked bus; stores to `$00`/`$01` reach the bus |
| Relocation (any base) | all | addresses bound from the layout |
| Uninitialised / power-on RAM, post-init state | all | spec runs on `EmuPlayer`'s post-init power-on RAM |
| SID register mirrors | MA | binding folds the mirror (layout) |

Not expressed, by design or not yet implemented:
- **Timing inside a call.** A call is an instant: a read of a `volatile` cell is
  an input whose value the spec does not tie to a cycle (see "Volatile inputs").
- **Timing of a trapped call under interrupts.** A tune with play address 0 runs
  on an `InterruptPlayer` (its interrupts live, `render_grid` framed at the play
  source's raises) and `SpecPlayer` traps each call into an `entries(view)` entry
  there and runs it as the spec. The call is an instant at the cycle the handler
  reaches it, so the cycles the 6502 would spend in it are not spent, and a
  handler that reaches an entry by a vector or `JMP` rather than `JSR` is
  returned from as an `RTS`. Pass `sid=` (the PSID/RSID) for the header's SID
  model and extra SIDs; without it a header is built around the image. A tune
  whose interrupts are taken 3 times (the raises `play_source` needs to name a
  play source) with no call reaching an entry plays only its own code, and
  `SpecPlayer` raises `SidParseError` (from `play_frame` or `render_grid`).
- **Opcodes.** Translation validation keeps opcodes constant: a routine that
  rewrites an opcode it then executes is refused, not modelled. The DSL still
  expresses such a choice as an `operand` cell.
