# playspec: playroutines as synchronous state machines

`pysidtracker.playspec` is a textual DSL that states a SID playroutine
losslessly as communicating extended finite state machines over memory-mapped
cells, plus its toolchain:

| Stage | API | CLI (`pysidtracker playspec ...`) |
|---|---|---|
| Parse (Lark, `playspec.lark`), derive variants | `parse(text, resolve)`, `load(path)`, `derive` | |
| Static checks + bind + lower to IR | `elaborate(spec, binding)` | `check SPEC [--binding JSON]` |
| Concrete execution (compiled Python) | `compile_program`, `run`, `SpecPlayer` | `python`, `run TUNE --binder mod:attr` |
| SMT transition relation, bounded checking | `smt.transition`, `smt.bmc` | `bmc SPEC --binding JSON --frames N` |
| Translation validation against 6502 code | `tv.validate`, `tv.replay` | `tv TUNE --binder ... --routine ... --pc ...` |
| Replace `native` hooks with spec statements | `unnative`, `equivalent` | `unnative SPEC TUNE --binder ... -o OUT` |

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
    - `notes(lo|hi|word, K)`: the `ro` table holds the native note table; index `i` means standard note `i + K` (`K` a static, possibly negative). `lo`/`hi` mark a `u8[n]` as the low/high byte of the note word, `word` a `u16[n]`/`ptr[n]` as the whole word. Cells aliasing one table may declare different `K`.
- `machine m [x in a..b] { cell/signal/control/proc/step }` declares a machine
  and its instances; `x` is the static instance index.
- `control c [on cell mask M] { s = v, ... }` declares control states. With
  `on cell mask M` they are bits of a memory cell (`in c { s => {...} }`,
  `goto s`); without it they are a hidden persistent variable.
- `proc f(a: u8, s: stream) -> u8 { ... }` declares a procedure. Procedures are
  inlined, `return` is allowed, and streams pass by reference.
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

## Execution

The lowered IR is one structured program per call:
- procs are inlined;
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
- **Timing inside a call.** A call is an instant, so a read of `$D41B`/`$D41C`
  (oscillator 3 / envelope 3) or a raster poll sees the value at the call's
  start cycle, not the cycle the 6502 would reach it.
- **Timing of a trapped call under interrupts.** A tune with play address 0 runs
  on an `InterruptPlayer` (its interrupts live, `render_grid` framed at the play
  source's raises) and `SpecPlayer` traps each call into an `entries(view)` entry
  there and runs it as the spec. The call is an instant at the cycle the handler
  reaches it, so the cycles the 6502 would spend in it are not spent, and a
  handler that reaches an entry by a vector or `JMP` rather than `JSR` is
  returned from as an `RTS`. Pass `sid=` (the PSID/RSID) for the header's SID
  model and extra SIDs; without it a header is built around the image.
- **Opcodes.** Translation validation keeps opcodes constant: a routine that
  rewrites an opcode it then executes is refused, not modelled. The DSL still
  expresses such a choice as an `operand` cell.
