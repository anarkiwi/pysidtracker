# Design

## Why a shared base

The seven format packages each unwrapped a PSID/RSID container, resolved the
load address, built a memory image, and dispatched `read(path|bytes|file)` — the
same code seven times, with subtly different error types and accessors. This
package is the single source of truth for the container and image handling, and
a base class that gives every parser the same public surface.

## Untrustworthy headers

A `.sid` header's `loadAddress`, `initAddress`, `playAddress` and `songs`
describe what the *file* loads and calls, not necessarily where the *song data*
lives. Packed, crunched, and relocating tunes ship a loader/depacker: after the
init routine runs it unpacks or relocates the real tables into memory. Reading
the header fields as if they pointed at song data mis-parses these tunes.

`detect_playroutine(image, recognize, ...)` handles this uniformly:

1. **Static recognition.** The format's `recognize(image)` callback looks for
   its signature/anchor in the freshly loaded image. Found → `DIRECT`.
2. **Emulated init.** Otherwise run the init routine in a 6502 emulator
   (`py65`), then recognise again. Classify by what init did to memory:
   - wrote more *outside* the original load region than it changed *inside* →
     `PACKED` (decompression expands memory);
   - otherwise → `RELOCATED` (moved/rewrote the existing image).
3. Still not found → `UNKNOWN`.

The inside/outside byte counts are heuristics for the `PACKED` vs `RELOCATED`
label and are exposed on the `Detection` (`changed_inside`, `written_outside`)
for callers that want to apply their own rule. The load-bearing guarantee is
step 1 vs step 2: whether the header described the data directly (`DIRECT`,
`trustworthy_header`) or the tune had to be run to materialise it.

This generalises pygoattracker's `.sid` decompiler, which finds GoatTracker's
split note-frequency table statically and falls back to an emulated init for
crunched/relocated images. `SidImage.find_split_table` is that anchor search,
lifted here for reuse.

## Relocation-invariant primitives

Relocatable players hide their per-tune data behind absolute addresses baked
into the player code as instruction operands; the player relocates as one block,
so a fixed image offset cannot read a table. Every format's reader independently
re-invented the same three primitives, now shared here.

### `codescan` — masked code-fragment search

`CodePattern("A9 {imm} 8D 05 D4")` compiles an opcode skeleton where a two-hex
literal matches exactly, `??` is a wildcard byte, `{name}` captures one operand
byte and `{name:w}` captures a little-endian word. `find_code_all(image, pat)` /
`find_code_first(...)` scan `image.mem` (from `image.load`) and return
`Match(addr, captures)`. This one abstraction expresses every parser
fingerprint: pyfuturecomposer's and pydefmon's `None`-wildcard token lists,
pymusicassembler's `re` capture groups (`BD {lo:w} 85 FA …`), pyjch's
`prefix<operand>suffix` idioms, and JCH/defMON SID-store signatures. The
literal/wildcard prefilter is numpy-accelerated with a pure-stdlib fallback.

### `registers` — hardware register map

Documented C64 hardware addresses (SID `$D400`–`$D418` mirrored every `$20`;
CIA1 `$DC00`–`$DC0F`; CIA2 `$DD00`–`$DD0F`; VIC `$D000`–`$D02E`; RAM IRQ/NMI
vectors `$0314`/`$0318`; CPU vectors `$FFFA`/`$FFFC`/`$FFFE`) as named constants
plus predicates (`is_sid_reg`, `is_cia_timer`, `is_vic_reg`, …) and
`find_register_stores(image, addrs)`, which scans absolute `STA`/`STX`/`STY`
(`8D`/`8E`/`8C`/`9D`/`99`) targeting a set of addresses. These are public
hardware facts, not player code.

### `trace_init` — py65 init/vector tracer

`trace_init(image, play_calls=N)` runs the tune's init (the `run_init`
mechanics) in py65 with an `ObservableMemory` write observer over the
hardware-register and interrupt-vector addresses, optionally calls play `N`
times, and returns an `InitTrace`: the CIA timer latches (`cia1_timer_latch` /
`cia2_timer_latch`, the play cadence), the installed `irq_vector`
(`$0314`/`$0315`) and `hw_irq_vector` (`$FFFE`/`$FFFF`, the real play routine),
`nmi_vector`, `vic_raster`, the `registers_touched` set and the `sid_writes`.
This reveals the play address and cadence that an IRQ-driven header hides
(e.g. Soundmonitor's CIA-timed cohort). Requires the `emu` extra; raises
`EmulatorUnavailable` if py65 is missing.

## What each format supplies

A parser subclasses `BaseSidParser`, implements `parse(data) -> model`, and
optionally overrides `recognize(image) -> anchor` to opt into detection. It
inherits `read`, `load_image`, and `detect`. Format-specific error classes
subclass `SidError` (or `SidParseError`) so `except SidError` works across all
of them.
