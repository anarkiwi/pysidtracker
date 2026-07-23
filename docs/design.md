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
   (`jennings`), then recognise again. Classify by what init did to memory:
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

### `trace_init` — jennings init/vector tracer

`trace_init(image, play_calls=N)` runs the tune's init on the shared `emu`
host (see below) with an `ObservableMemory` write observer over the
hardware-register and interrupt-vector addresses, optionally calls play `N`
times, and returns an `InitTrace`: the CIA timer latches (`cia1_timer_latch` /
`cia2_timer_latch`, the play cadence), the installed `irq_vector`
(`$0314`/`$0315`) and `hw_irq_vector` (`$FFFE`/`$FFFF`, the real play routine),
`nmi_vector`, `vic_raster`, the `registers_touched` set and the `sid_writes`.
This reveals the play address and cadence that an IRQ-driven header hides
(e.g. Soundmonitor's CIA-timed cohort). Requires the core `jennings` dependency;
raises `EmulatorUnavailable` if jennings is missing.

### `emu` — the shared jennings 6502 host

`wire_mpu(subject, illegal_opcodes=True)` builds the MPU that `run_init`,
`trace_init` and `register_grid` all run on. jennings decodes the full NMOS
illegal set (SLO/RLA/SRE/RRA/SAX/LAX/DCP/ISC, ANC/ALR/ARR/SBX/SBC/LAX#/ANE,
SHY/SHX/AHX/TAS/LAS and the multi-byte NOPs) natively, byte/cycle-exact, but has
no C64 reads, so `wire_mpu` synthesises cycle-derived VIC raster (`$D011`/`$D012`)
and SID osc3/env3 (`$D41B`/`$D41C`) reads to let sync spin loops terminate.
Without them an init that raster-syncs never reaches its `$DC04`/`$DC05` writes and
its cadence silently falls back to the video frame. The `illegal_opcodes` flag is
kept for API compatibility only (jennings is always native). `run_to_rts(mpu, mem,
pc, acc, max_cycles)` is the shared push-a-return-address-and-step mechanic.

## What each format supplies

A parser subclasses `BaseSidParser`, implements `parse(data) -> model`, and
optionally overrides `recognize(image) -> anchor` to opt into detection. It
inherits `read`, `load_image`, and `detect`. Format-specific error classes
subclass `SidError` (or `SidParseError`) so `except SidError` works across all
of them.

## 0.3.0 shared player/validation surfaces

Five surfaces the format packages had each hand-copied during byte-exact HVSC
validation, consolidated here.

### Hardware constants (`registers`)

`registers` gains documented C64 timing/layout facts: `PAL_CYCLES_PER_FRAME`
(19656), `NTSC_CYCLES_PER_FRAME` (17095), `PAL_CLOCK_HZ` (985248),
`NTSC_CLOCK_HZ` (1022727), `PW_HI_REGS` (`(0x03, 0x0A, 0x11)` — the
pulse-width-high register offsets, masked to 4 bits in a register grid),
`SID_VOICE_OFFSET` (`(0, 7, 14)`) and `SID_REG_COUNT` (25, the `$D400..$D418`
file). `SID_BASE` (`0xD400`) is unchanged.

### `reglog` — register-write log convention

A register log is a player's output flattened to timed chip writes: one
`RegWrite(clock, reg, val)` (namedtuple; `reg` is the `0..$18` offset from
`$D400`) per SID write, serialized as one decimal `clock reg val` triple per
line (`#` comments allowed, first line `REGLOG_HEADER`).

- `write_reglog(writes, dst, header=True)` / `read_reglog(src)` — path or text
  file-like; a malformed line raises `SidParseError` (not `ValueError`).
- `frame_writes(per_frame_iter, *, cycles_per_frame, write_spacing=16,
  start_frame=0, sid_reg_base=0xD400, reg_count=25)` — the shared framing loop.
  For frame `f`, each `(reg, val)` write is rebased (`reg - sid_reg_base`) and
  masked (`val & 0xFF`); rebased regs in `0..0x18` are emitted at
  `clock = f*cycles_per_frame + offset*write_spacing` (`offset` increments per
  emitted write). Pass `sid_reg_base=0` when the player already yields `0..24`
  offsets. Raises `SidParseError` if `write_spacing * reg_count >=
  cycles_per_frame`.

### `oracle` — register-grid ground truth

- `register_grid(image_or_bytes, nframes, *, subtune=0, illegal_opcodes=False)`
  — runs `init` (accumulator `= subtune`) then `nframes` `play` calls on the
  shared `emu` host, sampling `$D400..$D418` (25 registers) per frame.
  `illegal_opcodes=True` installs the NMOS illegal opcodes defMON needs
  (default off here, on everywhere else). Requires jennings; raises
  `EmulatorUnavailable` if missing.
- `grid_from_writes(writes, *, cycles_per_frame=19656, reg_count=25,
  pw_hi_regs=(0x03,0x0A,0x11), gap=10000)` — pure-stdlib framer: anchor frame 0
  to the first play call (first write after a `>gap`-cycle gap), forward-fill,
  nibble-mask `pw_hi_regs`; frame assignment rounds to nearest
  (`(clock - t0 + cpf//2)//cpf`).
- `read_sidwr(path)` — parse a `preframr-sidtrace` `.sidwr.bin` record stream
  (`struct.Struct("<qHBB")` = clock, addr, reg, val); drops the addr field and
  records with `reg >= 25`, returning `(clock, reg, val)` triples.
- `aligned_match(oracle, rendered, *, max_lead=4) -> bool` — True if `rendered`
  matches `oracle` allowing up to `max_lead` leading silent frames.

### `testing` — HVSC fetch/resolve

Ships in the wheel (pure stdlib + a lazy pytest import).

- `DEFAULT_MIRROR = "https://hvsc.brona.dk/HVSC/C64Music"` (`$HVSC_MIRROR`
  overrides).
- `fetch_tune(relpath, *, cache_dir, mirror=DEFAULT_MIRROR, retries=4,
  force=False)` — cache-check, HTTPS GET with a User-Agent, PSID/RSID magic
  validation, exponential backoff, atomic write; raises `TuneFetchError` on a
  404 or after `retries`.
- `resolve_tune(relpath, *, cache_dir, local_env="HVSC")` — local `$HVSC` tree,
  then `cache_dir`, then fetch; `None` only when genuinely unreachable.
- `make_tune_fixtures(tunes, cache_dir, ...)` — a pytest fixture factory
  returning parametrized `tune_id` / `tune_path` fixtures (pytest imported
  lazily).

### `audio` — pyresidfp WAV render (`audio` extra)

`render_samples(frame_iter, *, model, sampling_frequency, cycles_per_frame,
clock_frequency, write_spacing=16, device=None)` and `render_wav(...)` clock a
per-frame `(reg, val)` write stream through an emulated SID one write at a time
(so renders line up with `reglog`). `model` in `("6581", "8580")`. pyresidfp is
imported lazily; a missing `audio` extra raises `AudioUnavailable`. Any object
with `write_register` / `clock(timedelta)` / `sampling_frequency` may be passed
as `device`.

`default_device(model, sampling_frequency=None, clock_frequency=None)` is the
public accessor dependents use to construct that default pyresidfp device (the
old private `_default_device` remains as a thin alias for back-compat), and
`device_sampling_frequency(device)` reads back a device's output rate.

## 0.4.0 playroutine cadence and native decrunch

### `cadence` — derived play-routine cadence

*Playroutine cadence* — the CPU cycles between consecutive play-routine calls —
is a **global SID concept**, not a per-format constant. It is set by whatever
triggers the play interrupt:

- a **PAL** video frame (`PAL_CYCLES_PER_FRAME` = 19656 cycles),
- an **NTSC** video frame (`NTSC_CYCLES_PER_FRAME` = 17095 cycles), or
- a **CIA/NMI timer** whose latch defines an arbitrary period.

The header's speed/clock fields are only a *hint* (the base's guiding principle:
headers are untrustworthy). IRQ-driven tunes program the real trigger from
inside init, and a player can reprogram the timer mid-tune, so cadence is not
guaranteed constant.

`playroutine_cadence(image_or_bytes, *, clock=None, play_calls=8) -> Cadence`:

1. Resolve the video standard — explicit `clock` (`"PAL"`/`"NTSC"`) wins, else
   the PSID/RSID v2+ header `flags` clock bits (bits 2–3: `%01`=PAL, `%10`=NTSC)
   as a hint, else PAL.
2. `trace_init` observes what init installs (reusing the jennings plumbing, not
   duplicating it): the CIA Timer-A latch(es), the IRQ/NMI vectors, the VIC
   raster compare, and — via a small extension — whether a play call rewrites a
   Timer-A latch (`cia{1,2}_latch_rewritten`).
3. Decide the source: if init programs a plausible CIA Timer-A latch (≥ 256, so
   a lone `$FF` reset write to `$DC04` is ignored) the cadence is
   **CIA-driven**; otherwise it is **video-timed** (one PAL/NTSC frame).
4. `dynamic` is set when a play call rewrites the chosen Timer-A latch to a
   different value (a variable-tempo player). A full per-call cadence *schedule*
   is a documented future extension (`# TODO`); the dynamic case is detected and
   flagged now.

`Cadence` (frozen dataclass): `cycles_per_call`, `source` (`TriggerSource`
enum: `PAL_VIDEO` / `NTSC_VIDEO` / `CIA_TIMER`), `clock_hz` (`PAL_CLOCK_HZ` /
`NTSC_CLOCK_HZ`), `latch` (the CIA latch when CIA-driven, else `None`),
`dynamic`.

**CIA period (the +1).** A CIA timer in continuous (reload) mode is loaded with
its 16-bit latch `L` and counts `L, L-1, …, 1, 0`; on the cycle after `0` it
underflows (raising the interrupt) and reloads `L`. Consecutive underflows are
therefore `L + 1` cycles apart, so **`cycles_per_call = latch + 1`**. This is
validated against real defMON tunes (Aleksi Eeben), whose init self-installs a
CIA-timer play IRQ: e.g. `Stella_2600_by_Starlight` latches 19759 → 19760
cycles, and defMON's canonical **23546** cadence is a **23545** latch + 1. The
latch is the tune's tempo, so it is tune-specific; the `+1` relationship is the
invariant. Replaces per-format cadence constants (e.g. defMON's hard-coded
23546) with a single derivation.

### `decrunch` — native exomizer unpack

Packed tunes ship an exomizer-crunched payload; `detect.run_init` materialises
it by emulating the tune's whole init. `native_decrunch(image_or_bytes) ->
Optional[SidImage]` is the targeted alternative: it runs *only* the exomizer
decruncher via [`pydexomizer`](https://pypi.org/project/pydexomizer/) (a core
dependency, like jennings) — trying the self-extracting `sfx` format (bounded step
cap so a non-exomizer image fails fast), then the `mem` format
(`decrunch_mem_auto`) — and returns the unpacked image, or `None` when the image
is not exomizer-packed. It never raises for the not-packed case, so it is safe
as a first-try step.

`detect_playroutine(..., native=False)` gains an **opt-in** `native` flag: when
true, an exomizer-packed image is decrunched natively first (no init emulated) →
reported `PACKED` with `ran_init=False`; if that does not yield a recognised
image, detection falls back to the existing emulated-init path. Native decrunch
is opt-in specifically so the default detection behaviour — and every existing
detect test — is unchanged.
