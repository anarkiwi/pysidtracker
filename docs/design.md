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

## What each format supplies

A parser subclasses `BaseSidParser`, implements `parse(data) -> model`, and
optionally overrides `recognize(image) -> anchor` to opt into detection. It
inherits `read`, `load_image`, and `detect`. Format-specific error classes
subclass `SidError` (or `SidParseError`) so `except SidError` works across all
of them.
