# Native playspec execution

`SpecPlayer` runs the lowered program of each modelled entry on `pysidtracker._specvm`, a Rust
extension (`crates/specvm-core`, `crates/specvm-py`), when the extension is built and the player
is one it runs. The Python executor (`pyexec.compile_program`) is the oracle and the fallback:
every result of the native machine is identical to it.

## Use

- Wheels on Linux carry the extension (`cp311-abi3`, manylinux, x86_64 and aarch64); the sdist and
  other platforms install the pure-Python package.
- `PYSIDTRACKER_SPEC_NATIVE=0` disables the machine; `SpecPlayer(native=True|False)` overrides the
  environment for one player.
- `SpecPlayer.render_grid` plays whole runs of frames in one native call and returns the same
  lists of lists; `play_frame` plays one frame natively. `SpecPlayer.writes` is that of the last
  frame either way.
- `maturin develop --release` builds the extension into the source tree for development.

## What runs natively

A player runs natively when it is not hosted (play address non-zero), has no stream validator
and no `inputs`, records no reads, and runs on jennings' native core. Its machine then plays
whole frames: the tune's play routine on `jennings-core` over the tune's RAM, and each call into
a modelled entry as that entry's program.

- **Programs.** `playspec.native.encode` flattens a lowered `Program` to `(op, arg)` instructions
  (expression operators with Python's integer semantics, constants folded into immediate forms,
  compare-and-branch, conditional expressions evaluated lazily as Python does). The core checks a
  program once (operands, jump targets, a consistent stack at every instruction) and then runs it
  without bounds checks.
- **Memory maps.** Each setting of the CPU port's low bits has its own map (ROM pages, I/O) and
  modelled entries, added by Python as it meets them; the port is read and written natively.
- **Bails.** A frame that needs anything Python answers is undone from a log of its stores and
  played by Python, and native play resumes at the next frame. The causes: a program or a memory
  map not yet built, a spec store to the CPU port, a read or write that reaches a device (the VIC
  raster registers, the SID's `$D41B`/`$D41C` and the writes that feed them), a value or address
  Python would reject, a failed assertion or loop bound (Python then raises it). A player whose
  bails are frequent plays in Python from then on.
- **Python only.** Programs with `native` blocks, hosted tunes, `validate`, `inputs`.

## Checks

- `tools/spec_native_diff.py` plays tunes with both executors and compares the register grids and
  the per-frame `writes`; it names the format, tune, frame and register of a difference.
- `tools/spec_bench.py` times both on tunes.
- `tests/test_playspec_native.py` compares random programs, and tunes with wrapped, banked and
  device-reading play routines, against the Python executor.
