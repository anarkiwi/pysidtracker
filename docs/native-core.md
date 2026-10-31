# Native core

`emu.wire_mpu` runs on jennings' native 6510 core (`jennings.memory.NATIVE`) through
`vagrant_moses.cpu.flat_mpu`, and on the pure-Python engine when the core is absent or
`native=False`. Results are identical on both; the test suite and `tools/diff_backends.py`
check it.

## RAM ownership

The native core copies the buffer given to `wire_mpu` and owns the RAM afterwards.

- `live_ram(mpu, buffer)`: the buffer the MPU runs on (a zero-copy view natively).
- `sync_ram(mpu)`: copy the RAM back into the wired buffer, for code that reads it after a run.

## Drivers

| Helper | Native | Python engine |
|---|---|---|
| `run_to_rts`, `run_to_rts_until` | `mpu.run_to_rts[_until]` | stepped |
| `run_until(mpu, stop_pcs, sp_above, max_cycles)` | `mpu.run_until` | stepped, same stop reasons |
| `trace_pcs(mpu, mem, pc, acc, max_cycles, sink)` | `mpu.run_to_rts_traced` | stepped, PC appended live |
| `step_to_rts` | stepped | stepped |

A JAM ends a native `run_until` after the instruction; the Python engine's JAM skips a byte
and continues, so callers treat `"jam"` as progress.

## Observers

`observe_data_access(mpu, on_read, on_write, reads=, writes=)` uses the core's observer with
`stack=True`, so `PHA`/`PLA`/`PHP`/`PLP` page-1 accesses are reported as the Python engine
reports them. `SidWriteLog` observes SID writes only and reads `mem.last_load`.

## What still steps from Python

- `cadence._Machine` runs to the next event (`execute_events`); `execute_stepped` is the
  reference loop. A run shorter than `_SHORT_RUN` cycles, or starting at a watched address
  (possible self-spin, instruction that can unmask a pending IRQ), is stepped.
- `whydiff.cpu_history`, `provenance` and `playspec/tv.replay` decode or compare memory at
  each instruction, so they step.
- `irqhost.InterruptHost` stays on `vagrant_moses.machine.Machine`.

## Tools

- `tools/diff_backends.py`: every result on both engines; exit status is the number of differences.
- `tools/bench_detect.py`: detection, cadence and one second of oracle play per tune on both
  engines; `--threads N` runs detection of N tunes in N threads.
