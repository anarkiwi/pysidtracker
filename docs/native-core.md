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

- `cadence._Machine` runs to the next event (`execute_events`): `run_until` with the KERNAL window as a
  `stop_ranges` entry, `stop_on_idle` for the idle test and `stop_on_irq` while an asserted IRQ is masked; `execute_stepped`
  is the reference loop.
- `whydiff.cpu_history` and `provenance` run one traced native call (`run_to_rts_traced`, registers
  on) and rebuild the code as it stood at each instruction by replaying the recorded stores
  onto a copy of the memory; `playspec/tv.replay` runs natively between JSR/RTS opcodes with the
  instruction budget kept by cycle-bounded chunks.
- `irqhost.InterruptHost` runs on `vagrant_moses.machine.Machine` with one `NativeSid` per chip. The
  device logs writes natively and answers every register but `$19-$1C` itself; a read of those calls
  back into Python, which feeds the chip's logged writes to a pylibresidfp `SidReads` and returns its value.
  `raw` is built lazily from the devices' changed-write rows and the event sink, merged by cycle
  (stable, writes before a raise at the same cycle).

## Tools

- `tools/diff_backends.py`: every result on both engines; exit status is the number of differences.
- `tools/bench_detect.py`: detection, cadence and one second of oracle play per tune on both
  engines, with the native run's Python/native split and SID write count; `--threads N` runs
  detection of N tunes in N threads.
- `tools/bench_irqhost.py`: `InterruptHost` wall time, x-realtime, Python/native split and SID
  write and `$19-$1C` read counts per tune.
- `tools/dump_irqhost_rows.py`: pickles `InterruptHost.raw` of every fixture tune, to diff two
  implementations.

## Threads

`tools/bench_threads.py` runs 16 tasks per workload on 8 workers. Speedup over one worker:

| Workload | Threads | Processes |
|---|---|---|
| detection with init emulation, 20 per task | 0.3x | 5.1x |
| cadence, 20 per task | 0.2x | 2.7x |
| oracle play, 20 per task | 0.2x | 2.2x |
| `InterruptHost`, 30 emulated seconds, no SID reads | 4.8x | 5.6x |
| `InterruptHost`, 3 emulated seconds, rows collected | 0.7x | 2.4x |
| `InterruptHost`, 3 emulated seconds, OSC3 reads | 0.6x | 5.8x |
| native core, no callbacks | 5.2x | 4.3x |
| SIDId signature scan | 0.3x | 0.5x |

The native core and the native `Machine` release the GIL while they run, and scale when nothing
calls back into Python. Event callbacks, the Python glue around detection, cadence and the oracle,
building the row tuples of `InterruptHost.raw`, pylibresidfp and the `$19-$1C` read callbacks hold the
GIL, so the stages the tools run (detection, cadence, oracle, rows) are slower on threads than serial.
Process pools therefore stay the default everywhere; `--threads` (`sidid`, `survey find`,
`survey triage`) runs the same work on a thread pool, with worker state initialised once and
read-only. Jobs under a `deadline` are killed from outside and need processes.

Module state, for a thread pool: the `lru_cache` memos are thread-safe and idempotent, the
assembler builds its MPU per call, and each task builds its own MPU or `Machine`.
