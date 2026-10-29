# Native-tick play wrappers

Many releases call the player from a play routine of their own: a sound-effect
mixer, a multi-speed loop, or a flush that copies a "ghost" register block the
player filled into the SID. A Python port of the player reproduces the
player, not that wrapper. `pysidtracker.wrap` runs the wrapper itself in the
jennings emulator and stands a Python tick in for each call it makes into the
player.

## Measuring the wrapper

```python
from pysidtracker.codescan import jump_table, play_entries
from pysidtracker.wrap import Wrapped, count_entries, measure_wrap

base = jump_table(image, data_start)        # init/play/... JMP table, or None
entries = play_entries(image, base)         # {base + 3, where its JMP leads}
wrap = measure_wrap(image, song, image.header.play_address, entries)
ticks = count_entries(image, song, image.header.play_address, entries)
```

| call | result |
|------|--------|
| `jump_table(image, end)` | the first entry of the last run of two or more `JMP abs` before `end` whose second entry's target lies in `[run, end)` (a three-entry init/play/volume table's play entry is its second), else `None` |
| `play_entries(image, base)` | `{base + 3}` plus the target of the `JMP` there |
| `measure_wrap(image, song, play, entries, max_cycles=PLAY_CYCLES)` | `None` when `play` is `None`, `entries` is empty or `play` is an entry; else `Wrap(play, entries, song, sink)` |
| `count_entries(image, song, play, entries, max_cycles=PLAY_CYCLES)` | how many times the play call after the first enters the target of its first `JSR`/`JMP` when that target is in `entries`, else 1 |

`measure_wrap` boots the image and runs two play calls, marking each SID write
(`$D400-$D418`) as inside or outside the dynamic extent of an entry: entered at a
PC in `entries`, left when the stack pointer rises above the one it was entered
at. On the second call, if no write is inside and some are outside, `sink` is the
most common `address read last − register` of the outside writes: the ghost block
the wrapper copies from, register for register. Otherwise `sink` is `None` and the
player writes the SID itself.

## Playing through the wrapper

```python
def tick():                                 # one native player frame
    player.play_frame()
    return {wrap.sink or 0xD400: player.regs}

for writes in Wrapped(tick, image, wrap).iter_frames(frames):
    ...
```

`Wrapped(tick, image, wrap, max_cycles=PLAY_CYCLES)` is a `MemPlayer`. It boots
the image for `wrap.song` (`boot`: `oracle.boot_flat`, with `idle_init` running
the init of a tune without a play address to its idle loop) and each frame calls
`wrap.play` as `run_to_rts` does. Whenever the emulated PC reaches an entry it
calls `tick()`, writes each returned `{address: values}` block through the bus,
then returns from the entry as `rts` would (the return address is pulled from the
stack). A wrapper that calls the player twice gets two ticks a frame; one that
flushes before jumping into the player plays the previous tick's registers.
`MASK_PW_HI = True` nibble-masks the pulse-width highs in `snapshot` as
`EmuPlayer` does.

`enter(mpu, mem, pc, acc=0)` and `leave(mpu, mem)` are the call/return mechanics
for a caller stepping the CPU itself.

## SID write log

`emu.SidWriteLog(mpu, mirrors=True)` attaches to an MPU with
`observe_data_access` and records, per play call, `(register, address read last,
pc, value)` for each SID write (`mirrors=False`: `$D400-$D418` only). `start()`
opens a call's list; `call(mem, play, max_cycles, error)` opens one and runs
`emu.call_play`, which runs one play call and raises `error` (default `SidError`)
when it does not return within `max_cycles`.
