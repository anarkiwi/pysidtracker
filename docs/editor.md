# Editor-player oracles and subtune start tracing

## `pysidtracker.editor`

A tracker editor's own code is the reference for what its LOAD decoder, SAVE
packer and live player do. Given the editor's decrunched 64 KiB RAM image (a
copyright work: fetch and cache it, never commit it):

| call | result |
|------|--------|
| `ram_image(ram, init, play=None, flags=0x10)` | the RAM as a flat `SidImage` at `$0000` whose header calls `init` / `play` |
| `run_editor(ram, payload, at, entries, *, load_end=(0xAE,), keep_bank=True, keep_volume=True, max_cycles=30_000_000)` | the memory after placing `payload` at `at` and running `entries` |
| `editor_replay(ram, payload, at, entries, window, init, play, *, flags=0, **run_kwargs)` | a PSID of `window = (lo, hi)` of that memory, loaded at `lo`, playing `init` / `play` |

`run_editor` boots with `oracle.boot_flat` (power-on RAM, the PSID driver's
bank and `$D418`), then, before the first entry runs, puts back the RAM's own
`$01` (`keep_bank`, also after the boot) and `$D418` (`keep_volume`), places the
payload and writes the KERNAL `LOAD` end word `at + len(payload)` to each
`load_end` address. `entries[0]` runs as init, the rest in order with `A = 0`
(`emu.run_to_rts`). Pass the editor's LOAD decoder (and a packer) as entries;
`editor_replay` then wraps the editor's live player so an `EmuPlayer` subclass
renders the song exactly as the composer hears it.

```python
psid = editor_replay(ram, prg[2:], 0x1800, (DECODE,), (0x1000, 0x7167), 0x1000, 0x1003)
grid = register_grid(SidImage.from_bytes(psid), frames)
```

## `pysidtracker.trace` subtune starts

A replay's init often wraps the player: `TAX / LDA table,X / JMP base`, so the
song the player starts is the value `A` holds at `base`, not the subtune.

- `trace_init_argument(image, init, entry, subtune=0, max_cycles=1_000_000)`
  runs `init` (`None`: the header's) with `A = subtune` on a copy of
  `image.mem`, banked as the PSID driver banks init, and returns `A` when the
  PC first reaches `entry`, or `None` if init returns or runs out of cycles.
- `init_arguments(image, entry, init=None, max_cycles=1_000_000)` is that for
  every subtune of the header's song count (one without a header).
