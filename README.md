# pysidtracker

Shared base for the pure-Python C64 SID tracker parsers (pygoattracker,
pysidwizard, pydmcsid, pyfuturecomposer, pymusicassembler, pydefmon, pyjch,
pysoundmonitor).

Reads `.sid` containers (PSID/RSID) and bare `.prg` images into a 64 KiB C64
memory model, and detects packed/relocating playroutines by running the tune's
init in a 6502 emulator — container headers are not trusted. Each format
package subclasses `BaseSidParser` for a consistent `read` / `parse` / `detect`
API.

## Components

- `parse_sid_header` / `SidHeader` — PSID/RSID container header parsing.
- `SidImage` — a loaded 64 KiB C64 memory image with absolute-addressed
  accessors, from a `.sid` container or a bare `.prg`.
- `read_bytes` — path / `bytes` / file-like source dispatch.
- `SidError` hierarchy — `SidParseError`, `SidFormatError`,
  `EmulatorUnavailable`.
- `detect_playroutine` / `PlayroutineKind` — static signature recognition, then
  an emulated init run to classify `DIRECT` / `RELOCATED` / `PACKED` /
  `UNKNOWN`.
- `BaseSidParser` — the base class each format subclasses.
- `CodePattern` / `find_code_all` / `find_code_first` — masked 6502
  code-fragment search with operand capture.
- `registers` — C64 hardware register map (SID/CIA/VIC, IRQ/NMI and CPU
  vectors) with predicates and `find_register_stores`.
- `trace_init` / `InitTrace` — run a tune's init under a write observer to
  report CIA timer cadence and the IRQ/NMI vectors an IRQ-driven header hides.
- `playroutine_cadence` / `Cadence` / `TriggerSource` — derive the play-routine
  cadence (PAL/NTSC video frame vs CIA-timer latch) from what init programs, not
  the header.
- `native_decrunch` — native exomizer unpack (pydexomizer), an init-free
  alternative depack path; opt-in via `detect_playroutine(..., native=True)`.
- `reglog` — `RegWrite` register-log format (`read_reglog` / `write_reglog`) and
  `frame_writes`, the shared per-frame SID-write framing loop.
- `oracle` — per-frame SID register grids: `register_grid` (py65), the stdlib
  `grid_from_writes` framer, `read_sidwr` (`preframr-sidtrace` `.sidwr.bin`),
  `aligned_match`, and the `sidtrace` (sidplayfp) oracle bridge — `run_sidtrace`,
  `read_sidtrace`, `sidtrace_cadence`, `sidtrace_grid`.
- `testing` — HVSC tune fetch/resolve for test suites (`fetch_tune`,
  `resolve_tune`, `make_tune_fixtures`), and `make_oracle_fixtures` /
  `oracle_grid` — a reusable byte-exact player-vs-`sidtrace`-oracle test
  ([docs/oracle-testing.md](docs/oracle-testing.md)).
- `audio` — `render_player_wav` / `render_player_samples` drive a `MemPlayer`
  straight to WAV through an emulated SID (pyresidfp, a core dependency); the
  lower-level `render_wav` / `render_samples` take a bare write stream, plus
  `resolve_device` / `seconds_to_frames`.
- Writers — `SidHeader.to_bytes` / `write_psid` / `SidImage.to_prg` (the inverse
  of the parsers), `parse_prg`, and `encode_cstr` / `decode_cstr`.
- `SidImage` absolute-address accessors — `byte_at` / `word_at` / `contains` /
  `poke`, relocation-safe and bounds-checked.
- `mos6502` — shared 6502 primitives: `OP_LEN`, opcode-class sets, `s8` / `adc`
  / `sbc`, `SidWriteCapturingMemory`, `walk_until`.
- `MemPlayer` — per-frame register-diff player scaffold; subclasses implement
  `_init` / `_frame`. `register_writes_from_player` drives it to a reglog.
- `notefreq` — `NoteFreqTable`, computed PAL/NTSC pitch tables, and
  `locate_note_freq` (find hi/lo tables from paired absolute-indexed reads).
- `make_package_errors` — per-package error hierarchy factory whose `*ParseError`
  / `*FormatError` subclass the base ones, so base `except` clauses catch them.
- `formats` / `maincli` — the one `pysidtracker` CLI (`info` / `reglog` / `wav`)
  that discovers installed formats through the `pysidtracker.formats`
  entry-point group and runs against each format's parser + player, plus any
  subcommands it contributes. A format package registers a `SidFormat` instead
  of shipping its own CLI. `cli` holds the underlying argparse scaffold
  (`run_cli`, `print_info`, `add_reglog_command`, `add_wav_command`).
- `registers` additions — `SID_VOICES`, voice/global register indices, and the
  `attack_decay` / `sustain_release` ADSR nibble packers.
- `resolve_entry_points` / `is_jmp_vector`, `cadence_from_latch`, and the
  `ByteCursor` / `check` / `byte_range` parse helpers.

## Install

```bash
pip install pysidtracker   # core: py65 (init emulation), numpy, pyresidfp (WAV render)
```

All dependencies are core: py65 (detection runs a tune's 6502 init), pydexomizer
(`native_decrunch` unpacks exomizer-packed images natively), pyresidfp (the
emulated SID the WAV render drives), and numpy (sample buffers and the image
anchor scan).

## Usage

```python
from pysidtracker import BaseSidParser, PlayroutineKind

class MyParser(BaseSidParser):
    def recognize(self, image):
        return image.find(b"MYSIG")            # truthy anchor when found
    def parse(self, data, **kw):
        image = self.load_image(data)          # .sid or .prg -> SidImage
        ...                                    # decode image.mem into a model

det = MyParser().detect("tune.sid")
if det.kind is PlayroutineKind.PACKED:
    ...                                        # header was not trustworthy
```

## Development

```bash
pip install -e ".[dev]"
pytest --cov=pysidtracker
```

See [docs/design.md](docs/design.md) for the detection model and shared
primitives.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
