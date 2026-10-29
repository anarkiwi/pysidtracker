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
- `registers` — SID register semantics, PSID flag helpers and
  `find_register_stores`; the C64 memory map is vagrant-moses' `memmap` (the
  frame timing and CIA latch names dependents import are re-exported).
- `trace_init` / `InitTrace` — run a tune's init under a write observer to
  report CIA timer cadence and the IRQ/NMI vectors an IRQ-driven header hides.
- `init_effects` / `subtune_selection` / `init_diff` — run one song's init on a
  private snapshot and record what it left in RAM (cell stores/reads with their
  PCs, the installed vectors), mapping header songs to a selection cell or
  listing the cells that differ between songs.
- `emu` — the shared jennings host every init/replay path runs on: `wire_mpu`
  (vagrant-moses' banked `cpu.flat_mpu` plus reSIDfp osc3/env3 reads) and
  `run_to_rts`.
- `playroutine_cadence` / `Cadence` / `TriggerSource` — derive the play-routine
  cadence (raster calls per frame, CIA/NMI timer period) from how the tune
  programs its interrupts in sidplayfp's PSID driver environment
  ([docs/cadence.md](docs/cadence.md)); `cadence_info` is its survey `info`
  fields and `cadence_check` validates it against sidtrace interrupt rows.
- `irqhost` — `InterruptHost` / `InterruptPlayer` boot and run a PSID or RSID as
  the sidtrace oracle's sidplayfp does, on vagrant-moses' cycle-exact `Machine`
  with reSIDfp SID devices, its trace identical to the oracle's; `TunePlayer`
  plays any PSID/RSID's own driver (hosting play address 0);
  `host_cadence` measures its play cadence and `host_check` compares traces
  ([docs/irqhost.md](docs/irqhost.md)).
- `native_decrunch` — native exomizer unpack (pydexomizer), an init-free
  alternative depack path; opt-in via `detect_playroutine(..., native=True)`.
- `reglog` — `RegWrite` register-log format (`read_reglog` / `write_reglog`) and
  `frame_writes`, the shared per-frame SID-write framing loop.
- `oracle` — per-frame SID register grids: `register_grid` (jennings), the stdlib
  `grid_from_writes` framer, `read_sidwr` (`preframr-sidtrace` `.sidwr.bin`),
  `aligned_match`, and the `sidtrace` (sidplayfp) oracle bridge — `run_sidtrace`,
  `read_sidtrace`, `sidtrace_cadence`, `sidtrace_grid`.
- Caching — every download lands under one root, `.tunecache`
  (`$PYSID_TUNECACHE`; `default_cache(kind)` for `native` / `archives` /
  `disks` / `roms` / `csv`), CSDB requests are serialised machine-wide, and the
  `.github/actions/cache` composite action persists and pre-fetches it in CI
  ([docs/caching.md](docs/caching.md)).
- `testing` — HVSC tune fetch/resolve for test suites (`fetch_tune`,
  `resolve_tune`, `make_tune_fixtures`), SHA-256-pinned release downloads
  (`fetch_verified`, `fetch_archive_member` for one zip/tar member; a changed
  pin raises `ChecksumMismatch`), and
  `make_oracle_fixtures` / `oracle_grid` — a reusable byte-exact player-vs-`sidtrace`-oracle test
  ([docs/oracle-testing.md](docs/oracle-testing.md)); `assert_matches` /
  `assert_oracle_match` fail with the `whydiff` explanation.
- `whydiff` — why a `MemPlayer` render diverges: the player's traced Python lines
  (values read, branches, assignments) beside the jennings 6502 history, with each
  side's backward dataflow of the diverging value, and the state behind it chased to
  the frame it first parted (`python -m pysidtracker.whydiff`,
  [docs/whydiff.md](docs/whydiff.md)).
- `paths` / `faultloc` / `pathdiff` — per-frame code paths of two players (6502,
  a format's player or any `PySubject`, e.g. a lift) over a survey corpus, each
  divergence keyed on the branch where the executions first part, the minimum set
  of fixes, and strict xfail cases run with the `whydiff` explanation
  (`python -m pysidtracker.pathdiff build|rank|xfails`,
  [docs/pathdiff.md](docs/pathdiff.md)).
- `pathfuzz` — coverage gaps against the static path universes, data-byte mutants
  steered to unreached branch outcomes, and ddmin reproducers
  (`python -m pysidtracker.pathfuzz gaps|exercise|minimise`,
  [docs/pathfuzz.md](docs/pathfuzz.md)).
- `survey` — whole-song HVSC survey and failure triage of any format's renderer
  against the `sidtrace` oracle, verdict moves against a baseline (`compare`),
  an oracle-free decompile scope check (`scope`),
  and the same compare for single native files (`survey file`, bare PRGs
  wrapped); `sidid` — pure-Python SIDId with the bundled
  `sidid.cfg` (`pysidid` CLI). `pip install "pysidtracker[survey]"`,
  [docs/survey.md](docs/survey.md).
- `poweron` — the libsidplayfp power-on C64 RAM a PSID boots onto: `ram_pattern`
  (the `SystemRAMBank::reset` image, vagrant-moses' `c64.poweron_ram` as a
  mutable copy) and `power_on_ram` (it with the PSID driver's `$0000-$03FE`
  cleared, `PSID_DRIVER_CLEARED`), and `place_tune` (the tune loaded with the
  BASIC `LOAD` pointers in `$2D`-`$32`/`$AC`-`$AF`). `MemPlayer`, the `oracle`
  `register_grid`, and `irqhost` all start from them, so a tune reading
  uninitialised memory or its load bounds reads what sidplayfp leaves there.
- `loader` — the flat PSID/RSID loader for hosts with their own 64 KiB memory:
  `load_psid` (`(mem, load, init, play)`, honouring the in-body load address),
  `psid_image` (load bounds), `psid_songs`, `is_ntsc`, `is_rsid`, `psid_speed`
  and `speed_cia` — every one over the shared `parse_sid_header` / `SidImage`
  container parsing.
- `roms` — vagrant-moses' C64 ROM fetch, SHA-256 check and mapping (`fetch_rom`,
  `resolve_rom`, `rom_image`, `apply_roms`), cached beside the tunes. The ROMs
  are copyright works, never committed ([docs/c64-roms.md](docs/c64-roms.md)).
- `kernalhost` — `KernalHost` runs a whole C64 program with the real KERNAL
  file/keyboard calls serviced on a host directory, and `sys_addr` reads a BASIC
  `SYS` stub (vagrant-moses' `kernal`, raising `KernalHostError`).
- `d64` — release media through vagrant-moses' readers: `read_d64`, and
  `iter_prgs` over nested zip/gzip, d64 (35/40/42 tracks, error block), d71, d81,
  t64 and p00; `pysidtracker media extract SRC... -o DIR` writes every PRG with a
  `sha256 size $load path` index line.
- `fixtures` — CSDB (csdb.dk) native-format editor files (GoatTracker/DMCSID/JCH)
  are copyright works and never committed: `fetch_csdb` scrapes release pages and
  downloads every file into a gitignored, resume-safe cache indexed by SHA-256 in
  `SHA256SUMS.txt`, `index_media` pinning files found inside release disks and
  archives; `fixture(sha256, cache)` resolves a file by hash,
  `make_native_fixtures` is a pytest factory that resolves or **skips** when the
  cache is cold, and `load_manifest`/`write_manifest` are the committed
  hash-only manifest. `pysidtracker fixtures fetch <manifest>` (and the
  `pysid-fixtures` script) recreate the cache from it
  ([docs/native-fixtures.md](docs/native-fixtures.md)).
- `audio` — `render_player_wav` / `render_player_samples` drive a `MemPlayer`
  (or any `play_frame()` player) straight to WAV through an emulated SID at the
  render's clock, with optional tqdm progress (pyresidfp, a core dependency); the
  lower-level `render_wav` / `render_samples` take a bare write stream, plus
  `resolve_device` / `seconds_to_frames`.
- Writers — `SidHeader.to_bytes` / `write_psid` / `SidImage.to_prg` (the inverse
  of the parsers), `parse_prg`, and `encode_cstr` / `decode_cstr`.
- `SidImage` absolute-address accessors — `byte_at` / `word_at` / `contains` /
  `poke`, relocation-safe and bounds-checked.
- `mos6502` — shared 6502 primitives: `OP_LEN`, opcode-class sets, `s8` / `adc`
  / `sbc`, `SidWriteCapturingMemory`, `walk_until`.
- `MemPlayer` — per-frame register-diff player scaffold; subclasses implement
  `_init` / `_frame`; `MASK_PW_HI = True` nibble-masks the pulse-width-high
  registers in `snapshot` (as `EmuPlayer` does) to match the oracle grid.
  `register_writes_from_player` drives it to a reglog.
- `notefreq` — `NoteFreqTable`, computed PAL/NTSC pitch tables, and
  `locate_note_freq` (find hi/lo tables from paired absolute-indexed reads).
- `make_package_errors` — per-package error hierarchy factory whose `*ParseError`
  / `*FormatError` subclass the base ones, so base `except` clauses catch them.
- `disasm` — disassemble any range of a `.sid`/`.prg` image (on jennings, so
  the NMOS illegal opcodes render as real mnemonics), with the header's
  init/play entries labelled (`entry_labels`) and `image_after_init` the image
  state after the tune's init ran; backs the `pysidtracker disasm` command and
  replaces the per-repo disassembler scripts.
- `formats` / `maincli` — the one `pysidtracker` CLI (`info` / `reglog` / `wav`
  / `disasm`) that discovers installed formats through the `pysidtracker.formats`
  entry-point group and runs against each format's parser + player, plus any
  subcommands it contributes (a name several formats contribute, such as
  `export`, runs the command of the format recognising the tune). A format package registers a `SidFormat` instead
  of shipping its own CLI; its optional `grammar` / `streams` and `canonical`
  hooks back `pysidtracker grammar` and `pysidtracker equiv A B`
  ([docs/formats.md](docs/formats.md)); `find_format(data, name)` and `SidFormat.play(data,
  subtune)` play a tune's subtune (default its start song) on the installed
  format, as `survey --format` and `whydiff --format` do. `cli` holds the underlying argparse scaffold
  (`run_cli`, `print_info`, `add_reglog_command`, `add_wav_command`,
  `add_grammar_command`, `address`).
- `registers` additions — `SID_VOICES`, voice/global register indices, and the
  `attack_decay` / `sustain_release` ADSR nibble packers.
- `bytegrammar` — song byte streams as Lark grammars: `ByteGrammar`
  (`parse_prefix` / `parse_exact` / `tree`, pointer-table `entry` / `stored`
  reads, the `check` round trip and `listing`), `Parsed`, `select_rule`, the
  per-element `fits` codec round trip, and the
  shared `%import bytes (BYTE, _TAIL)` fragment, `BitmapPostLex` / `BitField`
  bitmap-selected fields; `SidImage.window` / `put_ptr`,
  `reraise_as`, `require_ram`, `load_prefix`, the `address` argparse type and
  `add_grammar_command` ([docs/bytegrammar.md](docs/bytegrammar.md)).
- `relocate` — `walk_code` (recursive-descent 6502 code walk) and
  `relocate_operands` (move, remap or refuse each absolute operand).
- `automaton` — program tables as functional graphs: `minimise`, `number_chains`
  and `layout_chains` ([docs/automaton.md](docs/automaton.md)).
- `editor` — `run_editor` / `editor_replay` run an editor's own LOAD decoder and
  live player on its RAM image; `trace_init_argument` / `init_arguments` trace
  the argument each subtune's init hands the player ([docs/editor.md](docs/editor.md)).
- `snapshot` — corpus digests of each tune's model and `SidFormat.probes` and
  their diff (`pysidtracker snapshot build|diff`), proving a refactor
  behaviour-preserving; `expected_errors` with `testing.corpus_params` /
  `check_outcome` turn one into exact-outcome corpus test cases.
- `wrap` — a tune's own play routine run around a native player tick:
  `measure_wrap` / `Wrap`, `Wrapped` (a `MemPlayer`), `count_entries`, with
  `codescan.jump_table` / `play_entries` and `emu.SidWriteLog` / `call_play`
  ([docs/wrap.md](docs/wrap.md)).
- `textgrammar` — typed dataclass models as text: `ModelTransformer`,
  `classes`, `term`, `seq`, `TextGrammar` and the `%import text` fragment;
  `pysidtracker grammardiff A.lark B.lark` compares two grammars structurally
  ([docs/textgrammar.md](docs/textgrammar.md)).
- `resolve_entry_points` / `is_jmp_vector`, `cadence_from_latch`, and the
  `ByteCursor` / `check` / `byte_range` parse helpers.

## Install

```bash
pip install pysidtracker   # core: jennings (init emulation), numpy, pyresidfp (WAV render), lark
```

All dependencies are core: jennings (detection runs a tune's 6502 init), pydexomizer
(`native_decrunch` unpacks exomizer-packed images natively), pyresidfp (the
emulated SID the WAV render drives), vagrant-moses (the C64 machine model: memory
map, banked bus, ROMs, the cycle-exact machine, the KERNAL host and media
readers), numpy (sample buffers and the image anchor scan), and lark (the
byte-stream grammar parsers).

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

Dead-symbol / doc-reference lint for dependents: [docs/lint.md](docs/lint.md).

See [docs/design.md](docs/design.md) for the detection model and shared
primitives.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
