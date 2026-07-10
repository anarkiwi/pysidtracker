# Adoption guide: de-duplicating the parser fingerprints

The 0.2.0 primitives (`codescan`, `registers`, `trace_init`) generalise the
relocation-invariant recognizer/operand-capture code each `py*` parser hand-rolled
during HVSC validation. This guide maps each parser's bespoke code to the
pysidtracker primitive that replaces it, so the follow-up de-duplication in the
parsers is mechanical. **No parser repo is edited here** — this is the plan.

Common import in every parser after adoption:

```python
from pysidtracker import CodePattern, find_code_all, find_code_first
from pysidtracker import registers as reg
from pysidtracker import find_register_stores, trace_init
```

## pyfuturecomposer — `src/pyfuturecomposer/reader.py`

| Bespoke code | Replace with |
| --- | --- |
| `_FC_SIGNATURES` (`tuple(None if "??" … )` token lists) | `CodePattern(spec)` per signature — the spec strings are the **same** token text already in the file (literals + `??`). |
| `_match_at(data, pattern, i)` | drop — `CodePattern` literal matching. |
| `_find_signature(data, pattern)` | `find_code_first(image, pat)` (returns `Match.addr`; no separate `image.load + offset`). |
| `recognize()` loop over signatures | `for pat in _FC_PATS: m = find_code_first(image, pat)` → `m.addr`. |

FC captures nothing (recognition only), so the specs use only literals and `??`.

## pymusicassembler — `src/pymusicassembler/reader.py`

| Bespoke code | Replace with |
| --- | --- |
| `_SIG_ORDER`/`_SIG_PATTERN`/`_SIG_INSTR`/`_SIG_FREQ` (`re.compile(rb"\xbd(..)…")`) | `CodePattern` specs, e.g. `"BD {order_lo:w} 85 FA BD {order_hi:w} 85 FB"`. The `(..)` groups become `{name:w}` word captures. |
| `_w16(match.group(n))` | gone — `Match.captures["name"]` is already the LE word. |
| `_match_signatures` / `_find_freq` `.search`/`.finditer` | `find_code_first` / `find_code_all`. The freq `+3` stride check stays as a caller-side filter over `find_code_all` matches. |
| `_discover_bases` operand reads | read from `Match.captures`. |

## pysidwizard — `src/pysidwizard/sidreader.py`

| Bespoke code | Replace with |
| --- | --- |
| `data.find(SWP_MAGIC/SWM_MAGIC, base)` | keep `image.find` for the plain magic bytes (no wildcards needed). |
| `recognize()` magic search | `image.find` is fine; if a wildcarded player-stub signature is added for the relocated cohort, express it as a `CodePattern`. |
| relocated/loader layout (currently unsupported → error) | `trace_init` to recover the real base/vector for the relocated export tail the reader bails on. |

pysidwizard is mostly fixed-offset magic; its adoption is the smallest (the
signature-site + relocated-base idiom is the candidate, not the magic search).

## pydefmon — `pydefmon/_sid_format.py`

| Bespoke code | Replace with |
| --- | --- |
| `_SIG_TOKENS` / `SIGNATURE` (`tuple(None if "??" …)`) | `CodePattern(_SIG_TOKENS_STR)` — identical token text. |
| `find_signature(mem, start, end)` (hand-rolled masked scan loop) | `find_code_first(image, pat, start=…, end=…).addr`. |
| `is_defmon_replay(image)` | `find_code_first(image, pat) is not None`. |
| `_DATA_BASE_FROM_SIGNATURE` delta from signature site | unchanged — `Match.addr + delta` (signature-site → relocated-base derivation). |

## pysoundmonitor — `src/pysoundmonitor/reader.py` + `constants.py`

| Bespoke code | Replace with |
| --- | --- |
| `STA_DC04`/`STA_DC05`/`CMP_06` byte consts + `find_cia_fingerprint` windowed search | `CodePattern("8D 04 DC …")` for the contiguous part; the CIA store targets are `reg.CIA1_TIMER_A_LO`/`_HI`. The register addresses come from `pysidtracker.registers` instead of local byte literals. |
| `_operand_targets(image, opcode)` (numpy operand harvest) | `find_register_stores` for stores to known addrs; for the general `LDA abs,X` table-operand harvest, `find_code_all(image, "BD {t:w}")` giving every `Match.captures["t"]`. |
| `LDA_ABSX`/`CPX_ABS`/`LDX_ABS` opcode consts | keep as CodePattern literals (`BD`/`EC`/`AE`); the store-opcode set is `reg.STORE_ABS_OPCODES`. |
| CIA-latch cadence discovery | `trace_init(image).cia1_timer_latch` gives the play period directly for the relocating cohort that installs `$0314`/`$0315` and latches the CIA timer in init. |

Soundmonitor is the headline beneficiary of `trace_init`: its CIA-timed builds
program the play cadence and IRQ handler from inside init, which the tracer reads
back as `cia1_timer_latch` + `irq_vector` without re-deriving them from operands.

## pyjch — `src/pyjch/reader.py`

| Bespoke code | Replace with |
| --- | --- |
| `_find_operands(image, prefix, suffix, oplen)` | `find_code_all(image, f"{prefix} {{op}} {suffix}")` — `prefix<operand>suffix` **is** a masked pattern with one capture. |
| `_one_operand(...)` | `find_code_first(...)`; raise on `None` as before. |
| `_AD_SUF`/`_SR_SUF`/`_GATE_SUF`/`_LDA_IMM`/`_LDA_ABSY` byte consts | inline into specs: `"A9 {ad} 8D 05 D4"`, `"A9 {sr} 8D 06 D4"`, `"A9 {gate} 99 04 D4"`, `"B9 {ptr:w} 85 FB"`. The `$D405/$D406` targets are `reg.SID_*`. |
| `_RECOGNIZE_SIG` (`b"\x8d\x05\xd4\x8d\x0c\xd4"`) | `CodePattern("8D 05 D4 8D 0C D4")` or `find_register_stores(image, [reg.sid_register(0xD405), 0xD40C])`. |

## pydmcsid — `src/pydmcsid/reader.py`

| Bespoke code | Replace with |
| --- | --- |
| `find_dmc_base` JMP-table scan (`mem[base]==0x4C …`, relative-target check) | partial: the `4C ?? ??` opcode gate is a `CodePattern("4C {t:w}")`; the 4-entry base-relative offset check is a structural constraint that stays caller-side over the captured targets. Not a pure skeleton match, so only the `JMP`-operand capture is lifted. |

## Summary of which primitive replaces what

- **`CodePattern` + `find_code_*`** replaces: pyfuturecomposer `_FC_SIGNATURES`/
  `_match_at`/`_find_signature`; pymusicassembler `_SIG_*` regexes + `_w16`;
  pydefmon `SIGNATURE`/`find_signature`; pyjch `_find_operands`/`_one_operand` +
  idiom byte consts; pysoundmonitor `_operand_targets` (table-operand harvest);
  pydmcsid JMP-operand capture.
- **`registers`** replaces: pysoundmonitor `STA_DC04`/`STA_DC05`/`LDA_ABSX`/
  `CPX_ABS`/`LDX_ABS` and the store-opcode literals; pyjch `$D405`/`$D406`/`$D40C`
  and `_GATE_SUF` targets; any per-parser hardware-address literals.
- **`trace_init` / `InitTrace`** replaces: bespoke cadence/vector derivation for
  the IRQ-driven, relocating cohorts — pysoundmonitor's CIA-timed builds (latch +
  `$0314`/`$0315` handler) and pysidwizard's relocated-export tail the readers
  currently cannot locate statically.

## 0.3.0 shared player/validation surfaces

The 0.3.0 surfaces consolidate the register-log / oracle / tune-fetch / audio
code the format packages hand-copied. Each parser's old copy maps to the new
shared module as below. **No parser repo is edited here** — this is the plan.

Common import after adoption:

```python
from pysidtracker import RegWrite, read_reglog, write_reglog, frame_writes
from pysidtracker import register_grid, grid_from_writes, read_sidwr, aligned_match
from pysidtracker import fetch_tune, resolve_tune, make_tune_fixtures
from pysidtracker import render_samples, render_wav
from pysidtracker import PAL_CYCLES_PER_FRAME, PAL_CLOCK_HZ, PW_HI_REGS
```

### `reglog`

| Parser's old copy | Replace with |
| --- | --- |
| `pygoattracker/src/pygoattracker/reglog.py` (`RegWrite`, `read_reglog`, `write_reglog`) | `pysidtracker.reglog` (identical text format). |
| `pyjch/src/pyjch/reglog.py`, `pydefmon/pydefmon/reglog.py`, `pymusicassembler` / `pyfuturecomposer` `reglog.py` | same. |
| each package's `iter_register_writes` framing body (`enumerate(iter_frames(...))`, `clock + offset*spacing`, the `write_spacing * SID_REGISTERS >= cycles_per_frame` guard) | `frame_writes(per_frame_iter, cycles_per_frame=..., write_spacing=...)`. Keep the package's player driver; feed its per-frame `(reg, val)` iterables in. defMON's absolute-`$D400..$D418` writes keep `sid_reg_base=0xD400`; players already yielding `0..24` pass `sid_reg_base=0`. |
| per-package `REGLOG_HEADER` / `DEFAULT_WRITE_SPACING` | `pysidtracker.reglog` constants (the header string carries the `pysidtracker` name). |
| `read_reglog` raising each package's own error (or `ValueError` in pydefmon) | now raises `SidParseError` uniformly. |

### `oracle`

| Parser's old copy | Replace with |
| --- | --- |
| `pydefmon/tests/_support/py65_oracle.py` `Oracle.grid` + `_patch_illegals` | `register_grid(image, nframes, illegal_opcodes=True)`. |
| `pyjch/tests/_v20oracle.py` `Oracle.grid` / `oracle_grid` | `register_grid(image, nframes)`. |
| `pyjch` / `pymusicassembler` / `pyfuturecomposer` `tests/conftest.py` `_grid_from_writes` / `grid_from_writes` | `grid_from_writes(writes)`. |
| `pydmcsid/tests/helpers.py` `grid_from_writes` | same. |
| `_read_sidwr` (`struct.Struct("<qHBB")`) in those conftests | `read_sidwr(path)`. |
| `pyjch` conftest `aligned_match` | `aligned_match(oracle, rendered)` (now returns `bool`; the old `(ok, lead, divergence)` tuple is dropped — callers that reported divergence keep their own comparison). |

### `testing`

| Parser's old copy | Replace with |
| --- | --- |
| `pygoattracker` / `pydmcsid` (and peers) `scripts/fetch_tunes.py` `fetch` + `_download` + `_is_sid` + mirror/retry logic | `fetch_tune(relpath, cache_dir=..., mirror=..., retries=...)`. |
| conftest `_try_resolve` (`$HVSC` local → cache → fetch) | `resolve_tune(relpath, cache_dir=..., local_env="HVSC")`. |
| conftest `tune_id` / `tune_path` parametrized fixtures | `tune_id, tune_path = make_tune_fixtures(TUNES, CACHE)`. Keep each package's `TUNES` mapping. |

`scripts/fetch_tunes.py` keeps only its package-specific `TUNES` table + CLI,
delegating the fetch to `pysidtracker.testing`.

### `audio`

| Parser's old copy | Replace with |
| --- | --- |
| `pygoattracker` / `pyfuturecomposer` / `pymusicassembler` `src/*/audio.py` (`_default_device`, `render_samples`, `write_wav`, `render_wav`) | `pysidtracker.audio` (behind the `audio` extra). Pass the package player's per-frame `(reg, val)` iterables as `frame_iter`; supply `cycles_per_frame` / `clock_frequency` from `registers`. |
| per-package `CHIP_MODELS`, pyresidfp import guard raising the format error | `pysidtracker.audio.CHIP_MODELS`; a missing extra raises `AudioUnavailable`. |
| the private `_default_device` import each `audio.py` copied | the public `default_device(model, sampling_frequency, clock_frequency)`; read a device's rate with `device_sampling_frequency(device)`. `_default_device` stays as a back-compat alias. |

## 0.4.0 cadence + native decrunch

| Parser's old code | Replace with |
| --- | --- |
| per-format play-cadence **constants** (e.g. pydefmon's hard-coded CIA period `23546`, per-format PAL/NTSC frame constants keyed off the header) | `playroutine_cadence(image_or_bytes, clock=None)` — derives the cadence from what init actually programs (video frame vs CIA-timer latch), returning a `Cadence(cycles_per_call, source, clock_hz, latch, dynamic)`. defMON's `23546` = a `23545` CIA latch + 1; the latch is the tune's tempo, so it is derived, never assumed. |
| bespoke "is this tune CIA-timed?" checks reading the header speed bits | `playroutine_cadence(...).source is TriggerSource.CIA_TIMER` (headers are only a hint; the trigger is read from init). |
| emulated-init unpacking as the *only* depack path for exomizer-packed tunes | `native_decrunch(image_or_bytes)` (native pydexomizer decrunch, no init run) as an init-free alternative; `detect_playroutine(..., native=True)` wires it in as an opt-in first try with the emulated-init fallback intact. |

Per-format cadence constants (defMON's 23546 and friends) are **replaced** by
`playroutine_cadence`; a dependent that pinned a constant should call the
derivation and read `cadence.cycles_per_call`.
