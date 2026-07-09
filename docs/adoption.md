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
