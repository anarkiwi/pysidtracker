# C64 ROM fixtures: provenance and use

`pysidtracker.roms` fetches the Commodore 64 KERNAL, BASIC and CHARGEN ROMs into
the gitignored fixture cache and maps them into a 64 KiB image. Some tunes cannot
be executed without them: a PSID/RSID **BASIC tune** is a BASIC program the
interpreter has to run, and unclean rips call KERNAL routines from their init.

## Licensing

The C64 ROMs are **copyright Commodore**. They are **never committed** to this
repository in any form — no bytes, no base64, no test fixtures, no `dist/`.
They are fetched on demand into `.tunecache/roms/` (gitignored, `$PYSID_ROMCACHE`
or `$C64_ROMS` override it) and every fetch is verified by SHA-256 before it is
written; a mismatch raises `RomFetchError` and nothing is cached. This is the same
contract the HVSC `.sid` tune fixtures use.

## Source and checksums

Upstream is the **VICE emulator distribution**, which ships the ROM set, pinned to
an immutable tag of the official VICE git mirror:

```
https://raw.githubusercontent.com/VICE-Team/svn-mirror/r46192/vice/data/C64/<file>
```

`$C64_ROM_MIRROR` overrides the base URL (e.g. for a local VICE `data/C64`
directory served over HTTP, or an internal mirror).

| ROM | File | Size | Maps at | SHA-256 | MD5 |
|---|---|---|---|---|---|
| KERNAL rev 3 | `kernal-901227-03.bin` | 8192 | `$E000-$FFFF` | `83c60d47047d7beab8e5b7bf6f67f80daa088b7a6a27de0d7e016f6484042721` | `39065497630802346bce17963f13c092` |
| BASIC V2 | `basic-901226-01.bin` | 8192 | `$A000-$BFFF` | `89878cea0a268734696de11c4bae593eaaa506465d2029d619c0e0cbccdfa62d` | `57af4ae21d4b705c2991d98ed5c1f7b8` |
| CHARGEN | `chargen-901225-01.bin` | 4096 | `$D000-$DFFF` | `fd0d53b8480e86163ac98998976c72cc58d5dd8eb824ed7b829774e74213b420` | `12a4202f5331d45af846af6c58fba946` |

The KERNAL and BASIC MD5s match the published values in the CBM firmware archive's
`revisions.txt` (<https://www.zimmers.net/anonftp/pub/cbm/firmware/computers/c64/>),
which identify exactly these two part numbers — 901227-03 being the third and last
KERNAL revision and 901226-01 the only BASIC V2 revision. Contents were checked
structurally as well: KERNAL RESET `$FFFC` → `$FCE2`, NMI `$FFFA` → `$FE43`,
IRQ/BRK `$FFFE` → `$FF48`; BASIC cold start `$A000` → `$E394`, warm `$E37B`;
CHARGEN's first glyph is the standard `@`. The `roms`-marked test asserts these.

## `$D000`: I/O, not CHARGEN

`$D000-$DFFF` is the I/O bank *or* the character ROM depending on the `$01` CPU
port. `DEFAULT_ROMS` is therefore `("kernal", "basic")` — **CHARGEN is not mapped
by default**. A flat 64 KiB image can only hold one of the two, and a replay host
needs the I/O bank there: the SID is at `$D400`, the VIC at `$D000`, the CIAs at
`$DC00`. Overlaying CHARGEN would silently break every SID register access.

`apply_roms` writes the power-on CPU port (`$00 = $2F`, `$01 = $37`: BASIC +
KERNAL + I/O visible, CHARGEN hidden), which is exactly the banking the flat
mapping represents. Pass `chargen` explicitly (with `bank=None`) only to read
glyph data out of an image.

Two consequences of a flat image, both by design:

- there is no shadow RAM, so a tune that *writes* under a mapped ROM overwrites
  it instead of the RAM beneath;
- a tune that re-banks `$01` itself (`$35` pages both ROMs out, `$36` pages BASIC
  out) is no longer described by the mapping.

Either case needs a host that models the PLA, not a flat image.

## Use

```python
from pysidtracker import rom_image, apply_roms, fetch_rom, roms_available

mem = rom_image()                      # bytearray(0x10000), KERNAL + BASIC mapped
apply_roms(img.mem)                    # or overlay an existing SidImage
path = fetch_rom("kernal")             # just the cached, verified file
if not roms_available():               # gate a test that needs them
    ...
```

Test suites get the `c64_roms` pytest fixture from the `pysidtracker` pytest11
plugin with no conftest: `c64_roms("kernal")` for the path, `c64_roms.image()`
for a mapped image, `c64_roms.available()` to skip cleanly when offline and
uncached. The real-download test is marked `roms`, excluded from the default
matrix (`-m 'not roms'`) and run by its own CI job, matching how the `oracle` and
`exomizer` network tests are gated.

## Booting the ROMs

Mapped ROMs are not enough on their own: KERNAL routines expect the zero page and
`$0200-$03FF` a reset leaves behind. `KERNAL_BOOT` is the call sequence a host
`JSR`s, in order, before the tune is placed:

| Address | Routine | Sets up |
|---|---|---|
| `$FD15` | RESTOR | the KERNAL vectors at `$0314-$0333` |
| `$FDA3` | IOINIT | CIA/VIC registers |
| `$FF5B` | CINT | the screen editor and its zero page |

RAMTAS (`$FD50`) is deliberately **not** in the list: its RAM-size probe writes a
test pattern upward and relies on the write to `$A000` failing to stop. In a flat
image it does not fail, so RAMTAS would run on into the ROMs. Set its RAM-side
results directly instead — `$0281/$0282` = `$0800`, `$0283/$0284` = `$A000`,
`$0288` = `$04`, and zero `$0002-$0101` and `$0200-$03FF`.

For a BASIC tune add `BASIC_BOOT` — `$E453` (BASIC vectors) then `$E3BF` (BASIC
RAM: CHRGET, `TXTTAB`, `MEMSIZ`) — then place the program at `$0801`, set
`$2D/$2E` to its end (what the ROM's LOAD does), `JSR $A533` to rebuild the line
links, apply CLR's effects, and enter the interpreter at `BASIC_RUN` (`$A7B1`)
with the subtune number in `$030C`. This mirrors what libsidplayfp's `psiddrv`
does, minus its captured power-on RAM snapshot.
