# Interrupt host

`pysidtracker.irqhost` runs a PSID or RSID the way the
[sidtrace](https://github.com/anarkiwi/sidtrace) oracle's sidplayfp runs it:
the same power-on, the same PSID driver, every interrupt source the tune
programs live, and the CPU cycles the VIC-II steals. Its trace is the oracle's
trace, row for row and cycle for cycle, so a library can play a tune's real
interrupt structure (an NMI digi beside a music IRQ, a main-loop sample player
under badlines) byte-exact.

```python
from pysidtracker.irqhost import InterruptHost, InterruptPlayer, host_cadence

host = InterruptHost(open("tune.sid", "rb").read(), subtune=0)
grid = host.grid(500)          # 25-register rows, framed as the oracle grid is
rows = host.rows()             # SidtraceRow list: writes, raises, entries, rtis

player = InterruptPlayer(data)  # a MemPlayer: render_grid / play_frame
cadence = host_cadence(data)    # the measured play Cadence
```

## What sidplayfp does, and the host mirrors

- **Power-on.** `Player::initialise` resets the C64 and fires
  `3 * (powerOnDelay + 8000)` scheduler events with the CPU halted; sidtrace
  pins the delay to `0x1FFF`. Only the VIC, the four CIA timers and the two TOD
  clocks schedule events then, so the CPU's first cycle is a fixed cycle
  (`boot_cycle`: 339711 PAL, 315445 NTSC). The beam position at that cycle
  follows from the VIC's reset state.
- **Memory.** RAM powers up in libsidplayfp's pattern (4-byte runs of `$00` and
  `$FF`, flipped every 16K). `$0000`-`$03FE` is cleared; an RSID then gets
  libsidplayfp's `poweron.bin` image of a booted C64, fetched and SHA-256
  verified like the ROMs (`roms.poweron_writes`, `docs/c64-roms.md`), falling
  back to the KERNAL's RAM vectors offline. The tune's start/end are stored in
  `$2D`-`$32` and `$AC`-`$AF` as the BASIC loader leaves them.
- **Driver.** The PSID driver (`psiddrv.a65`) is assembled in its relocated page
  and run from reset: it acknowledges everything, sets `$D418`, the CIA #1
  KERNAL latch, `$D011`/`$D012`, arms the VIC (PSID speed bit 0) or CIA #1
  Timer A, banks in init and calls it. `irqjob`/`irqret` serve a PSID's play
  calls and an RSID's unclaimed IRQs.
- **ROMs.** sidplayfp has no ROMs: its KERNAL is NOPs with a few stubs (IRQ entry
  `$FF48`, NMI entry `$FE43`, `$EA31`/`$EA7E` acknowledging CIA #1, `$FE47`, the
  jump table as `RTS`). The host maps the same bytes, so a handler leaving
  through any KERNAL address runs the same cycles. BASIC and CHARGEN read as
  zeros.

## The model

The chips are ports of libsidplayfp's onto a two-phase event scheduler
(`eventsched`: PHI1 for the chips, PHI2 for the CPU, ties in scheduling order).

- **VIC-II** (`vicii`): raster IRQ edge detection (line 0 starts one cycle late),
  `$D019` acknowledge and `$D01A` mask, and BA: a badline (`raster & 7 ==
  YSCROLL`, lines `$30`-`$F7`, DEN seen on line `$30`) pulls BA low from line cycle
  11 to 54; `$D011` writes create or cancel a badline mid-line; sprite DMA pulls
  BA low around each enabled sprite's fetch, with Y-expansion and crunch.
- **CIA** (`cia`): both timers as VICE's `ciatimer` pipeline (start delay, force
  load, one-shot, Timer B counting Timer A), ICR masks and read-to-clear, the
  6526's interrupt one cycle after an underflow (sidplayfp's default CIA). CIA
  #1 drives the IRQ line with the VIC; CIA #2's interrupt is an NMI edge.
- **CPU** (`irqhost.C64`): jennings executes each instruction at its data access;
  around it the host follows libsidplayfp's `MOS6510` cycle tables. Each cycle
  is a read or a write; a read waits while BA is low (a write does not), so the
  VIC steals 40-43 cycles on a badline depending on the instruction. Interrupts
  follow the table's delay: one asserted before an instruction's last cycle is
  taken after it, else after the next one; a taken branch without a page cross
  defers one asserted during it; a stall counts once; `CLI`/`SEI`/`PLP`/`RTI`
  change I where the tables do; an NMI arriving by the push of the low PC
  byte hijacks an IRQ or `BRK`; a released IRQ can still be taken for two
  cycles. Writes to I/O land on their bus cycle, read-modify-write
  instructions writing the unmodified value first.
- **SID** writes are logged when the value changes (as sidtrace logs them);
  reads come from reSIDfp chips fed every write (`emu.SidReads`), so osc3, env3
  and the bus value of write-only registers match. Second and third SIDs map at
  their header addresses.

## Rows and framing

`rows()` are `SidtraceRow`s: a SID write at the cycle sidtrace stamps it, a
raise (`video_irq`, `cia_irq`, `nmi`) when a source asserts its line, an entry
(`*_entry`, `brk_entry`) when the CPU pushes the low PC byte, and `rti`. They
are compared with the oracle's with `host_check` (below).

`grid(n)` frames them exactly as the survey frames the oracle
(`oracle.sidtrace_grid` at the median raise gap of the play source), running
until the play source's raises cover `n` frames. `frames()` yields rows one
frame at a time with the source and frame length the oracle would pick over a
probe window; `InterruptPlayer` is the `MemPlayer` over both.

## Not modelled

TOD clocks and alarms, the serial port and the light pen; code executed from I/O
space (jennings prefetches operand bytes); the `SH*` instructions' value when RDY
drops during their throw-away read; CPU port bits 6-7 fall-off; RSID BASIC tunes
(rejected); the order of the CIA's cascade tick against a same-cycle RDY change.

## Validation

`tests/irqhost_tunes.py` holds synthetic tunes for each source and vector path:
CIA #1 KERNAL IRQ, raster IRQ through `$0314`, NMI digi through `$0318` with a
restarted one-shot CIA #1 Timer B, IRQ and NMI through `$FFFE`/`$FFFA` with the
KERNAL banked out, Timer B cascading with `BRK` and nested handlers, KERNAL NOP
slides, badline main loops, mid-screen YSCROLL and DEN changes, sprite DMA,
osc3 reads, NTSC, PSID VBI and CIA drivers, a second SID, banking, and `ANE`.
`tests/test_irqhost.py` compares the host with their recorded sidtrace traces,
and checks documented timing (timer periods, IRQ latency, badline and sprite
steals, raster lines); the `oracle`-marked `tests/test_irqhost_oracle.py` renders
them and real HVSC RSIDs under sidtrace and requires identical traces.

```bash
python -m pysidtracker.host_check run $HVSC tunes.csv --work .host-check --seconds 20 --jobs 6
python -m pysidtracker.host_check report .host-check/host_check.csv
python -m pysidtracker.host_check trace tune.sid 339711 340100   # instruction trace
```
