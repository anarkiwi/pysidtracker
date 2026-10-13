# Play cadence

`playroutine_cadence(data, *, clock=None, frames=8, subtune=0, max_cycles=8_000_000)`
returns a `Cadence`: how often the play routine is called, derived from how the
tune's interrupts are actually programmed. The SID header's speed bit is never
read as the answer. It only selects what sidplayfp's PSID driver arms before
init, and anything init or play programs afterwards overrides it. The header
clock bits only pick PAL or NTSC (unless `clock` is given).

## What sidplayfp's PSID driver does

libsidplayfp (`src/psiddrv.a65`, `src/psiddrv.cpp`) relocates a small driver
into the first free page from `$04` up that is outside the tune and
`$A0`–`$BF`. On reset the CPU jumps straight to the driver, so the KERNAL
initialisation never runs and every CIA register keeps its reset value
(timers stopped, latches `$FFFF`, all interrupts masked). The driver then:

1. masks the VIC IRQ (`$D01A=0`), acknowledges `$D019`, writes `$7F` to both
   CIA ICRs and reads them;
2. sets `$D418=$0F`;
3. loads CIA #1 Timer A's latch with the KERNAL 60 Hz value (`$4025` PAL,
   `$4295` NTSC);
4. sets the raster compare to line 311 for an RSID whose clock matches the
   machine, else line 0;
5. for a PSID with a play address, points `$0314` at its `irqjob`;
6. arms one interrupt source. A PSID song whose speed bit is 0 gets the raster
   IRQ (`$D01A=$81`). An RSID (speed always treated as CIA), or a PSID song
   whose bit is 1, gets CIA #1 Timer A in continuous mode with its IRQ enabled
   (`$DC0D=$81`, `$DC0E=$01`);
7. sets `$01` (`$37`, or the bank holding init for a PSID) and calls init with
   the subtune in A. The I flag is set for a PSID and clear for an RSID;
8. for a PSID, sets `$01=$37` when there is a play address, clears I and idles.
   An RSID's driver just idles.

`irqjob` saves `$01`, calls play, acknowledges the VIC (`DEC $D019`) and, as
`irqret`, reads `$DC0D` and returns. An RSID's `$0314` points at `irqret`, so
an interrupt the tune does not claim plays nothing. The vectors `$0316`/`$0318`
hold the driver's halt for a PSID and the KERNAL defaults for an RSID.

## Model

`playroutine_cadence` writes that driver state through the same register
semantics the tune's own writes go through, and installs the driver's `irqjob`,
`irqret` and idle loop as 6502 code in the driver's page. It then runs init on
jennings and raises and takes interrupts as the C64 does:

- **VIC raster:** a raise at the compare line each frame. The line is asserted
  until it is acknowledged by a write of bit 0 to `$D019` (read-modify-write
  instructions included). `$D019` reads give the flag and the IRQ bit.
- **CIA timers:** each CIA's Timer A and B latches, control registers
  (start, one-shot, force load, Timer B counting Timer A) and ICR mask. A
  running timer underflows every `latch + 1` cycles and stops after one
  underflow in one-shot mode. The ICR flags assert the IRQ (CIA #1) or NMI
  (CIA #2) while unmasked and unread. Reads of the timer counters, the
  control registers and the ICR give the modelled values.
- **Dispatch:** an IRQ is taken when the line is asserted and I is clear. An
  NMI is taken on each new edge. With the KERNAL banked in (`$01` bit 1) the
  CPU goes through `$0314` (after the KERNAL's `A`/`X`/`Y` push) or `$0318`,
  else through `$FFFE`/`$FFFA` in RAM. Code that leaves a handler into KERNAL
  ROM (e.g. `JMP $EA31`) gets the KERNAL's register pull and `RTI`.

Every SID write is attributed to the most recent raise, as the sidtrace
oracle's rows are. The play source is the one whose raises drive the most
distinct SID registers, then the one with the most writing raises, so a
sample player that drives only `$D418` at audio rate does not outvote the
music routine.
Init is run until it returns or idles (a `JMP *`), then `frames` more video
frames. If it never does, the run stops at `max_cycles`. The last `frames - 1`
frames are measured:

- a raster source gives `calls_per_frame` (its raises per frame) and
  `raster_lines` (the compare lines that fired). A play routine chaining `$D012`
  through k lines is k calls per frame, however unevenly they are spaced;
- a continuous timer gives `cycles_per_call = latch + 1`; a one-shot timer that
  play restarts gives the mean raise interval. `dynamic` is set when the
  period changed while it was measured;
- if no interrupt was followed by SID writes, `source` is `None` and
  `cycles_per_call` is one video frame.

`TriggerSource` is `PAL_VIDEO`, `NTSC_VIDEO`, `CIA_TIMER` (CIA #1) or
`NMI_TIMER` (CIA #2). `Cadence.calls_per_second` is
`calls_per_frame * clock_hz / frame_cycles`.

## Validation against sidplayfp

`python -m pysidtracker.cadence_check` compares any `module:function` cadence
(default `pysidtracker.cadence:playroutine_cadence`) with the sidtrace oracle's
interrupt rows for subtune 1 of every distinct tune in one or more survey
`tunes.csv` files:

```bash
python -m pysidtracker.cadence_check run $HVSC */.hvsc-survey/tunes.csv --work .cadence-check --jobs 24
python -m pysidtracker.cadence_check report .cadence-check/cadence.csv
```

The measured source is picked from the trace's SID writes by the same rule
(each write attributed to the latest raise; most distinct registers, then most
writes). Its rate is taken over its raises
after the first, which init can hold back. A row agrees when the source matches
and the predicted calls over the measured span are within one frame's calls of
the count. Verdicts: `agree`, `rate`, `source`, `unknown` (no predicted source),
`no_raises`, `oracle_error`, `cadence_error`.
