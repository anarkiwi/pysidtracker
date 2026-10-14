"""Synthetic 6502 tunes exercising each interrupt source, vector path and VIC DMA.

Functional code written for the tests (no tune bytes); ``build`` wraps one in a
PSID/RSID container. ``python -m tests.irqhost_tunes`` re-records the fixtures.
"""

import random
import sys
from pathlib import Path

from pysidtracker.asm6502 import assemble

from .helpers import build_psid_embedded_load

LOAD = 0x1000
FIXTURES = Path(__file__).parent / "data" / "irqhost"
FIXTURE_CYCLES = 100000

TUNES = {
    "init_writes": (
        """
        init: LDA #$01
          STA $D400
          LDA #$02
          STA $D401
          RTS
""",
        {},
    ),
    "badline_loop": (
        """
        ; main loop counter into $D418 under the default screen (badlines on), CIA1 KERNAL IRQ
        init: LDX #$00
        loop: INX
          STX $D418
          JMP loop
""",
        {},
    ),
    "raster_kernal": (
        """
        ; raster IRQ via $0314 at line $80, handler bumps $D400 and acks with INC $D019, exits $EA31;
        ; main loop counts into $D418
        init: SEI
          LDA #<irq
          STA $0314
          LDA #>irq
          STA $0315
          LDA #$7F
          STA $DC0D
          LDA $DC0D
          LDA #$01
          STA $D01A
          LDA #$1B
          STA $D011
          LDA #$80
          STA $D012
          CLI
          LDX #$00
        loop: INX
          STX $D418
          JMP loop
        irq: INC $D019
          INC cnt
          LDA cnt
          STA $D400
          JMP $EA31
        cnt: .byte 0
""",
        {},
    ),
    "nmi_digi_oneshot": (
        """
        ; CIA2 TA NMI digi via $0318 (KERNAL), CIA1 TB IRQ in one-shot restarted, main loop
        init: SEI
          LDA #<nmi
          STA $0318
          LDA #>nmi
          STA $0319
          LDA #<irq
          STA $0314
          LDA #>irq
          STA $0315
          LDA #$7F
          STA $DC0D
          LDA #$64
          STA $DD04
          LDA #$00
          STA $DD05
          LDA #$81
          STA $DD0D
          LDA #$11
          STA $DD0E
          LDA #$00
          STA $DC06
          LDA #$30
          STA $DC07
          LDA #$82
          STA $DC0D
          LDA #$19
          STA $DC0F
          CLI
        loop: INC $D415
          JMP loop
        nmi: PHA
          INC ncnt
          LDA ncnt
          STA $D418
          LDA $DD0D
          PLA
          RTI
        irq: LDA $DC0D
          INC icnt
          LDA icnt
          STA $D401
          LDA #$19
          STA $DC0F
          JMP $EA81
        ncnt: .byte 0
        icnt: .byte 0
""",
        {},
    ),
    "sprite_dma": (
        """
        ; sprites 0,3,7 enabled at varied Y (one expanded), main loop counter into $D418
        init: LDA #$89
          STA $D015
          LDA #$60
          STA $D001
          LDA #$64
          STA $D007
          LDA #$F0
          STA $D00F
          LDA #$08
          STA $D017
          LDX #$00
        loop: INX
          STX $D418
          JMP loop
""",
        {},
    ),
    "vectors_in_ram": (
        """
        ; KERNAL banked out: NMI via $FFFA and IRQ via $FFFE with RTI; CIA2 TB NMI, raster IRQ
        init: SEI
          LDA #$35
          STA $01
          LDA #<nmi
          STA $FFFA
          LDA #>nmi
          STA $FFFB
          LDA #<irq
          STA $FFFE
          LDA #>irq
          STA $FFFF
          LDA #$7F
          STA $DC0D
          LDA #$01
          STA $D01A
          LDA #$32
          STA $D012
          LDA #$95
          STA $DD06
          LDA #$01
          STA $DD07
          LDA #$82
          STA $DD0D
          LDA #$11
          STA $DD0F
          CLI
        loop: LDA $DC04
          STA $D416
          JMP loop
        nmi: PHA
          LDA $DD0D
          INC $D418
          PLA
          RTI
        irq: PHA
          LSR $D019
          LDA $D012
          STA $D400
          CLC
          ADC #$40
          STA $D012
          PLA
          RTI
""",
        {},
    ),
    "yscroll_den": (
        """
        ; YSCROLL rewritten every frame from a raster IRQ, DEN toggled; main loop
        init: SEI
          LDA #<irq
          STA $0314
          LDA #>irq
          STA $0315
          LDA #$7F
          STA $DC0D
          LDA #$01
          STA $D01A
          LDA #$40
          STA $D012
          CLI
          LDX #$00
        loop: INX
          STX $D418
          JMP loop
        irq: LDA $D019
          STA $D019
          INC cnt
          LDA cnt
          AND #$17
          ORA #$08
          STA $D011
          STA $D405
          JMP $EA81
        cnt: .byte 0
""",
        {},
    ),
    "cascade_brk_nested": (
        """
        ; CIA1 TB cascading TA underflows, force load, timer reads; BRK in main; CLI inside IRQ
        init: SEI
          LDA #<irq
          STA $0314
          LDA #>irq
          STA $0315
          LDA #<brk
          STA $0316
          LDA #>brk
          STA $0317
          LDA #$10
          STA $DC04
          LDA #$01
          STA $DC05
          LDA #$20
          STA $DC06
          LDA #$00
          STA $DC07
          LDA #$7F
          STA $DC0D
          LDA #$82
          STA $DC0D
          LDA #$51
          STA $DC0F
          LDA #$11
          STA $DC0E
          CLI
        loop: LDA $DC06
          STA $D402
          BRK
          NOP
          JMP loop
        brk: INC $D403
          JMP $EA81
        irq: LDA $DC0D
          CLI
          INC cnt
          LDA cnt
          STA $D404
          LDX #$40
        w: DEX
          BNE w
          JMP $EA81
        cnt: .byte 0
""",
        {},
    ),
    "latency_stress": (
        """
        ; interrupt-latency stress: fast NMI, raster + CIA IRQs, main loop of SEI/CLI/PLP/branches
        init: SEI
          LDA #<nmi
          STA $0318
          LDA #>nmi
          STA $0319
          LDA #<irq
          STA $0314
          LDA #>irq
          STA $0315
          LDA #$47
          STA $DD04
          LDA #$00
          STA $DD05
          LDA #$81
          STA $DD0D
          LDA #$11
          STA $DD0E
          LDA #$01
          STA $D01A
          LDA #$90
          STA $D012
          LDA #$81
          STA $DC0D
          LDA #$3F
          STA $DC04
          LDA #$02
          STA $DC05
          LDA #$11
          STA $DC0E
          CLI
        loop: INX
          STX $D400
          SEI
          NOP
          CLI
          PHP
          PLP
          TXA
          AND #$03
          BEQ skip
          BNE skip
        skip: BIT $D012
          BMI loop
          INY
          STY $D401
          JMP loop
        nmi: PHA
          INC $D418
          LDA $DD0D
          PLA
          RTI
        irq: LDA $D019
          STA $D019
          LDA $DC0D
          INC $D402
          JMP $EA81
""",
        {},
    ),
    "osc3_reads": (
        """
        ; osc3 noise reads stored to $D418, voice 3 noise running
        init: LDA #$FF
          STA $D40E
          STA $D40F
          LDA #$81
          STA $D412
        loop: LDA $D41B
          STA $D418
          LDA $D41C
          STA $D417
          JMP loop
""",
        {},
    ),
    "ntsc_raster": (
        """
        ; NTSC: raster IRQ via $0314 at line 0, main loop
        init: SEI
          LDA #<irq
          STA $0314
          LDA #>irq
          STA $0315
          LDA #$7F
          STA $DC0D
          LDA #$01
          STA $D01A
          LDA #$1B
          STA $D011
          LDA #$00
          STA $D012
          CLI
        loop: INX
          STX $D418
          JMP loop
        irq: DEC $D019
          INC $D400
          JMP $EA31
""",
        {"flags": 0x08},
    ),
    "psid_vbi": (
        """
        init: LDA #$0F
          STA $D418
          RTS
        play: INC $D400
          LDA $D012
          STA $D401
          RTS
""",
        {"magic": b"PSID", "play": "play"},
    ),
    "psid_cia": (
        """
        init: LDA #$0F
          STA $D418
          LDA #$00
          STA $DC04
          LDA #$30
          STA $DC05
          RTS
        play: INC $D400
          LDA $DC04
          STA $D401
          RTS
""",
        {"magic": b"PSID", "play": "play", "speed": 1},
    ),
    "second_sid": (
        """
        init: LDA #$0F
          STA $D418
          STA $D438
          RTS
        play: INC $D400
          INC $D420
          INC $D43F
          RTS
""",
        {"magic": b"PSID", "play": "play", "version": 3, "second_sid": 0x42},
    ),
    "badline_tricks": (
        """
        ; YSCROLL and $D017 rewritten every pass with four sprites on, main loop only
        init: LDA #$0F
          STA $D015
          LDA #$40
          STA $D001
          STA $D003
          LDA #$90
          STA $D005
          STA $D007
        loop: INX
          TXA
          AND #$07
          ORA #$18
          STA $D011
          STX $D017
          STX $D418
          JMP loop
""",
        {},
    ),
    "kernal_slides": (
        """
        ; raster IRQ leaving via a NOP slide into $EA7E; CIA2 NMI on the KERNAL's $FE47
        init: SEI
          LDA #<irq
          STA $0314
          LDA #>irq
          STA $0315
          LDA #$7F
          STA $DC0D
          LDA #$01
          STA $D01A
          LDA #$A0
          STA $D012
          LDA #$00
          STA $DD04
          LDA #$10
          STA $DD05
          LDA #$81
          STA $DD0D
          LDA #$01
          STA $DD0E
          CLI
        loop: INY
          STY $D418
          JMP loop
        irq: ASL $D019
          INC $D400
          JMP $EA34
""",
        {},
    ),
    "banking": (
        """
        ; RAM under the ROMs and I/O read back through the CPU port
        init: SEI
          LDA #$34
          STA $01
          LDA #$5A
          STA $D400
          STA $E000
          LDA #$35
          STA $01
          LDA $E000
          STA $D401
          LDA #$30
          STA $01
          LDA $D400
          LDX #$37
          STX $01
          STA $D402
          LDA $E000
          STA $D403
          RTS
""",
        {},
    ),
    "badline_midline": (
        """
        ; no sprites: YSCROLL rewritten at drifting line cycles, badlines made mid-line
        init: SEI
          LDA #$7F
          STA $DC0D
        loop: INX
          TXA
          AND #$07
          ORA #$18
          STA $D011
          STX $D418
          NOP
          JMP loop
""",
        {},
    ),
    "illegal_ane": (
        """
        ; ANE ($8B) and its magic constant, flags pushed to the SID
        init: LDX #$00
        loop: INX
          TXA
          .byte $8B, $3C
          STA $D400
          PHP
          PLA
          STA $D401
          LDA #$00
          .byte $8B, $FF
          STA $D402
          JMP loop
""",
        {},
    ),
}


VIC_FUZZ_SEEDS = (0, 1, 2, 3)
CIA_FUZZ_SEEDS = (0, 1, 2, 3)


def fuzz_vic(seed: int) -> str:
    """A main loop hammering VIC registers at drifting cycles, raster IRQ on."""
    rnd = random.Random(seed)
    regs = [0x11, 0x11, 0x11, 0x15, 0x17, 0x01, 0x03, 0x0F, 0x12]
    body = []
    for _ in range(rnd.randint(4, 12)):
        r = rnd.choice(regs)
        if r == 0x11:
            body.append(
                f"  LDA #${rnd.choice([0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F,0x0B,0x13,0x3B,0x9B]):02X}"
            )
        elif r == 0x12:
            body.append(f"  LDA #${rnd.randint(0,255):02X}")
        else:
            body.append(
                f"  TXA" if rnd.random() < 0.5 else f"  LDA #${rnd.randint(0,255):02X}"
            )
        body.append(f"  STA $D0{r:02X}")
        for _ in range(rnd.randint(0, 3)):
            body.append(
                rnd.choice(
                    ["  NOP", "  INY", "  BIT $10", "  LDA $D012", "  STY $D400"]
                )
            )
    return "\n".join(
        [
            "init: SEI",
            "  LDA #<irq",
            "  STA $0314",
            "  LDA #>irq",
            "  STA $0315",
            "  LDA #$7F",
            "  STA $DC0D",
            "  LDA #$01",
            "  STA $D01A",
            "  CLI",
            "loop: INX",
            "  STX $D418",
            *body,
            "  JMP loop",
            "irq: INC $D019",
            "  INC cnt",
            "  LDA cnt",
            "  STA $D401",
            "  JMP $EA81",
            "cnt: .byte 0",
        ]
    )


def fuzz_cia(seed: int) -> str:
    """A main loop hammering both CIAs' timers, controls and ICRs, IRQ and NMI on."""
    rnd = random.Random(seed)
    body = []
    for _ in range(rnd.randint(4, 14)):
        base = rnd.choice([0xDC00, 0xDD00])
        reg = rnd.choice([4, 5, 6, 7, 0x0D, 0x0D, 0x0E, 0x0F, 0x0E, 0x0F])
        if rnd.random() < 0.3:
            body.append(f"  LDA ${base + reg:04X}")
            body.append("  STA $D402")
            continue
        if reg in (0x0E, 0x0F):
            val = rnd.choice(
                [0x01, 0x11, 0x09, 0x19, 0x00, 0x10, 0x41, 0x51, 0x03, 0x07]
            )
        elif reg == 0x0D:
            val = rnd.choice([0x81, 0x82, 0x83, 0x7F, 0x01, 0x02])
        else:
            val = rnd.randint(0, 255) if rnd.random() < 0.5 else rnd.randint(0, 3)
        body.append(f"  LDA #${val:02X}")
        body.append(f"  STA ${base + reg:04X}")
        for _ in range(rnd.randint(0, 2)):
            body.append(rnd.choice(["  NOP", "  INY", "  SEI", "  CLI", "  STY $D400"]))
    return "\n".join(
        [
            "init: SEI",
            "  LDA #<irq",
            "  STA $0314",
            "  LDA #>irq",
            "  STA $0315",
            "  LDA #<nmi",
            "  STA $0318",
            "  LDA #>nmi",
            "  STA $0319",
            "  CLI",
            "loop: INX",
            "  STX $D418",
            *body,
            "  JMP loop",
            "irq: LDA $DC0D",
            "  STA $D401",
            "  JMP $EA81",
            "nmi: PHA",
            "  LDA $DD0D",
            "  STA $D403",
            "  PLA",
            "  RTI",
        ]
    )


for _seed in VIC_FUZZ_SEEDS:
    TUNES[f"fuzz_vic_{_seed}"] = (fuzz_vic(_seed), {})
for _seed in CIA_FUZZ_SEEDS:
    TUNES[f"fuzz_cia_{_seed}"] = (fuzz_cia(_seed), {})


def build(name: str) -> bytes:
    """The PSID/RSID of tune ``name``."""
    src, kwargs = TUNES[name]
    kwargs = dict(kwargs)
    code, labels = assemble(src, LOAD)
    play = kwargs.pop("play", 0)
    play = labels[play] if isinstance(play, str) else play
    kwargs.setdefault("magic", b"RSID")
    return build_psid_embedded_load(
        code, LOAD, init=labels.get("init", LOAD), play=play, **kwargs
    )


def fixture(name: str) -> Path:
    """The recorded oracle trace of tune ``name`` (its first FIXTURE_CYCLES)."""
    return FIXTURES / f"{name}.csv.zst"


def record(names=None, work=Path(".oracle-cache/irqhost-fixtures")) -> None:
    """Trace each tune under sidtrace and keep its rows up to FIXTURE_CYCLES."""
    import os

    import zstandard

    from pysidtracker.irqhost import boot_cycle
    from pysidtracker.oracle import SIDTRACE_COLUMNS, read_sidtrace, run_sidtrace

    for name in names or TUNES:
        work.mkdir(parents=True, exist_ok=True)
        sid = work / f"{name}.sid"
        sid.write_bytes(build(name))
        user = f"{os.getuid()}:{os.getgid()}"
        trace = run_sidtrace(sid, work / f"{name}.csv.zst", seconds=1, user=user)
        ntsc = "ntsc" in name
        end = boot_cycle(ntsc) + FIXTURE_CYCLES
        lines = [",".join(SIDTRACE_COLUMNS)]
        for r in read_sidtrace(trace):
            if r.cycle <= end:
                cells = (r.cycle, r.since_nmi, r.since_video_irq, r.since_cia_irq)
                cells += (r.chip, r.reg, r.value)
                lines.append(",".join("" if c is None else str(c) for c in cells))
        text = "\n".join(lines) + "\n"
        FIXTURES.mkdir(parents=True, exist_ok=True)
        data = zstandard.ZstdCompressor(level=19).compress(text.encode())
        fixture(name).write_bytes(data)


if __name__ == "__main__":
    record(sys.argv[1:])
