"""Per-write provenance and a RAM watch from a tune's own 6502 play routine.

Run the tune's own init+play calls to a divergent frame, tracing each instruction
and SID write. ``explain`` names each write's instruction, load chain and cycle;
``watch`` prints watched RAM cells per frame as an oracle-aligned grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from vagrant_moses import memmap as mm

from . import registers as reg
from .emu import run_to_rts, set_iomap, wire_mpu
from .errors import EmulatorUnavailable, SidParseError
from .image import SidImage
from .poweron import power_on_ram, psid_init_bank, psid_play_bank

_W_A: FrozenSet[str] = frozenset(
    (
        "LDA",
        "PLA",
        "TXA",
        "TYA",
        "LXA",
        "LAX",
        "ORA",
        "AND",
        "EOR",
        "ADC",
        "SBC",
        "ASL",
        "LSR",
        "ROL",
        "ROR",
    )
)
_W_X: FrozenSet[str] = frozenset(("LDX", "TAX", "TSX", "INX", "DEX", "LAX", "LAS"))
_W_Y: FrozenSet[str] = frozenset(("LDY", "TAY", "INY", "DEY"))

_LOADS: FrozenSet[str] = frozenset(("LDA", "LDX", "LDY", "LAX", "LXA", "LAS"))
_ALU_A: FrozenSet[str] = frozenset(("ORA", "AND", "EOR", "ADC", "SBC"))
_XFER_SRC: Dict[str, str] = {"TXA": "X", "TYA": "Y", "TAX": "A", "TAY": "A"}
_INPLACE_A: FrozenSet[str] = frozenset(("ASL", "LSR", "ROL", "ROR"))
_STORE_SRC: Dict[str, str] = {"STA": "A", "STX": "X", "STY": "Y"}

_IDX_X: FrozenSet[str] = frozenset(("zpx", "absx"))
_IDX_Y: FrozenSet[str] = frozenset(("zpy", "absy"))

_DEFAULT_MAX_DEPTH = 8
_DEFAULT_MAX_CYCLES = 8_000_000


def _written_regs(mnemonic: str, mode: str) -> FrozenSet[str]:
    """The CPU registers ``mnemonic``/``mode`` leaves defined (a subset of A, X, Y)."""
    regs: set = set()
    if mnemonic in _W_A and (mnemonic not in _INPLACE_A or mode == "acc"):
        regs.add("A")
    if mnemonic in _W_X:
        regs.add("X")
    if mnemonic in _W_Y:
        regs.add("Y")
    return frozenset(regs)


@dataclass
class Insn:
    """One traced instruction and the operand it loads or stores.

    ``base`` names the operand as a label would (before any index); ``effective`` is
    the resolved address; ``index``/``index_value`` name the index register and its
    value for indexed modes.
    """

    pc: int
    opcode: int
    mnemonic: str
    mode: str
    writes: FrozenSet[str]
    imm: Optional[int] = None
    base: Optional[int] = None
    effective: Optional[int] = None
    index: Optional[str] = None
    index_value: Optional[int] = None


@dataclass
class Step:
    """A hop in a write's load chain, nearest to the store first.

    ``kind`` is ``mem`` (``source`` the cell loaded), ``imm`` (``source`` an
    immediate constant), ``alu`` (``operand`` the other argument of an accumulator
    op), ``xfer``, ``inplace``, or ``entry`` (live from call entry, undefined here).
    """

    pc: Optional[int]
    mnemonic: str
    kind: str
    source: Optional[int] = None
    operand: Optional[int] = None
    index: Optional[str] = None
    index_value: Optional[int] = None
    effective: Optional[int] = None
    disasm: str = ""


@dataclass
class Write:
    """A SID register write in one traced frame, with its value-load chain.

    ``cycle`` is the CPU cycle within the play call; ``register`` the ``$D400``-base
    SID offset; ``disasm`` the writing instruction under the label map; ``chain``
    the bounded back-track that produced ``value``.
    """

    frame: int
    cycle: int
    pc: int
    register: int
    value: int
    disasm: str
    chain: List[Step] = field(default_factory=list)


@dataclass
class Provenance:
    """A traced run across frames ``0..nframes-1``.

    ``frames[f]`` is frame ``f``'s :class:`Insn` trace (the chain back-track stays
    within one frame); ``writes[f]`` that frame's SID writes ``(pc, reg, value,
    cycle)``; ``watch`` the sampled cell value at each frame boundary.
    """

    subject: bytearray
    labels: Dict[str, int]
    nframes: int
    frames: List[List[Insn]] = field(default_factory=list)
    writes: List[List[Tuple[int, int, int, int]]] = field(default_factory=list)
    watch: Dict[int, List[int]] = field(default_factory=dict)
    clock: str = "PAL"


def _operand(
    pc: int, mode: str, mpu, mem
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[str], Optional[int]]:
    """``(imm, base, effective, index, index_value)`` of the instruction at ``pc``."""
    x, y = mpu.x, mpu.y
    if mode == "imm":
        return mem[pc + 1] & 0xFF, None, None, None, None
    if mode == "zp":
        addr = mem[pc + 1] & 0xFF
        return None, addr, addr, None, None
    if mode in _IDX_X:
        base = (mem[pc + 1] | (mem[pc + 2] << 8)) & 0xFFFF
        if mode == "zpx":
            base &= 0xFF
        return None, base, (base + x) & 0xFFFF, "X", x
    if mode in _IDX_Y:
        base = (mem[pc + 1] | (mem[pc + 2] << 8)) & 0xFFFF
        if mode == "zpy":
            base &= 0xFF
        return None, base, (base + y) & 0xFFFF, "Y", y
    if mode == "abs":
        addr = (mem[pc + 1] | (mem[pc + 2] << 8)) & 0xFFFF
        return None, addr, addr, None, None
    if mode == "indx":
        pointer = (mem[pc + 1] | (mem[pc + 2] << 8)) & 0xFFFF
        lo_addr = (pointer + x) & 0xFFFF
        return None, pointer, mem[lo_addr] | (mem[lo_addr + 1] << 8), "X", x
    if mode == "indy":
        pointer = (mem[pc + 1] | (mem[pc + 2] << 8)) & 0xFFFF
        effective = (mem[pointer] | (mem[pointer + 1] << 8)) + y
        return None, pointer, effective & 0xFFFF, "Y", y
    return None, None, None, None, None


def _mpu(memory: Optional[bytearray] = None):
    try:
        from jennings.devices.mpu6502 import MPU
    except ImportError as exc:  # pragma: no cover - jennings is a core dependency
        raise EmulatorUnavailable(
            "jennings is required to trace provenance: pip install pysidtracker"
        ) from exc
    return MPU(memory=memory) if memory is not None else MPU()


def decode_insn(pc: int, mpu, mem) -> Insn:
    """Decode the instruction at ``pc`` as the CPU is about to execute it.

    Reads the opcode, operands and index registers from the pre-step ``mem``/``mpu``
    state and classifies the registers it leaves defined; requires ``jennings``.
    """
    opcode = mem[pc] & 0xFF
    from jennings.opcodes import OPCODES

    mnemonic, mode = OPCODES[opcode]
    imm, base, effective, index, index_value = _operand(pc, mode, mpu, mem)
    return Insn(
        pc=pc & 0xFFFF,
        opcode=opcode,
        mnemonic=mnemonic,
        mode=mode,
        writes=_written_regs(mnemonic, mode),
        imm=imm,
        base=base,
        effective=effective,
        index=index,
        index_value=index_value,
    )


def _find_def(insns: Sequence[Insn], upto: int, reg: str) -> Optional[int]:
    """The last index ``<= upto`` whose instruction defines ``reg``, else ``None``."""
    for i in range(min(upto, len(insns) - 1), -1, -1):
        if reg in insns[i].writes:
            return i
    return None


def _load_chain(insns, store_idx, stored_reg, max_depth):
    """Back-track a store's value, one defining hop at a time, bounded by ``max_depth``.

    The nearest definition of the stored register is the first hop; a memory load names
    its cell, an immediate names the constant, an accumulator ALU also back-tracks the
    prior A, a transfer back-tracks its source register; entry-lives end at ``entry``.
    """

    def expand(reg: str, upto: int, depth: int) -> None:
        if depth > max_depth:
            return
        i = _find_def(insns, upto, reg)
        if i is None:
            steps.append(Step(None, f"<{reg} on call entry>", "entry"))
            return
        d = insns[i]
        if d.mnemonic in _LOADS:
            if d.mode == "imm":
                steps.append(Step(d.pc, d.mnemonic, "imm", source=d.imm))
            else:
                steps.append(
                    Step(
                        d.pc,
                        d.mnemonic,
                        "mem",
                        source=d.base,
                        index=d.index,
                        index_value=d.index_value,
                        effective=d.effective,
                    )
                )
        elif reg == "A" and d.mnemonic in _ALU_A:
            steps.append(
                Step(
                    d.pc,
                    d.mnemonic,
                    "alu",
                    operand=d.imm if d.mode == "imm" else d.base,
                    index=d.index,
                    index_value=d.index_value,
                    effective=d.effective,
                )
            )
            expand("A", i - 1, depth + 1)
        elif d.mnemonic in _XFER_SRC:
            steps.append(Step(d.pc, d.mnemonic, "xfer"))
            expand(_XFER_SRC[d.mnemonic], i - 1, depth + 1)
        elif reg == "A" and d.mnemonic in _INPLACE_A and d.mode == "acc":
            steps.append(Step(d.pc, d.mnemonic, "inplace"))
            expand("A", i - 1, depth + 1)
        else:
            steps.append(Step(d.pc, d.mnemonic, "other"))

    steps: List[Step] = []
    expand(stored_reg, store_idx - 1, 0)
    return steps


def _disasm(subject: bytearray, labels: Dict[str, int]):
    try:
        from jennings.disassembler import Disassembler
        from jennings.utils.addressing import AddressParser
    except ImportError as exc:  # pragma: no cover - jennings is a core dependency
        raise EmulatorUnavailable(
            "jennings is required to explain a write: pip install pysidtracker"
        ) from exc
    return Disassembler(
        _mpu(bytearray(subject)), AddressParser(labels=dict(labels or {}))
    )


class _PlaycallTracer:
    """One ``play`` call, stepped instruction by instruction with tracing on.

    Mirrors :func:`~pysidtracker.emu.run_to_rts` (a two-byte return address is pushed so
    the routine's final RTS ends the loop), recording each instruction and each SID
    write with the writing PC read from the CPU (still the executing instruction).
    """

    def __init__(self, subject: bytearray, mem, mpu):
        self._subject = subject
        self._mem = mem
        self._mpu = mpu
        self._captures: List[Tuple[int, int, int, int]] = []
        mem.subscribe_to_write(
            range(mm.SID_BASE, mm.SID_BASE + reg.SID_REG_COUNT), self._on_write
        )

    def _on_write(self, addr: int, value: int) -> None:
        offset = (addr - mm.SID_BASE) % mm.SID_STRIDE
        if 0 <= offset < reg.SID_REG_COUNT:
            self._captures.append(
                (self._mpu.pc, offset, value & 0xFF, self._mpu.processorCycles)
            )

    def run(self, play, max_cycles):
        mpu, mem, subject = self._mpu, self._mem, self._subject
        start_sp = mpu.sp
        mem[0x0100 + mpu.sp] = 0x00
        mpu.sp = (mpu.sp - 1) & 0xFF
        mem[0x0100 + mpu.sp] = 0x01
        mpu.sp = (mpu.sp - 1) & 0xFF
        mpu.a = 0
        mpu.pc = play
        start = mpu.processorCycles
        insns: List[Insn] = []
        self._captures = []
        while mpu.sp < start_sp:
            insns.append(decode_insn(mpu.pc, mpu, subject))
            mpu.step()
            if mpu.processorCycles - start > max_cycles:
                break
        writes = [(pc, r, v, c - start) for pc, r, v, c in self._captures]
        return insns, writes


def trace_frames(
    image: SidImage,
    nframes: int,
    *,
    watch: Sequence[int] = (),
    labels: Optional[Dict[str, int]] = None,
    ntsc: bool = False,
    max_cycles: int = _DEFAULT_MAX_CYCLES,
) -> Provenance:
    """Run ``image``'s play routine for ``nframes`` frames, tracing each frame.

    Loads over power-on RAM, seeds ``$D418``, runs ``init`` once, then steps
    ``nframes`` ``play`` calls per instruction so each SID write keeps its writing PC
    and cycle, with the frame's instruction list for the load chain.

    Args:
      image: a SID image with a header.
      nframes: frames (0-based) to trace.
      watch: RAM cells to sample at each frame boundary.
      labels: label map for disassembly.
      ntsc: trace at the NTSC clock.
      max_cycles: per-play-call cycle cap.

    Raises:
      SidParseError: ``image`` has no SID header.
      EmulatorUnavailable: ``jennings`` is missing.
    """
    if image.header is None:
        raise SidParseError("cannot trace provenance: image has no SID header")
    _mpu()

    subject = power_on_ram()
    subject[image.load : image.load + len(image.image)] = image.image
    subject[mm.SID_BASE + 0x18] = 0x0F
    init = image.header.init_address or image.header.real_load_address
    play = image.header.play_address or init
    subject[1] = psid_init_bank(init)
    mpu, mem = wire_mpu(
        subject, False, reg.sid_model_for_flags(image.header.flags), ntsc
    )
    run_to_rts(mpu, mem, init, 0, max_cycles)
    set_iomap(mpu, psid_play_bank(play))

    tracer = _PlaycallTracer(subject, mem, mpu)
    frames, writes = [], []
    grid: Dict[int, List[int]] = {a & 0xFFFF: [] for a in watch}
    for _ in range(nframes):
        insns, frame_writes = tracer.run(play, max_cycles)
        frames.append(insns)
        writes.append(frame_writes)
        for addr in grid:
            grid[addr].append(subject[addr] & 0xFF)

    return Provenance(
        subject=subject,
        labels=dict(labels or {}),
        nframes=nframes,
        frames=frames,
        writes=writes,
        watch=grid,
        clock="NTSC" if ntsc else "PAL",
    )


def explain(
    pro: Provenance,
    frame: int,
    *,
    reg: Optional[int] = None,
    max_depth: int = _DEFAULT_MAX_DEPTH,
) -> List[Write]:
    """The load chains of the SID writes in ``pro``'s frame ``frame``.

    One :class:`Write` per SID write in the frame (or just ``reg``), each with the
    writing instruction's address/disassembly, the frame and cycle within the frame,
    and the bounded back-track that produced the value.

    Args:
      pro: a :class:`Provenance`.
      frame: frame to explain.
      reg: restrict to one SID register offset.
      max_depth: load-chain back-track depth.

    Raises:
      SidParseError: ``frame`` is outside the traced range.
    """
    if not 0 <= frame < pro.nframes:
        raise SidParseError(f"frame {frame} out of range (traced {pro.nframes})")
    dis = _disasm(pro.subject, pro.labels)
    insns = pro.frames[frame]
    out: List[Write] = []
    for pc, reg_off, value, cycle in pro.writes[frame]:
        if reg is not None and reg_off != reg:
            continue
        store_idx = next(
            (i for i, d in enumerate(insns) if d.pc == (pc & 0xFFFF)), None
        )
        chain: List[Step] = []
        if store_idx is not None:
            chain = _load_chain(
                insns,
                store_idx,
                _STORE_SRC.get(insns[store_idx].mnemonic, "A"),
                max_depth,
            )
        for step in chain:
            if step.pc is not None:
                _, step.disasm = dis.instruction_at(step.pc)
        out.append(
            Write(
                frame,
                cycle,
                pc & 0xFFFF,
                reg_off,
                value,
                _text(dis, pc & 0xFFFF) if store_idx is not None else "",
                chain,
            )
        )
    return out


def _text(dis, pc: int) -> str:
    _, text = dis.instruction_at(pc)
    return text


def watch_grid(
    pro: Provenance, addrs: Optional[Sequence[int]] = None
) -> List[List[int]]:
    """The watched cells as ``[addr] -> [value per frame]``, one row per cell.

    Row ``f`` is a cell's value after frame ``f``'s play call (aligned with the
    oracle's frames); ``addrs`` defaults to every cell :func:`trace_frames` sampled.
    """
    cells = list(addrs) if addrs else sorted(pro.watch)
    return [
        [
            pro.watch[a & 0xFFFF][f] if (a & 0xFFFF) in pro.watch else 0
            for f in range(pro.nframes)
        ]
        for a in cells
    ]


def format_explain(pro, frame, *, reg=None, depth=_DEFAULT_MAX_DEPTH) -> str:
    """The :func:`explain` rows for ``frame`` as text, naming each chain hop's source."""
    lines = [f"frame {frame} (clock {pro.clock}, {pro.nframes} traced)"]
    for w in explain(pro, frame, reg=reg, max_depth=depth):
        lines.append(
            f"  reg {w.register:2d}  ${w.pc:04X}  {w.disasm}   value ${w.value:02X}  @ cycle {w.cycle}"
        )
        for s in w.chain:
            what = s.disasm or s.mnemonic
            if s.kind == "mem" and s.index is not None:
                extra = f"   cell ${s.source:04X} + {s.index}=${s.index_value:02X}"
            elif s.kind == "mem":
                extra = f"   cell ${s.source:04X}"
            elif s.kind == "imm":
                extra = f"   constant ${s.source:02X}"
            elif s.kind == "alu" and s.operand is not None:
                extra = f"   operand ${s.operand:04X}"
            else:
                extra = ""
            lines.append(f"      {s.kind:9}  {what}{extra}")
    return "\n".join(lines)


def format_watch(pro, addrs: Optional[Sequence[int]] = None) -> str:
    """The :func:`watch_grid` table as text: frames down, watched cells across."""
    cells = list(addrs) if addrs else sorted(pro.watch)
    grid = watch_grid(pro, cells)
    lines = ["frame " + "  ".join("${%04X}" % a for a in cells)]
    for f in range(pro.nframes):
        lines.append(
            f"{f:>5} " + "  ".join("%06X" % grid[i][f] for i in range(len(grid)))
        )
    return "\n".join(lines)


__all__ = [
    "Insn",
    "Provenance",
    "Step",
    "Write",
    "decode_insn",
    "explain",
    "format_explain",
    "format_watch",
    "trace_frames",
    "watch_grid",
]
