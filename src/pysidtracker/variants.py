"""Cluster unknown player builds by the code that actually runs.

Run a tune's executed init/play on jennings (``emu`` for a direct ``play``,
``irqhost`` for an IRQ tune) and fingerprint the span: each executed instruction
normalised to a relocation-invariant form, hashed per block and JSR-routine.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from .errors import EmulatorUnavailable, SidParseError
from .image import SidImage

_LENS = {
    "impl": 1,
    "acc": 1,
    "imm": 2,
    "rel": 2,
    "zp": 2,
    "zpx": 2,
    "zpy": 2,
    "abs": 3,
    "absx": 3,
    "absy": 3,
    "ind": 3,
    "indx": 3,
    "indy": 3,
}

_TERMINATORS = frozenset("BPL BMI BVC BVS BCC BCS BNE BEQ JSR JMP RTS RTI BRK".split())

_VECTOR_NAMES = {
    0x0314: "IRQ",
    0x0315: "IRQ",
    0x0316: "BRK",
    0x0317: "BRK",
    0x0318: "NMI",
    0x0319: "NMI",
    0xFFFA: "NMI",
    0xFFFB: "NMI",
    0xFFFC: "RESET",
    0xFFFD: "RESET",
    0xFFFE: "IRQ",
    0xFFFF: "IRQ",
}

_IO_PAGE_LO = 0xD000
_IO_PAGE_HI = 0xDFFF


def _io_name(addr: int) -> str:
    """The symbolic name of an I/O-page register (the $20 mirrors fold together)."""
    addr &= 0xFFFF
    if 0xD400 <= addr <= 0xD7FF:
        reg_off = (addr - 0xD400) % 0x20
        if reg_off < 0x19:
            voice, inner = divmod(reg_off, 7)
            name = ("FREQ_LO", "FREQ_HI", "PW_LO", "PW_HI", "CTRL", "AD", "SR")[inner]
            return f"S{voice}_{name}"
        return f"SD{addr:04X}"
    if 0xDC00 <= addr <= 0xDC0F:
        return f"CIA1_{addr & 0x0F:02X}"
    if 0xDD00 <= addr <= 0xDD0F:
        return f"CIA2_{addr & 0x0F:02X}"
    if 0xD000 <= addr <= 0xD0FF:
        return f"VIC_{addr - 0xD000:02X}"
    return f"IO_{addr:04X}"


def _op_class(addr: int, base: int, lo: int, hi: int) -> str:
    """The relocation-invariant operand class of an address operand."""
    addr &= 0xFFFF
    if addr <= 0xFF:
        return "zp"
    if lo <= addr < hi:
        return f"rel:{addr - base}"
    if _IO_PAGE_LO <= addr <= _IO_PAGE_HI:
        return "io:" + _io_name(addr)
    if addr in _VECTOR_NAMES:
        return "vec:" + _VECTOR_NAMES[addr]
    if addr < 0xA000:
        return f"ram:{addr}"
    return f"ext:{addr}"


def _decode(mem, pc: int, base: int, lo: int, hi: int) -> Tuple[int, str, str, str]:
    """``(length, mnemonic, mode, operand)`` of the instruction at ``pc``."""
    from jennings.opcodes import OPCODES

    opcode = mem[pc] & 0xFF
    mnemonic, mode = OPCODES[opcode]
    length = _LENS.get(mode, 1)
    if mode == "imm":
        operand = str(mem[pc + 1] & 0xFF)
    elif mode == "rel":
        raw = mem[pc + 1] & 0xFF
        operand = str(raw - 0x100 if raw & 0x80 else raw)
    elif mode in ("impl", "acc"):
        operand = ""
    else:
        addr = (mem[pc + 1] & 0xFF) | ((mem[pc + 2] & 0xFF) << 8)
        operand = _op_class(addr, base, lo, hi)
    return length, mnemonic, mode, operand


def _is_direct(image: SidImage) -> bool:
    """True when the header names a ``play`` routine distinct from ``init``."""
    header = image.header
    if header is None:
        return False
    init = header.init_address or image.load
    return bool(header.play_address) and header.play_address != init


class _Tracer:
    """Install a ``mpu.step`` hook that records each executed PC as an offset.

    Only instruction starts inside ``[base, hi)`` are kept; the hook is removed
    when the context exits.
    """

    def __init__(self, mpu, base: int, hi: int):
        self._mpu = mpu
        self._step = mpu.step
        self._base = base
        self._hi = hi
        self.offsets: List[int] = []

    def __enter__(self) -> "_Tracer":
        self._mpu.step = self._traced  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc) -> None:
        self._mpu.step = self._step  # type: ignore[method-assign]

    def _traced(self) -> None:
        pc = self._mpu.pc
        if self._base <= pc < self._hi:
            self.offsets.append(pc - self._base)
        self._step()


def _executed_offsets(
    image: SidImage,
    *,
    subtune: int = 0,
    clock: Optional[str] = None,
    nframes: int = 8,
    max_cycles: int = 20_000_000,
) -> List[int]:
    """Ordered executed span-instruction-start offsets (with repeats) of a tune.

    Direct tunes (a distinct ``play``) run init once and ``play`` ``nframes``
    times on ``emu.wire_mpu``; IRQ tunes run ``nframes`` frames on
    ``irqhost.InterruptHost``.
    """
    header = image.header
    if header is None:
        raise SidParseError("cannot fingerprint a bare .prg: image has no SID header")
    if clock is not None and clock.strip().upper() not in ("PAL", "NTSC"):
        raise ValueError(f"clock must be 'PAL' or 'NTSC', got {clock!r}")
    ntsc = bool(clock and clock.strip().upper() == "NTSC") or (
        (header.flags >> 2) & 3 == 2
    )
    if _is_direct(image):
        return _executed_direct(image, subtune, ntsc, nframes, max_cycles)
    return _executed_irq(image, subtune, ntsc, nframes, image.load, image.end)


def _executed_direct(image, subtune, ntsc, nframes, max_cycles) -> List[int]:
    """The executed offsets of a direct (init + per-frame play) tune."""
    from .emu import run_to_rts, set_iomap, wire_mpu
    from .poweron import power_on_ram, psid_init_bank, psid_play_bank
    from .registers import sid_model_for_flags

    header = image.header
    init = header.init_address or image.load  # type: ignore[union-attr]
    play = header.play_address  # type: ignore[union-attr]
    subject = power_on_ram()
    subject[image.load : image.end] = image.image
    subject[0xD418] = 0x0F
    subject[1] = psid_init_bank(init)
    mpu, mem = wire_mpu(subject, sid_model=sid_model_for_flags(header.flags), ntsc=ntsc)
    with _Tracer(mpu, image.load, image.end) as tracer:
        run_to_rts(mpu, mem, init, subtune & 0xFF, max_cycles)
        set_iomap(mpu, psid_play_bank(play))
        for _ in range(nframes):
            run_to_rts(mpu, mem, play, 0, max_cycles)
    return tracer.offsets


def _executed_irq(image, subtune, ntsc, nframes, base, hi) -> List[int]:
    """The executed offsets of an IRQ tune, run for ``nframes`` frames."""
    from .irqhost import InterruptHost

    host = InterruptHost(image, subtune, clock="NTSC" if ntsc else "PAL")
    with _Tracer(host.machine.mpu, base, hi) as tracer:
        host.run(nframes * host.frame_cycles)
    return tracer.offsets


def _digest(parts) -> str:
    """A short, stable digest of an ordered iterable of hashable parts."""
    h = hashlib.sha256()
    for part in parts:
        h.update(repr(part).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:24]


@dataclass(frozen=True)
class NormInsn:
    """One executed span instruction and its relocation-invariant form.

    ``offset`` is the PC minus the load address; ``operand`` the normalised
    operand; ``disasm`` the instruction text under the header entry labels.
    """

    offset: int
    mn: str
    mode: str
    operand: str
    disasm: str

    @property
    def norm(self) -> Tuple[str, str, str]:
        """The ``(mnemonic, mode, operand)`` triple used for hashing and diffing."""
        return (self.mn, self.mode, self.operand)


@dataclass(frozen=True)
class Block:
    """A basic block: a run of span instructions ending at a control-flow point."""

    start: int
    norms: tuple

    @property
    def digest(self) -> str:
        """Stable hash of the block's normalised instruction sequence."""
        return _digest(self.norms)


@dataclass
class Signature:
    """A player build's relocation-invariant fingerprint.

    ``insns`` are the executed span instructions in offset order, ``path`` the
    host that ran it, ``blocks`` the basic blocks and ``routines`` the hash of
    the code each JSR target executes.
    """

    base: int
    path: str
    insns: tuple
    blocks: tuple
    routines: tuple

    @property
    def block_digests(self) -> frozenset:
        return frozenset(b.digest for b in self.blocks)

    @property
    def routine_digests(self) -> frozenset:
        return frozenset(self.routines)

    @property
    def all_digests(self) -> frozenset:
        return self.block_digests | frozenset(self.routines)

    @property
    def key(self) -> tuple:
        """The exact-match cluster key: sorted block and routine digests."""
        return (
            tuple(sorted(self.block_digests)),
            tuple(sorted(self.routine_digests)),
        )

    def jaccard(self, other: "Signature") -> float:
        """Similarity over the combined block/routine hash sets (1.0: identical)."""
        a, b = self.all_digests, other.all_digests
        if not a and not b:
            return 1.0
        union = len(a | b)
        return len(a & b) / union if union else 0.0


def _disassembler(image: SidImage):
    """A jennings Disassembler over the image's memory, with entry labels."""
    try:
        from jennings.disassembler import Disassembler
        from jennings.devices.mpu6502 import MPU
        from jennings.utils.addressing import AddressParser
    except ImportError as exc:  # pragma: no cover - jennings is a core dependency
        raise EmulatorUnavailable(
            "jennings is required to fingerprint a player: pip install pysidtracker"
        ) from exc
    from .disasm import entry_labels

    return Disassembler(
        MPU(memory=bytearray(image.mem)),
        AddressParser(labels=dict(entry_labels(image))),
    )


def build_signature(
    data: Union[bytes, bytearray, SidImage],
    *,
    subtune: int = 0,
    clock: Optional[str] = None,
    nframes: int = 8,
    max_cycles: int = 20_000_000,
) -> Signature:
    """Fingerprint one tune (PSID/RSID ``data`` bytes, or a :class:`SidImage`).

    Raises :class:`SidParseError` for a bare ``.prg`` or an empty span, and
    :class:`EmulatorUnavailable` if ``jennings`` is missing.
    """
    image = data if isinstance(data, SidImage) else SidImage.from_bytes(bytes(data))
    base, hi = image.load, image.end
    ordered = _executed_offsets(
        image, subtune=subtune, clock=clock, nframes=nframes, max_cycles=max_cycles
    )
    if not ordered:
        raise SidParseError(
            f"no executed instructions in the loaded span of {hex(base)}..{hex(hi)}"
        )

    disasm = _disassembler(image)
    info: Dict[int, NormInsn] = {}
    for offset in ordered:
        if offset in info:
            continue
        pc = base + offset
        _, mn, mode, operand = _decode(image.mem, pc, base, image.load, hi)
        try:
            text = disasm.instruction_at(pc)[1]
        except Exception:  # pylint: disable=broad-except
            text = f"{mn} ?"
        info[offset] = NormInsn(offset, mn, mode, operand, text)

    insns = tuple(info[o] for o in sorted(info))
    blocks, offset_block = _blocks(insns)
    routines = _routines(ordered, offset_block, info)
    return Signature(
        base,
        "direct" if _is_direct(image) else "irq",
        insns,
        tuple(blocks),
        tuple(sorted(routines)),
    )


def _blocks(insns: Sequence[NormInsn]) -> Tuple[List[Block], Dict[int, str]]:
    """Split ``insns`` into basic blocks; map each offset to its block digest."""
    blocks: List[Block] = []
    offset_block: Dict[int, str] = {}
    current: List[NormInsn] = []
    next_offset: Optional[int] = None

    def close() -> None:
        if not current:
            return
        block = Block(current[0].offset, tuple(i.norm for i in current))
        blocks.append(block)
        for i in current:
            offset_block[i.offset] = block.digest
        current.clear()

    for insn in insns:
        if current and insn.offset != next_offset:
            close()
        current.append(insn)
        next_offset = insn.offset + _LENS.get(insn.mode, 1)
        if insn.mn in _TERMINATORS:
            close()
    close()
    return blocks, offset_block


def _routines(ordered, offset_block, info) -> Set[str]:
    """The digest of the code each executed JSR target runs before it returns."""
    frames: List[Set[str]] = []
    digests: Set[str] = set()
    for offset in ordered:
        digest = offset_block.get(offset)
        if frames and digest is not None:
            frames[-1].add(digest)
        mn = info[offset].mn
        if mn == "JSR":
            frames.append(set())
        elif mn == "RTS" and frames:
            body = frames.pop()
            if body:
                digests.add(_digest(sorted(body)))
    return digests


def cluster(
    sigs: Sequence[Signature], threshold: float = 0.8
) -> Tuple[List[int], List[int]]:
    """Greedy cluster ``sigs`` by :meth:`Signature.jaccard` >= ``threshold``.

    Returns ``(assign, sizes)``: ``assign[i]`` the cluster id (its first member
    is the representative) and ``sizes[c]`` cluster ``c``'s member count.
    """
    assign: List[int] = []
    clusters: List[List[int]] = []
    for i, sig in enumerate(sigs):
        for ci, members in enumerate(clusters):
            if sig.jaccard(sigs[members[0]]) >= threshold:
                assign.append(ci)
                members.append(i)
                break
        else:
            clusters.append([i])
            assign.append(len(clusters) - 1)
    return assign, [len(m) for m in clusters]


def nearest_ref(
    sig: Signature, refs: Sequence[Signature]
) -> Tuple[Optional[int], float]:
    """The nearest reference's index and distance (1 - Jaccard); ``(None, -1)`` if none."""
    best_i, best_d = None, 1.0
    for i, ref in enumerate(refs):
        d = 1.0 - sig.jaccard(ref)
        if d < best_d:
            best_i, best_d = i, d
    return best_i, (best_d if best_i is not None else -1.0)


def diff_signatures(subject: Signature, ref: Signature) -> List[Tuple[str, str, str]]:
    """The ``(tag, subject_text, ref_text)`` opcodes subject has versus ref.

    ``tag`` is ``"inserted"`` (subject only), ``"removed"`` (ref only) or
    ``"changed"`` (both, differing); the two span instructions are aligned by
    their normalised form.
    """
    a = [i.norm for i in ref.insns]
    b = [i.norm for i in subject.insns]
    ops: List[Tuple[str, str, str]] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(
        None, a, b, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            ops += [("removed", "", ref.insns[k].disasm) for k in range(i1, i2)]
        elif tag == "insert":
            ops += [("inserted", subject.insns[k].disasm, "") for k in range(j1, j2)]
        else:
            ops += [
                ("changed", subject.insns[j1 + k].disasm, ref.insns[i1 + k].disasm)
                for k in range(min(i2 - i1, j2 - j1))
            ]
    return ops


def format_diff(subject: Signature, ref: Signature) -> str:
    """The :func:`diff_signatures` ops as ``+``/``-``/``~`` text (or ``identical``)."""
    ops = diff_signatures(subject, ref)
    if not ops:
        return "identical"
    lines = []
    for tag, subj, refdisp in ops:
        if tag == "inserted":
            lines.append(f"+ {subj}")
        elif tag == "removed":
            lines.append(f"- {refdisp}")
        else:
            lines.append(f"~ {subj}  <-  {refdisp}")
    return "\n".join(lines)


ROW_FIELDS = [
    "name",
    "cluster",
    "n_insns",
    "n_blocks",
    "n_routines",
    "nearest_ref",
    "distance",
    "ndiff",
]


def variants(
    tunes: Sequence[Union[bytes, bytearray, SidImage]],
    refs: Sequence[Tuple[str, Union[bytes, bytearray, SidImage]]] = (),
    *,
    names: Optional[Sequence[str]] = None,
    nframes: int = 8,
    threshold: float = 0.8,
) -> Tuple[List[Dict], List[int]]:
    """Fingerprint ``tunes``, cluster them and score each against ``refs``.

    Returns ``(rows, sizes)``: one :data:`ROW_FIELDS` dict per tune and each
    cluster's member count. A tune's ``nearest_ref``/``distance``/``ndiff`` are
    against its closest ref (empty when there are none).
    """
    names = list(names) if names else [f"tune{i}" for i in range(len(tunes))]
    sigs = [build_signature(data, nframes=nframes) for data in tunes]
    refsigs = [(name, build_signature(data, nframes=nframes)) for name, data in refs]
    assign, sizes = cluster(sigs, threshold)
    refmap = {name: sig for name, sig in refsigs}
    rows = []
    for i, sig in enumerate(sigs):
        row = {
            "name": names[i],
            "cluster": assign[i],
            "n_insns": len(sig.insns),
            "n_blocks": len(sig.blocks),
            "n_routines": len(sig.routines),
            "nearest_ref": "",
            "distance": "",
            "ndiff": "",
            "_diff": "",
        }
        if refsigs:
            ri, distance = nearest_ref(sig, [sig2 for _, sig2 in refsigs])
            name = refsigs[ri][0]
            ops = diff_signatures(sig, refmap[name])
            row.update(
                nearest_ref=name,
                distance=f"{distance:.4f}",
                ndiff=len(ops),
                _diff=format_diff(sig, refmap[name]),
            )
        rows.append(row)
    return rows, sizes


def _tunes_from_paths(
    paths: Sequence[Union[str, Path]], hvsc: Optional[Union[str, Path]]
) -> List[Tuple[str, bytes]]:
    """Resolve tune paths (``.sid``/``.prg`` files or a ``find`` CSV) to data."""
    tunes: List[Tuple[str, bytes]] = []
    for path in paths:
        pp = Path(path)
        if pp.suffix.lower() == ".csv":
            root = Path(hvsc) if hvsc else Path(".")
            with open(pp, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    sidpath = root / row["path"]
                    tunes.append((pp.name + ":" + row["path"], sidpath.read_bytes()))
        else:
            tunes.append((pp.name, pp.read_bytes()))
    return tunes


def variants_cli(
    paths: Sequence[Union[str, Path]],
    refs: Sequence[Union[str, Path]] = (),
    *,
    hvsc: Optional[Union[str, Path]] = None,
    out: Optional[Union[str, Path]] = None,
    nframes: int = 8,
    threshold: float = 0.8,
) -> List[Dict]:
    """``survey variants`` workhorse: load, fingerprint, cluster, diff and report.

    Prints a per-cluster summary and each tune's nearest ref and diff, and
    writes the ``ROW_FIELDS`` CSV to ``out`` when given.
    """
    tunes = _tunes_from_paths(paths, hvsc)
    names = [name for name, _ in tunes]
    refsigs = [(Path(r).name, Path(r).read_bytes()) for r in refs]
    rows, sizes = variants(
        [data for _, data in tunes],
        refsigs,
        names=names,
        nframes=nframes,
        threshold=threshold,
    )
    lines = [
        f"{len(tunes)} tunes, {len(sizes)} cluster(s): "
        + ", ".join(f"c{c}={n}" for c, n in enumerate(sizes))
    ]
    for row in rows:
        lines.append(
            f"  c{row['cluster']} {row['name']}: {row['n_blocks']} blocks, "
            f"{row['n_routines']} routines, nearest={row['nearest_ref']} "
            f"d={row['distance']} ndiff={row['ndiff']}"
        )
        if row["ndiff"]:
            for ln in row["_diff"].splitlines():
                lines.append(f"      {ln}")
    text = "\n".join(lines) + "\n"
    if out is not None:
        variants_report_csv(rows, out)
    sys.stdout.write(text)
    return rows


def variants_report_csv(rows: Sequence[Dict], out: Union[str, Path]) -> Path:
    """Write ``rows`` (``ROW_FIELDS`` dicts) to ``out`` as CSV (extra keys dropped)."""
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, ROW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return outp


__all__ = [
    "Block",
    "NormInsn",
    "ROW_FIELDS",
    "Signature",
    "build_signature",
    "cluster",
    "diff_signatures",
    "format_diff",
    "nearest_ref",
    "variants",
    "variants_cli",
    "variants_report_csv",
]
