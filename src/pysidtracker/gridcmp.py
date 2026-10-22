"""The shared core of the byte-exact register-grid comparison.

Grid serialization, the lead scan, the onset/carry cell diff, the verdicts and
their reports, and the ``python -m pysidtracker.gridcmp`` CLI. The survey
(:mod:`pysidtracker.survey`) re-exports the core it orchestrates.
"""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import registers as reg

LEAD = 8
NEAR = 3
NREGS = reg.SID_REG_COUNT
VOICE_ROLES = ("freq", "freq", "pw", "pw", "ctrl", "ad", "sr")
GATE_COLS = [off + reg.CTRL for off in reg.SID_VOICE_OFFSET]
ROLES = [n for _ in range(reg.SID_VOICES) for n in VOICE_ROLES] + [
    "fc",
    "fc",
    "res",
    "vol",
]
REG_NAMES = [
    f"v{v + 1}_{n}"
    for v in range(reg.SID_VOICES)
    for n in ("freq_lo", "freq_hi", "pw_lo", "pw_hi", "ctrl", "ad", "sr")
] + ["fc_lo", "fc_hi", "res_filt", "mode_vol"]
_VREGS = reg.SID_VOICE_REG_SIZE
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def write_grid(grid, out):
    """A register grid as a zstd CSV of per-frame register changes."""
    grid = np.asarray(grid)
    change = np.ones(grid.shape, bool)
    change[1:] = grid[1:] != grid[:-1]
    f, r = np.nonzero(change)
    buf = io.StringIO()
    buf.write("frame,reg,value\n")
    np.savetxt(buf, np.column_stack([f, r, grid[f, r]]), fmt="%d", delimiter=",")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import zstandard  # lazy: an optional dependency of the survey extra

    out.write_bytes(zstandard.ZstdCompressor().compress(buf.getvalue().encode()))


def read_grid(path, nframes=None):
    """The inverse of :func:`write_grid`, forward-filled to ``nframes``.

    ``nframes`` None reads to the last change's frame; the file is a zstd CSV
    or a plain CSV (detected by the zstd magic bytes).
    """
    data = Path(path).read_bytes()
    if data[:4] == _ZSTD_MAGIC:
        import zstandard  # lazy: an optional dependency of the survey extra

        data = zstandard.ZstdDecompressor().decompress(data)
    text = data.decode()
    body = "\n".join(text.splitlines()[1:])
    if not body.strip():
        return np.zeros((1 if nframes is None else nframes, NREGS), np.int16)
    rows = np.loadtxt(io.StringIO(body), np.int64, delimiter=",", ndmin=2)
    nframes = int(rows[:, 0].max()) + 1 if nframes is None else nframes
    f, r, v = rows[rows[:, 0] < nframes].T
    grid = np.zeros((nframes, NREGS), dtype=np.int16)
    src = np.zeros((nframes, NREGS), dtype=np.int64)
    grid[f, r], src[f, r] = v, f
    np.maximum.accumulate(src, axis=0, out=src)
    return np.take_along_axis(grid, src, 0)


def aligned(oracle, lib, lead):
    """Equal-length ``(oracle, lib)`` views, ``lib`` shifted by ``lead`` frames."""
    if lead >= 0:
        return oracle, lib[lead : lead + len(oracle)]
    return oracle[-lead:], lib[: len(oracle) + lead]


def scan_leads(oracle, lib, max_lead=LEAD):
    """Per-lead mismatches of ``lib`` against ``oracle`` up to ``max_lead``, sorted by ``|lead|``.

    Returns ``(leads, ne, size, first, nbad)``: per lead the mismatch mask
    ``ne[i, f, r]`` (False past ``size[i]``), the first bad frame ``first[i]``
    (``size[i]`` when none) and the bad-frame count ``nbad[i]``.
    """
    n = len(oracle)
    leads = np.array(sorted(range(-min(max_lead, n - 1), max_lead + 1), key=abs))
    frames = np.arange(n)
    size = n - np.maximum(-leads, 0)
    o = np.minimum(frames + np.maximum(-leads, 0)[:, None], n - 1)
    l = frames + np.maximum(leads, 0)[:, None]
    ne = (oracle[o] != lib[l]) & (frames < size[:, None])[..., None]
    bad = ne.any(2)
    first = np.where(bad.any(1), bad.argmax(1), size)
    return leads, ne, size, first, bad.sum(1)


def reg_names(mask):
    """Space-separated :data:`REG_NAMES` of a boolean register mask."""
    return " ".join(REG_NAMES[r] for r in np.flatnonzero(mask))


def compare(oracle, lib):
    """``(lead, bad, first)``: the lead of ``lib`` with the latest first mismatch."""
    leads, ne, size, first, _ = scan_leads(oracle, lib)
    i = int(np.argmax(first))
    return int(leads[i]), ne[i, : size[i]].any(1), int(first[i])


def min_lead(scan):
    """The fewest-bad-frames lead of a :func:`scan_leads`, with its stats.

    The smallest ``|lead|`` on ties; ``min_bad`` its bad count, ``min_first``
    its first bad frame (empty when none) and ``min_regs`` their registers.
    """
    leads, ne, size, first, nbad = scan
    i = int(np.argmin(nbad))
    return dict(
        min_lead=int(leads[i]),
        min_bad=int(nbad[i]),
        min_first=int(first[i]) if nbad[i] else "",
        min_regs=reg_names(ne[i, : size[i]].any(0)),
    )


def verdict_fields(scan):
    """The ``results.csv`` comparison columns of a :func:`scan_leads`.

    ``lead`` has the latest first mismatch; ``min_lead`` the fewest bad frames.
    """
    leads, ne, size, first, nbad = scan
    i = int(np.argmax(first))
    return dict(
        verdict="differs" if nbad[i] else "exact",
        lead=int(leads[i]),
        first_diff=int(first[i]) if nbad[i] else "",
        diff_frames=int(nbad[i]),
        diff_regs=reg_names(ne[i, : size[i]].any(0)),
        **min_lead(scan),
    )


def compare_grids(a, b, max_shift=LEAD):
    """:func:`verdict_fields` of render ``b`` against render ``a`` at every constant shift.

    Plus the compared ``frames``; ``b[f + min_lead]`` best matches ``a[f]``,
    and ``a`` is cut to ``len(b) - max_shift`` frames.
    """
    a, b = np.asarray(a, np.int16), np.asarray(b, np.int16)
    n = min(len(a), len(b) - max_shift)
    if n < 1:
        raise ValueError(f"grids of {len(a)} and {len(b)} frames, shift {max_shift}")
    return dict(frames=n, **verdict_fields(scan_leads(a[:n], b, max_shift)))


def gated(grid):
    """Whether any voice's gate bit is ever set."""
    return bool((np.asarray(grid)[:, GATE_COLS] & 1).any())


def classify(frame, diffs, row):
    """``(when, kind, roles)`` of a first difference (see docs/survey.md)."""
    when = (
        "frame0"
        if frame == 0
        else "early" if frame < 10 else "mid" if frame < 250 else "late"
    )
    roles = " ".join(sorted({ROLES[r] for r, _, _ in diffs}))
    wrong = {r: (o, l) for r, o, l in diffs}
    deltas = {l - o for o, l in wrong.values()}
    if row.get("library_gated") == "False" and row.get("oracle_gated") == "True":
        kind = "silent"
    elif any(
        o and any(l2 == o for r2, (_, l2) in wrong.items() if r2 != r)
        for r, (o, _) in wrong.items()
    ):
        kind = "misplaced"
    elif len(wrong) > 1 and len(deltas) == 1:
        kind = f"offset{deltas.pop():+#x}"
    elif all(not l for _, l in wrong.values()):
        kind = "missing"
    elif all(not o for o, _ in wrong.values()):
        kind = "extra"
    else:
        kind = "value"
    return when, kind, roles


def onset_mask(ours, starts):
    """The cells that count under the onset criterion, as a boolean mask.

    Each voice's registers from its first note in ``starts`` (``None`` voices
    skipped), the shared registers from the earliest note.
    """
    ours = np.asarray(ours)
    mask = np.zeros(ours.shape, dtype=bool)
    begun = [s for s in starts if s is not None]
    for voice, start in enumerate(starts):
        if start is not None:
            mask[start:, voice * _VREGS : voice * _VREGS + _VREGS] = True
    if begun:
        mask[min(begun) :, _VREGS * len(starts) :] = True
    return mask


def carry(ours, theirs):
    """The cells a 1-LSB carry makes legitimately differ, as a boolean mask.

    A waveride walk leaves the 6502 C flag set, adding 1 to a voice's 16-bit
    freq and to the shared 11-bit cutoff (a pulse-width add clears it); a pair
    differing only by that 1 is the carry, not a difference.
    """
    ours = np.asarray(ours, dtype=np.uint8)
    theirs = np.asarray(theirs, dtype=np.uint8)
    tol = np.zeros(ours.shape, dtype=bool)
    for v in range(ours.shape[1] // _VREGS):
        lo, hi = v * _VREGS, v * _VREGS + 1
        o = ours[:, hi].astype(int) * 256 + ours[:, lo]
        t = theirs[:, hi].astype(int) * 256 + theirs[:, lo]
        ok = ((t - o) % 65536 == 1) | ((o - t) % 65536 == 1)
        tol[ok, lo] = tol[ok, hi] = True
    if ours.shape[1] > reg.FC_HI:
        o = ours[:, reg.FC_HI].astype(int) * 8 + ours[:, reg.FC_LO]
        t = theirs[:, reg.FC_HI].astype(int) * 8 + theirs[:, reg.FC_LO]
        ok = ((t - o) % 2048 == 1) | ((o - t) % 2048 == 1)
        tol[ok, reg.FC_LO] = tol[ok, reg.FC_HI] = True
    return tol


def cell_diff(ours, theirs, *, starts=None, tol=None):
    """The ``(step, register, ours, theirs)`` cells where two grids differ.

    In ``(step, register)`` order, a ``(n, 4)`` int64 array. ``starts`` restricts
    to the onset criterion (None: every register from step 0); ``tol`` exempts
    the cells a mask or callable returning one marks true, or ``"carry"``.
    """
    ours = np.asarray(ours, dtype=np.uint8)
    theirs = np.asarray(theirs, dtype=np.uint8)
    n = min(ours.shape[0], theirs.shape[0])
    if n == 0:
        return np.zeros((0, 4), np.int64)
    ours, theirs = ours[:n], theirs[:n]
    bad = ours != theirs
    if starts is not None:
        bad &= onset_mask(ours, starts)
    if tol is not None:
        if isinstance(tol, str):
            exempt = carry(ours, theirs) if tol == "carry" else np.asarray(tol, bool)
        else:
            exempt = tol(ours, theirs) if callable(tol) else tol
        bad &= ~np.asarray(exempt, dtype=bool)
    f, r = np.nonzero(bad)
    out = np.empty((f.size, 4), np.int64)
    out[:, 0], out[:, 1] = f, r
    out[:, 2], out[:, 3] = ours[f, r].astype(np.int64), theirs[f, r].astype(np.int64)
    return out


def first_mismatch(ours, theirs, starts=None):
    """The first differing cell of a carry-tolerant :func:`cell_diff`.

    ``(step, register)``, or None where the grids agree on the onset criterion
    (every register from step 0 when ``starts`` is None).
    """
    cells = cell_diff(ours, theirs, starts=starts, tol="carry")
    return (int(cells[0][0]), int(cells[0][1])) if cells.size else None


def runs(cells, names=REG_NAMES):
    """The differing cells grouped by register, in register order.

    Returns each register's maximal step runs and, in each run, the maximal
    segments of constant ``(ours, theirs)`` value: ``{'reg', 'name', 'from',
    'to', 'cells', 'runs': [{'from', 'to', 'values': [[...], ...]}]}``.
    """
    grouped: dict = {}
    for cell in np.asarray(cells, dtype=int).reshape(-1, 4):
        step, register, a, b = (int(x) for x in cell)
        group = grouped.setdefault(register, [])
        run = group[-1] if group else None
        if run is None or step != run["to"] + 1:
            run = {"from": step, "to": step, "values": [[step, step, a, b]]}
            group.append(run)
        else:
            run["to"] = step
            seg = run["values"][-1]
            if seg[2] == a and seg[3] == b:
                seg[1] = step
            else:
                seg = [step, step, a, b]
                run["values"].append(seg)
    out = []
    for register in sorted(grouped):
        group = grouped[register]
        out.append(
            {
                "reg": register,
                "name": names[register],
                "from": group[0]["from"],
                "to": group[-1]["to"],
                "cells": sum(run["to"] - run["from"] + 1 for run in group),
                "runs": group,
            }
        )
    return out


@dataclass(frozen=True)
class Verdict:
    """A renderer grid against a reference grid, under one criterion.

    Onset: ``onset``/``whole``/``cells``/``starts``/``played``; lead:
    ``lead``/``first``/``first_reg``/``nbad``/``regs`` and the ``min_*``
    family. ``ok`` is the criterion's own agreement.
    """

    n: int = 0
    unit: str = "frame"
    ok: bool = False
    played: int | None = None
    starts: tuple = ()
    onset: tuple | None = None
    whole: tuple | None = None
    cells: tuple = ()
    lead: int | None = None
    first: int | None = None
    first_reg: int | None = None
    nbad: int | None = None
    regs: tuple = ()
    min_lead: int | None = None
    min_bad: int | None = None
    min_first: int | None = None
    min_regs: tuple = ()

    def line(self, names=REG_NAMES) -> str:
        """One line: verified, or the first mismatch under the criterion."""
        unit = self.unit
        if self.ok:
            return f"verified: {self.n} {unit}s play the reference grid"
        if self.onset is not None:
            whole = (
                "none"
                if self.whole is None
                else f"{unit} {self.whole[0]}, ${self.whole[1]:02X}"
            )
            return (
                f"mismatch: {unit} {self.onset[0]}, "
                f"register ${self.onset[1]:02X} (from {unit} 0: {whole})"
            )
        if self.first is not None:
            return (
                f"mismatch: {unit} {self.first}, "
                f"register ${self.first_reg:02X} {names[self.first_reg]} "
                f"(lead {self.lead}, {self.nbad} bad {unit}s)"
            )
        return f"mismatch: the render ends at {unit} {self.played} of {self.n}"

    def summary(self, names=REG_NAMES) -> dict:
        """The machine-readable diff, a flat dict for ``json.dumps``.

        The criterion's own fields, the differing registers' runs and both
        grids' values over them (``registers``), and a cell as
        ``[step, register, render, reference]``.
        """
        return dict(
            ok=self.ok,
            unit=self.unit,
            steps=self.n,
            onset=list(self.onset) if self.onset is not None else None,
            whole=list(self.whole) if self.whole is not None else None,
            starts=list(self.starts),
            played=self.played,
            ended=None if self.played is None else self.n - self.played,
            lead=self.lead,
            first=self.first,
            bad=self.nbad,
            regs=[names[r] for r in self.regs],
            min_lead=self.min_lead,
            min_bad=self.min_bad,
            min_first=self.min_first,
            min_regs=[names[r] for r in self.min_regs],
            registers=runs(self.cells, names),
            cells=[list(c) for c in self.cells],
        )


def verdict_onset(ours, starts, theirs, *, unit="tick"):
    """The onset-criterion :class:`Verdict` of render ``ours`` over reference ``theirs``.

    Voices' first notes in ``starts``; each voice counts from there, the 1-LSB
    carry is exempted, ``whole`` is the first mismatch from step 0, and a
    render ending early (``played < n``) is a mismatch in itself.
    """
    ours = np.asarray(ours, dtype=np.uint8)
    theirs = np.asarray(theirs, dtype=np.uint8)
    cells = cell_diff(ours, theirs, starts=starts, tol="carry")
    whole = cell_diff(ours, theirs, tol="carry")
    onset = (int(cells[0][0]), int(cells[0][1])) if cells.size else None
    wh = (int(whole[0][0]), int(whole[0][1])) if whole.size else None
    return Verdict(
        n=int(len(theirs)),
        unit=unit,
        ok=onset is None and int(len(ours)) == int(len(theirs)),
        played=int(len(ours)),
        starts=tuple(starts),
        onset=onset,
        whole=wh,
        cells=tuple(map(tuple, cells.tolist())) if cells.size else (),
    )


def verdict_lead(oracle, lib, *, max_lead=LEAD, unit="frame"):
    """The lead-criterion :class:`Verdict` of render ``lib`` against reference ``oracle``.

    ``lead`` is the latest first mismatch among the constant shifts up to
    ``max_lead``; the ``min_*`` family is the fewest bad steps (``lib`` must
    cover ``len(oracle) + max_lead``).
    """
    a, b = np.asarray(oracle, np.int16), np.asarray(lib, np.int16)
    n = min(len(a), len(b) - max_lead)
    if n < 1:
        raise ValueError(f"grids of {len(a)} and {len(b)} frames, shift {max_lead}")
    leads, ne, size, first, nbad = scan_leads(a[:n], b, max_lead)
    i = int(np.argmax(first))
    j = int(np.argmin(nbad))
    first_reg = int(np.flatnonzero(ne[i, first[i]])[0]) if nbad[i] else None
    return Verdict(
        n=n,
        unit=unit,
        ok=bool(nbad[i] == 0),
        lead=int(leads[i]),
        first=int(first[i]) if nbad[i] else None,
        first_reg=first_reg,
        nbad=int(nbad[i]),
        regs=tuple(int(r) for r in np.flatnonzero(ne[i, : size[i]].any(0))),
        min_lead=int(leads[j]),
        min_bad=int(nbad[j]),
        min_first=int(first[j]) if nbad[j] else None,
        min_regs=tuple(int(r) for r in np.flatnonzero(ne[j, : size[j]].any(0))),
    )


def _span(a: int, b: int) -> str:
    """A step or a run of steps."""
    return str(a) if a == b else f"{a}-{b}"


def _runs_lines(entry, shown=6, unit="frame", a="render", b="reference") -> list:
    """A differing register's block: step runs and both grids' values over them.

    The first ``shown`` value lines and the rest counted.
    """
    cells = entry["cells"]
    values = [line for run in entry["runs"] for line in run["values"]]
    steps = unit if entry["from"] == entry["to"] else unit + "s"
    lines = [
        f"  ${entry['reg']:02X} {entry['name']}: {cells} cell{'s' if cells != 1 else ''}, "
        f"{steps} {_span(entry['from'], entry['to'])}"
    ]
    for lo, hi, va, vb in values[:shown]:
        lines.append(f"    {_span(lo, hi)}: {a} {va:02x}, {b} {vb:02x}")
    if len(values) > shown:
        more = len(values) - shown
        lines.append(f"    ... {more} more value line{'s' if more != 1 else ''}")
    return lines


def runs_report(
    verdict,
    *,
    names=REG_NAMES,
    shown=6,
    unit=None,
    subject="render",
    a="render",
    b="reference",
) -> str:
    """The human-readable onset verdict, each differing register's runs and values."""
    unit = unit or verdict.unit
    if verdict.ok:
        return f"verified: {verdict.n} {unit}s play the {b} grid"
    if verdict.onset is not None:
        line = (
            f"mismatch: {unit} {verdict.onset[0]}, "
            f"register ${verdict.onset[1]:02X} {names[verdict.onset[1]]}"
        )
        if verdict.whole is not None:
            line += (
                f" (from {unit} 0: {unit} {verdict.whole[0]}, "
                f"${verdict.whole[1]:02X} {names[verdict.whole[1]]})"
            )
        lines = [line]
    else:
        lines = [
            f"mismatch: the {subject} ends at {unit} {verdict.played} of {verdict.n}"
        ]
    for entry in runs(verdict.cells, names):
        lines += _runs_lines(entry, shown, unit, a, b)
    return "\n".join(lines)


def lead_report(verdict, *, names=REG_NAMES) -> str:
    """The human-readable lead verdict: verified, or first mismatch with leads."""
    unit = verdict.unit
    if verdict.ok:
        return f"verified: {verdict.n} {unit}s agree at lead {verdict.lead}"
    reg = f"register ${verdict.first_reg:02X} {names[verdict.first_reg]}"
    return (
        f"differs: {unit} {verdict.first}, {reg} "
        f"(lead {verdict.lead}, {verdict.nbad} bad {unit}s; "
        f"fewest {verdict.min_bad} at lead {verdict.min_lead})"
    )


def window_report(oracle, lib, at, context=3):
    """Oracle (O) and render (L) values around ``at``, differing rows marked ``^^``."""
    lines = []
    for f in range(max(0, at - context), min(len(lib), at + context + 1)):
        o, l = oracle[f], lib[f]
        lines.append(f"{f:6d} O {' '.join(f'{x:02x}' for x in o)}")
        lines.append(f"{'':6} L {' '.join(f'{x:02x}' for x in l)}")
        if (o != l).any():
            lines.append(
                f"{'':8} " + " ".join("^^" if x != y else "  " for x, y in zip(o, l))
            )
    return "\n".join(lines)


def main(argv=None) -> int:
    """``python -m pysidtracker.gridcmp``: the diff of two grid files.

    Exit 0 on agreement, 1 on a difference. ``--mode lead`` scans the constant
    shifts up to ``--leads``; ``--mode onset`` compares from each ``--onset``
    first note with the 1-LSB carry exempted; ``--json`` the summary.
    """
    parser = argparse.ArgumentParser(
        description="Diff two register grids; exit 0 on agreement, 1 on a difference."
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("render", type=Path)
    parser.add_argument("--mode", choices=("lead", "onset"), default="lead")
    parser.add_argument(
        "--leads", type=int, default=LEAD, help="largest lead to scan (lead mode)"
    )
    parser.add_argument(
        "--onset",
        action="append",
        type=int,
        default=[],
        metavar="STEP",
        help="a voice's first-note step, one per voice in order (onset mode)",
    )
    parser.add_argument("--unit", default="frame", help="the step unit in reports")
    parser.add_argument(
        "--context",
        type=int,
        default=3,
        help="rows either side of the first difference",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        nargs="?",
        const="-",
        default=None,
        metavar="PATH",
        help="write the machine-readable diff to PATH (stdout with a dash)",
    )
    args = parser.parse_args(argv)
    ref = read_grid(args.reference)
    n = len(ref)
    if args.mode == "lead":
        lib = read_grid(args.render, n + args.leads)
        verdict = verdict_lead(ref, lib, max_lead=args.leads, unit=args.unit)
    else:
        lib = read_grid(args.render, n)
        verdict = verdict_onset(lib, args.onset or None, ref, unit=args.unit)
    if args.json_path is not None:
        text = json.dumps(verdict.summary(), indent=2) + "\n"
        if args.json_path == "-":
            print(text, end="")
        else:
            path = Path(args.json_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return 0 if verdict.ok else 1
    if verdict.played is not None:
        parts = [runs_report(verdict, unit=args.unit)]
    else:
        parts = [lead_report(verdict)]
    if not verdict.ok and verdict.first is not None:
        parts.append(
            window_report(*aligned(ref, lib, verdict.lead), verdict.first, args.context)
        )
    print("\n".join(parts))
    return 0 if verdict.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
