"""Whole-song HVSC survey of a format's renderer against the sidplayfp oracle.

``find`` lists tunes by SIDId signature, ``run`` compares every subtune's oracle
and ``module:function`` renderer grids, ``scope`` renders them without the oracle,
``report`` tallies, ``changes`` lists moved verdicts, ``compare`` moves verdicts against a baseline, ``diff`` shows
frames, ``triage`` classifies first differences and renderer ``info`` exact rates,
``leads`` tabulates every alignment lead's mismatches, ``file`` compares native
files (bare PRGs wrapped as PSID) with their own player.
"""

import argparse
import csv
import functools
import hashlib
import importlib
import importlib.util
import json
import multiprocessing
import os
import re
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from tqdm import tqdm

from vagrant_moses import memmap as mm

from .disasm import entry_labels
from .errors import SidParseError
from .formats import render as format_render
from .jobs import deadline, results
from .rendercache import CachedRender
from .gridcmp import (  # the shared grid-diff core, re-exported for the survey API
    LEAD,
    NEAR,
    NREGS,
    aligned,
    classify,
    gated,
    read_grid,
    reg_names,
    scan_leads,
    verdict_fields,
    window_report,
    write_grid,
)
from .gridcmp import (  # pylint: disable=unused-import
    REG_NAMES,
    ROLES,
    compare,
    compare_grids,
    min_lead,
)
from .header import PSID_MAGIC, RSID_MAGIC, parse_sid_header, write_psid
from .image import SidImage
from .irqhost import TunePlayer
from .oracle import (
    DEFAULT_SECONDS,
    ShortTrace,
    SidtraceUnavailable,
    cached_sidtrace,
    play_seconds,
    read_sidtrace,
    sidtrace_cadence,
    render_seconds,
    sidtrace_calls,
    sidtrace_grid,
    sidtrace_image_id,
)
from .testing import decode_fidelity, song_frames
from .sidid import identify, sidid_signatures
from .variants import variants_cli

ORACLE_ARGS = ("-f8000",)
ORACLE_ROWS = ("raise", "entry", "rti")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
TRIAGE_FIELDS = ["key", "path", "player", "song", "first_diff", "when", "kind"]
TRIAGE_FIELDS += ["roles", "diffs", "min_lead", "min_bad", "min_first", "min_regs"]
LEAD_FIELDS = ["key", "path", "song", "frames", "lead", "first_diff", "bad", "regs"]
TUNE_FIELDS = ["path", "md5", "player", "songs", "clock", "lengths"]
IMAGE_WIDTH = 0x4000
MAX_OFFSET = 0x200
LENGTH_GUARD = 0x200
MATCH_FIELDS = ["native", "hvsc", "length", "ndiff", "offset"]
RESULT_FIELDS = [
    "key",
    "path",
    "player",
    "song",
    "seconds",
    "frames",
    "verdict",
    "lead",
    "first_diff",
    "diff_frames",
    "diff_regs",
    "min_lead",
    "min_bad",
    "min_first",
    "min_regs",
    "info",
    "oracle_gated",
    "library_gated",
    "error",
    "cache",
]
FILE_FIELDS = RESULT_FIELDS + ["oracle"]
FIDELITY_FIELDS = ["fidelity", "fidelity_first", "fidelity_regs", "fidelity_error"]
SCOPE_FIELDS = ["key", "path", "player", "song", "verdict", "info", "error"]
_FIND = {}


def songlengths(hvsc_root):
    """``md5 -> [seconds per song]`` from HVSC's ``DOCUMENTS/Songlengths.md5``."""
    out = {}
    text = (Path(hvsc_root) / "DOCUMENTS" / "Songlengths.md5").read_text("latin-1")
    for line in text.splitlines():
        md5, sep, lens = line.partition("=")
        if sep and len(md5) == 32:
            out[md5] = [
                int(m) * 60 + int(s) + int((ms or "0").ljust(3, "0")) / 1000
                for m, s, ms in re.findall(r"(\d+):(\d+)(?:\.(\d+))?", lens)
            ]
    return out


THREADS = {"on": False}
THREADS_HELP = "run on a thread pool in this process, not worker processes"


def pool(jobs, initializer=None, initargs=()):
    """A spawn-context process pool whose workers start with one BLAS thread.

    With ``THREADS["on"]`` (the ``--threads`` option) a thread pool in this process
    instead, its initializer run once: for GIL-free work, and read-only worker state.
    """
    if THREADS["on"]:
        if initializer is not None:
            initializer(*initargs)
        return ThreadPoolExecutor(jobs)
    for var in THREAD_VARS:
        os.environ.setdefault(var, "1")
    return ProcessPoolExecutor(
        jobs,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initializer,
        initargs=initargs,
    )


def _find_init(cfg, player_regex):
    """Worker state for :func:`find`: the signatures and player filter."""
    _FIND.update(
        sigs=sidid_signatures(cfg), player=re.compile(player_regex or "", re.I)
    )


def _find_one(path):
    data = Path(path).read_bytes()
    name = identify(data, _FIND["sigs"])
    if name is None or not _FIND["player"].search(name):
        return None
    hdr = parse_sid_header(data)
    clock = "NTSC" if (hdr.flags >> 2) & 3 == 2 else "PAL"
    return hashlib.md5(data).hexdigest(), name, hdr.songs, clock


def find(hvsc, out_csv, cfg=None, player_regex=None, jobs=None):
    """Write the tunes SIDId names a ``player_regex`` player; return player tallies.

    ``cfg`` defaults to the bundled ``sidid.cfg`` (see :mod:`pysidtracker.sidid`).
    """
    hvsc = Path(hvsc)
    lengths = songlengths(hvsc)
    paths = sorted(hvsc.rglob("*.sid"))
    tally = Counter()
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with (
        pool(jobs, _find_init, (cfg, player_regex)) as ex,
        open(out_csv, "w", newline="", encoding="utf-8") as f,
    ):
        w = csv.writer(f)
        w.writerow(TUNE_FIELDS)
        hits = ex.map(_find_one, paths, chunksize=64)
        for path, hit in zip(paths, tqdm(hits, total=len(paths), unit="sid")):
            if hit:
                md5, name, songs, clock = hit
                tally[name] += 1
                lens = " ".join(f"{x:g}" for x in lengths.get(md5, []))
                w.writerow([path.relative_to(hvsc), md5, name, songs, clock, lens])
    return tally


def oracle_frames(ocsv, seconds=0.0, clock="PAL", nframes=None):
    """``(nframes, grid)`` of a sidtrace CSV (or its rows) over ``seconds`` at ``clock``.

    ``nframes`` overrides the frame count derived from ``seconds``. Frames past the
    last write repeat it; :class:`~pysidtracker.oracle.ShortTrace` if the trace
    covers fewer than ``nframes`` play calls (:func:`~pysidtracker.oracle.sidtrace_calls`).
    """
    rows = read_sidtrace(ocsv) if isinstance(ocsv, (str, Path)) else ocsv
    cpf = sidtrace_cadence(rows) or mm.PAL.cycles_per_frame
    hz = mm.video(clock == "NTSC").clock_hz
    nframes = int(seconds * hz / cpf) if nframes is None else nframes
    have = sidtrace_calls(rows, cycles_per_frame=cpf)
    if have < nframes:
        raise ShortTrace(f"trace covers {have} of {nframes} play calls")
    grid = np.asarray(sidtrace_grid(rows, cycles_per_frame=cpf), np.int16)
    grid = grid.reshape(-1, NREGS) if grid.size else np.zeros((1, NREGS), np.int16)
    return nframes, grid[np.minimum(np.arange(nframes), len(grid) - 1)]


def _err(exc):
    return f"{type(exc).__name__}: {exc}"


def oracle_cache(work):
    """``work/oracle``, emptied when its traces came from another sidtrace setup.

    The stamp holds the image id, the arguments and the logged row kinds; it is
    replaced atomically, so concurrent callers each see a whole stamp.
    """
    cache = Path(work) / "oracle"
    cache.mkdir(parents=True, exist_ok=True)
    image = sidtrace_image_id()
    stamp = cache / "STAMP"
    want = json.dumps({"image": image, "args": ORACLE_ARGS, "rows": ORACLE_ROWS})
    if image is None or (stamp.exists() and stamp.read_text("utf-8") == want):
        return cache
    stale = list(cache.glob("*.csv.zst"))
    if stale:
        print(f"oracle: re-rendering {len(stale)} stale traces", file=sys.stderr)
    for path in stale:
        path.unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=cache, prefix=".STAMP.", delete=False
    ) as tmp:
        tmp.write(want)
    os.replace(tmp.name, stamp)
    return cache


def oracle_rows(sid_path, ocsv, seconds, song, calls=0):
    """The sidtrace rows of one subtune over ``seconds``, cached at ``ocsv`` and
    re-rendered while covering fewer than ``calls`` play calls."""
    return cached_sidtrace(
        sid_path,
        ocsv,
        calls=calls,
        seconds=render_seconds(seconds),
        song=song,
        user=f"{os.getuid()}:{os.getgid()}",
        extra_args=ORACLE_ARGS,
    )


def rendered(row, render, data, song, need, *, timeout=0, lcsv=None):
    """``(row, grid)``: ``render(data, song, need)``'s int16 grid with ``row`` given
    its ``info``, or ``(row, None)`` with a ``not_decompiled`` / ``library_error``
    verdict. The grid is cached at ``lcsv`` if given."""
    late = dict(
        row, verdict="library_error", error=f"TimeoutError: exceeded {timeout}s"
    )
    try:
        with deadline(timeout, late):
            lib, info = render(data, song, need)
        lib = np.asarray(lib, dtype=np.int16)
        if lib.ndim != 2 or lib.shape[1] != NREGS or len(lib) < need:
            raise ValueError(f"render grid shape {lib.shape}, need {need}")
        if lcsv is not None:
            write_grid(lib, lcsv)
    except SidParseError as e:
        return dict(row, verdict="not_decompiled", error=str(e)), None
    except Exception as e:  # pylint: disable=broad-except
        return dict(row, verdict="library_error", error=_err(e)), None
    return dict(row, info=json.dumps(info, sort_keys=True, default=str)), lib


def library_row(row, oracle, render, data, song, *, lead=LEAD, timeout=0, lcsv=None):
    """``row`` completed by ``render(data, song, len(oracle) + lead)`` against
    ``oracle``: its :func:`verdict_fields`, or a ``not_decompiled`` /
    ``library_error`` verdict. The render grid is cached at ``lcsv`` if given."""
    row, lib = rendered(
        row, render, data, song, len(oracle) + lead, timeout=timeout, lcsv=lcsv
    )
    if lib is None:
        return row
    return dict(
        row,
        **verdict_fields(scan_leads(oracle, lib, lead)),
        oracle_gated=gated(oracle),
        library_gated=gated(lib),
    )


def fidelity_row(row, data, rebuild, *, render=None, song=1, lead=0):
    """``row`` plus the ``FIDELITY_FIELDS`` verdict column of a decode-fidelity check.

    ``rebuild(data) -> bytes`` writes the image back from the decoded model;
    ``fidelity`` is ``exact``, ``differs`` (with ``fidelity_first`` frame and
    ``fidelity_regs``), ``not_decompiled`` or ``rebuild_error`` (see
    :func:`~pysidtracker.testing.decode_fidelity`).
    """
    try:
        verdict = decode_fidelity(
            data,
            rebuild(data),
            render=render,
            song=song,
            max_lead=lead,
        )
    except SidParseError as e:
        return dict(row, fidelity="not_decompiled", fidelity_error=str(e))
    except Exception as e:  # pylint: disable=broad-except
        return dict(row, fidelity="rebuild_error", fidelity_error=_err(e))
    if verdict.ok:
        return dict(row, fidelity="exact")
    return dict(
        row,
        fidelity="differs",
        fidelity_first=verdict.first,
        fidelity_regs=",".join(str(r) for r in verdict.regs),
    )


def _tune_row(tune, song, **fields):
    """The identifying fields of subtune ``song`` of a ``find`` CSV row."""
    key = f"{tune['md5']}_{song}"
    return dict(key=key, path=tune["path"], player=tune["player"], song=song, **fields)


def _job_row(job):
    """The identifying fields of a :func:`check` job's row."""
    _, tune, song, seconds, *_ = job
    return _tune_row(tune, song, seconds=seconds)


def _failed(job, error):
    """The row of a :func:`check` job whose worker raised or died."""
    return dict(_job_row(job), verdict="library_error", error=_err(error))


def check(job):
    """One subtune's oracle-vs-renderer comparison row (see ``RESULT_FIELDS``)."""
    sid_path, tune, song, seconds, work, render, timeout = job
    row = _job_row(job)
    key = row["key"]
    ocsv, lcsv = (Path(work) / d / f"{key}.csv.zst" for d in ("oracle", "library"))
    try:
        nframes, oracle = oracle_frames(
            oracle_rows(sid_path, ocsv, seconds, song), seconds, tune["clock"]
        )
        row["frames"] = nframes
    except (SidtraceUnavailable, OSError, ValueError) as e:
        return dict(row, verdict="oracle_error", error=_err(e))
    data = Path(sid_path).read_bytes()
    row = library_row(row, oracle, render, data, song, timeout=timeout, lcsv=lcsv)
    return dict(row, cache=getattr(render, "last", ""))


def as_sid(data, init=None, play=None):
    """``data`` as a PSID/RSID: a container as is, a bare PRG wrapped by
    :func:`~pysidtracker.header.write_psid` with ``init`` (default the PRG's
    load address) and ``play`` (default ``init + 3``)."""
    data = bytes(data)
    if data[:4] in (PSID_MAGIC, RSID_MAGIC):
        return data
    if len(data) < 3:
        raise SidParseError(
            f"a PRG needs a load address and code, got {len(data)} bytes"
        )
    init = int.from_bytes(data[:2], "little") if init is None else init
    play = init + 3 if play is None else play
    return write_psid(load=0, init=init, play=play & 0xFFFF, image=data)


def file_oracle(sid, song, nframes, work=None, sid_path=None):
    """``(seconds, grid, source)``: ``nframes`` oracle frames of 1-based ``song``.

    With ``work``, the sidtrace trace cached under ``work/oracle`` (rendered
    over the :func:`~pysidtracker.cadence.playroutine_cadence` duration of
    ``nframes`` play calls from ``sid_path``, or a copy of ``sid`` under
    ``work/sid``) is framed by its interrupt rows; without ``work``, or when
    sidtrace is unavailable, :class:`~pysidtracker.irqhost.TunePlayer` runs the
    player instead (jennings, or the interrupt host for play address 0).
    """
    if work is not None:
        md5 = hashlib.md5(sid).hexdigest()
        if sid_path is None:
            sid_path = Path(work) / "sid" / f"{md5}.sid"
            sid_path.parent.mkdir(parents=True, exist_ok=True)
            sid_path.write_bytes(sid)
        seconds = play_seconds(sid, nframes, song)
        ocsv = oracle_cache(work) / f"{md5}_{song}.csv.zst"
        try:
            rows = oracle_rows(sid_path, ocsv, seconds, song, nframes)
            return seconds, oracle_frames(rows, nframes=nframes)[1], "sidtrace"
        except SidtraceUnavailable:
            pass
    grid = np.asarray(TunePlayer(sid, song - 1).render_grid(nframes), np.int16)
    return "", grid, "jennings"


def compare_file(
    path_or_prg,
    render,
    *,
    frames=None,
    init=None,
    play=None,
    song=1,
    lead=LEAD,
    work=None,
    timeout=0,
):
    """A ``results.csv`` row (plus ``oracle``) of ``render`` against a file's own player.

    ``path_or_prg`` (a path, or the bytes) is a PSID/RSID or a bare PRG wrapped
    by :func:`as_sid`. The oracle (:func:`file_oracle`, ``work`` and the
    ``oracle`` column choosing sidtrace or jennings) gives ``frames`` frames, by
    default the song's whole length (:func:`~pysidtracker.testing.song_frames`);
    the ``render`` hook (as for ``run``) gets ``frames + lead`` and is scanned
    over leads up to ``lead`` (``verdict``, ``lead``, ``min_lead``, ``min_bad``...).
    """
    path = Path(path_or_prg) if isinstance(path_or_prg, (str, Path)) else None
    raw = path.read_bytes() if path else bytes(path_or_prg)
    row = dict(path=str(path or ""), song=song, frames=frames)
    try:
        sid = as_sid(raw, init, play)
        if frames is None:
            row["frames"] = frames = song_frames(sid, song)
        row.update(key=f"{hashlib.md5(sid).hexdigest()}_{song}", player=identify(sid))
        own = path if path and sid == raw else None
        row["seconds"], oracle, row["oracle"] = file_oracle(
            sid, song, frames, work, own
        )
    except Exception as e:  # pylint: disable=broad-except
        return dict(row, verdict="oracle_error", error=_err(e))
    return library_row(row, oracle, render, sid, song, lead=lead, timeout=timeout)


def load_tunes(path):
    """The rows of a ``find`` CSV."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def player_tunes(path, player=None):
    """The rows of a ``find`` CSV whose player matches the case-insensitive regex."""
    return [t for t in load_tunes(path) if re.search(player or "", t["player"], re.I)]


@functools.lru_cache(maxsize=None)
def load_render(spec):
    """The callable named by ``module:function`` or ``path.py[:function]``
    (default ``render``); a ``path.py`` is imported as its stem's module, its directory
    joining ``sys.path`` for its own imports."""
    target, _, name = spec.partition(":")
    if target.endswith(".py"):
        home = str(Path(target).resolve().parent)
        if home not in sys.path:
            sys.path.insert(0, home)
        loader = importlib.util.spec_from_file_location(Path(target).stem, target)
        module = importlib.util.module_from_spec(loader)
        sys.modules[loader.name] = module
        loader.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return getattr(module, name or "render")


def renderer(render=None, fmt=None):
    """The installed format ``fmt``'s renderer (``auto``: the one recognising
    each tune), else :func:`load_render` of spec ``render`` (None without either)."""
    if fmt:
        return functools.partial(format_render, fmt)
    return render and load_render(render)


def song_list(spec, count=1):
    """The 1-based subtunes ``spec`` names: ``all`` (``1..count``), or comma-joined
    ``N`` / ``N-M`` items."""
    spec = str(spec)
    if spec == "all":
        return list(range(1, count + 1))
    songs = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        songs.extend(range(int(lo), int(hi or lo) + 1))
    return songs


def file_songs(path):
    """The subtune count of a PSID/RSID file (1 for a bare PRG)."""
    raw = Path(path).read_bytes()
    return parse_sid_header(raw).songs if raw[:4] in (PSID_MAGIC, RSID_MAGIC) else 1


def _file_row(job):
    spec, path, kw = job
    return compare_file(
        path, load_render(spec) if isinstance(spec, str) else spec, **kw
    )


def _failed_file(job, error):
    """The row of a :func:`_file_row` job whose worker raised or died."""
    song = job[2].get("song", 1)
    return dict(path=str(job[1]), song=song, verdict="library_error", error=_err(error))


def file_cli(render, files, *, song=1, out=None, jobs=None, **kw):
    """Compare a renderer with each native file's own player (bare PRGs wrapped).

    ``render`` is a ``load_render`` spec, e.g. a library's ``oracle_hvsc.py``,
    or a picklable renderer (:func:`renderer`); ``song`` picks each file's
    subtunes (:func:`song_list`); ``kw`` goes to :func:`compare_file`. Writes
    ``run``'s ``results.csv`` columns plus ``oracle`` to ``out`` (default
    stdout), one row per subtune; returns verdict tallies.
    """
    if kw.get("work") is not None:
        oracle_cache(kw["work"])
    tally = Counter()
    with (
        open(out, "w", newline="", encoding="utf-8") if out else nullcontext(sys.stdout)
    ) as f:
        w = csv.DictWriter(f, FILE_FIELDS)
        w.writeheader()
        jobs_ = [
            (render, p, dict(kw, song=n))
            for p in files
            for n in song_list(song, file_songs(p) if str(song) == "all" else 1)
        ]
        rows = (
            map(_file_row, jobs_)
            if jobs == 1
            else results(_file_row, jobs_, jobs, _failed_file)
        )
        bar = tqdm(rows, total=len(jobs_), unit="song")
        for row in bar:
            w.writerow(row)
            tally[row["verdict"]] += 1
            bar.set_postfix(tally)
    return tally


def song_seconds(lengths, song):
    """The HVSC length of 1-based ``song`` among ``lengths`` (seconds per song),
    :data:`~pysidtracker.oracle.DEFAULT_SECONDS` when it lists none."""
    return lengths[song - 1] if 0 < song <= len(lengths) else DEFAULT_SECONDS


def jobs_for(hvsc, tunes, work, render, timeout=600):
    """One :func:`check` job per (tune, song), timed by its HVSC song length."""
    for tune in tunes:
        lens = [float(x) for x in tune["lengths"].split()]
        for song in range(1, int(tune["songs"]) + 1):
            secs = song_seconds(lens, song)
            yield Path(hvsc) / tune["path"], tune, song, secs, Path(
                work
            ), render, timeout


def run(
    hvsc,
    tunes_csv,
    work,
    render,
    *,
    player=None,
    limit=None,
    jobs=None,
    timeout=600,
):
    """Compare every subtune; write ``results.csv`` under ``work``, return tallies.

    A ``--format`` renderer reuses ``work/render``: a subtune whose decoded model,
    subtune and player code are unchanged since an earlier run (or ``snapshot build
    --work``) is not rendered again."""
    work = Path(work)
    oracle_cache(work)
    if isinstance(render, functools.partial) and render.func is format_render:
        render = CachedRender(render.args[0], work / "render")
    tunes = player_tunes(tunes_csv, player)
    jobs_ = list(jobs_for(hvsc, tunes[:limit], work, render, timeout))
    tally, seen = Counter(), Counter()
    with open(work / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, RESULT_FIELDS)
        w.writeheader()
        bar = tqdm(results(check, jobs_, jobs, _failed), total=len(jobs_), unit="song")
        for row in bar:
            w.writerow(row)
            tally[row["verdict"]] += 1
            seen[row.get("cache", "")] += 1
            bar.set_postfix(dict(tally, rendered=seen["miss"], reused=seen["hit"]))
    return tally


def scope_check(job):
    """One subtune's render-only row: ``ok`` when the renderer (a :func:`load_render`
    spec or a callable) returns the frames, else :func:`rendered`'s verdict."""
    sid_path, tune, song, frames, render, timeout = job
    render = load_render(render) if isinstance(render, str) else render
    data = Path(sid_path).read_bytes()
    row, lib = rendered(
        _tune_row(tune, song), render, data, song, frames, timeout=timeout
    )
    return row if lib is None else dict(row, verdict="ok")


def _failed_scope(job, error):
    """The row of a :func:`scope_check` job whose worker raised or died."""
    return dict(_tune_row(job[1], job[2]), verdict="library_error", error=_err(error))


def scope(
    hvsc,
    tunes_csv,
    render,
    *,
    out=None,
    player=None,
    limit=None,
    frames=1,
    jobs=None,
    timeout=600,
):
    """Render every subtune's first ``frames`` frames without the oracle; write
    ``SCOPE_FIELDS`` rows to ``out`` (default stdout), return verdict tallies."""
    tunes = player_tunes(tunes_csv, player)[:limit]
    jobs_ = [
        (Path(hvsc) / t["path"], t, song, frames, render, timeout)
        for t in tunes
        for song in range(1, int(t["songs"]) + 1)
    ]
    tally = Counter()
    with (
        open(out, "w", newline="", encoding="utf-8") if out else nullcontext(sys.stdout)
    ) as f:
        w = csv.DictWriter(f, SCOPE_FIELDS)
        w.writeheader()
        rows = results(scope_check, jobs_, jobs, _failed_scope)
        bar = tqdm(rows, total=len(jobs_), unit="song")
        for row in bar:
            w.writerow(row)
            tally[row["verdict"]] += 1
            bar.set_postfix(tally)
    return tally


def image_body(raw, width=IMAGE_WIDTH):
    """Comparable body: the loaded image rebased to the load address.

    Returns ``(body, length)``: ``body`` a uint8 vector of width ``width``
    (the first ``length`` bytes the image, the rest zero).
    """
    image = SidImage.from_bytes(raw)
    body = np.zeros(width, np.uint8)
    img = bytes(image.image)[:width]
    body[: len(img)] = np.frombuffer(img, np.uint8)
    return body, len(img)


def _bodies(raws, body):
    """Stacked bodies (one per input) and their lengths (``raws`` non-empty)."""
    vecs, lens = [], []
    for raw in raws:
        vec, length = body(raw)
        vecs.append(np.asarray(vec, np.uint8))
        lens.append(int(length))
    if not vecs:
        raise ValueError("no candidate images to match")
    return np.stack(vecs), np.array(lens, np.int64)


def _overlap(cands, clens, q, qlen, d):
    """Per-candidate ``(valid, equal)`` comparing ``q`` with each candidate
    rebased by ``d``, over the bytes both images hold at that offset."""
    width = cands.shape[1]
    if d >= 0:
        a = cands[:, d:]
        b = q[: a.shape[1]]
        m = np.minimum(qlen, clens - d)
    else:
        h = -d
        a = cands[:, : width - h]
        b = q[h : h + a.shape[1]]
        m = np.minimum(qlen - h, clens)
    m = np.maximum(m, 0)
    k = np.arange(a.shape[1])[None, :]
    return m, np.count_nonzero((a == b) & (k < m[:, None]), axis=1)


def match_images(
    candidates,
    queries,
    body=image_body,
    *,
    max_offset=MAX_OFFSET,
    length_guard=LENGTH_GUARD,
):
    """Closest candidate per query, over every rebase offset in
    ``-max_offset .. max_offset``.

    Every candidate and query container's bytes pass through ``body`` (default
    :func:`image_body`) into a fixed-width uint8 vector and its loaded length.
    At each offset (in ``|d|`` order) a candidate row is compared with the
    query over the bytes both images hold there; per row the offset with the
    most equal bytes wins (smallest ``|d|``, then fewest differing bytes on
    ties) and a row whose image length differs from the query's by more than
    ``length_guard`` bytes is penalised out of contention. One
    ``dict(candidate, length, ndiff, offset)`` is returned per query: the
    chosen candidate's index, the query image length, the differing bytes over
    that overlap at the winning offset, and the offset itself.
    """
    cands, clens = _bodies(candidates, body)
    width = cands.shape[1]
    bound = min(max_offset, width - 1)
    penalty = 3 * width + 1
    rows = []
    for qraw in tqdm(queries, desc="match", unit="query"):
        q, qlen = body(qraw)
        q, qlen = np.asarray(q, np.uint8), int(qlen)
        score = np.full(cands.shape[0], 2 * penalty + width, np.int64)
        off = np.zeros(cands.shape[0], np.int64)
        ndiff = np.zeros(cands.shape[0], np.int64)
        for d in sorted(range(-bound, bound + 1), key=abs):
            m, eq = _overlap(cands, clens, q, qlen, d)
            nd = m - eq
            s = (
                (np.abs(clens - qlen) > length_guard) * penalty
                + (m == 0) * penalty
                - 2 * eq
                + nd
            )
            better = s < score
            score[better] = s[better]
            off[better] = d
            ndiff[better] = nd[better]
        i = int(np.argmin(score))
        rows.append(
            dict(candidate=i, length=qlen, ndiff=int(ndiff[i]), offset=int(off[i]))
        )
    return rows


def match(
    hvsc,
    tunes_csv,
    natives,
    body=image_body,
    *,
    out=None,
    max_offset=MAX_OFFSET,
    length_guard=LENGTH_GUARD,
):
    """TSV (``native, hvsc, length, ndiff, offset``) of the closest HVSC tune
    per native file.

    The tunes of a ``find`` CSV and each native file are run through ``body``
    and compared with :func:`match_images`. The TSV goes to stdout unless
    ``out`` (a file path) is given.
    """
    hvsc = Path(hvsc)
    tunes = load_tunes(tunes_csv)
    rows = match_images(
        [(hvsc / t["path"]).read_bytes() for t in tunes],
        [Path(n).read_bytes() for n in natives],
        body,
        max_offset=max_offset,
        length_guard=length_guard,
    )
    lines = ["\t".join(MATCH_FIELDS)]
    for n, r in zip(natives, rows):
        lines.append(
            "\t".join(
                str(x)
                for x in (
                    n,
                    tunes[r["candidate"]]["path"],
                    r["length"],
                    r["ndiff"],
                    r["offset"],
                )
            )
        )
    text = "\n".join(lines) + "\n"
    if out is None:
        sys.stdout.write(text)
    else:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, "utf-8")
    return rows


def near_exact(rows, near=NEAR, top=20):
    """Print and return the ``differs`` rows with at most ``near`` bad frames at
    their ``min_lead`` (rows without ``min_bad``, from older surveys, are skipped).
    """
    hits = [
        r
        for r in rows
        if r["verdict"] == "differs" and r.get("min_bad") and int(r["min_bad"]) <= near
    ]
    print(f"near-exact (<= {near} bad frames at min_lead): {len(hits)}")
    groups = {}
    for r in hits:
        group = (r["min_bad"], r["min_first"], r["min_lead"], r["min_regs"])
        groups.setdefault(group, []).append(r["path"])
    for (nbad, first, lead, regs), paths in sorted(
        groups.items(), key=lambda kv: -len(kv[1])
    )[:top]:
        where = f"bad={nbad} first={first} lead={lead}"
        print(f"{len(paths):7d} {where:28} {regs:28} e.g. {', '.join(paths[:2])}")
    return hits


def report(results_csv, near=NEAR):
    """Print tallies by player/verdict, differing registers, near-exact subtunes
    and errors."""
    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    verdicts = Counter((r["player"], r["verdict"]) for r in rows)
    print(verdicts.most_common())
    differs = [r for r in rows if r["verdict"] == "differs"]
    print("diff regs:", Counter(r["diff_regs"] for r in differs).most_common(20))
    near_exact(rows, near)
    errs = Counter(r["error"][:70] for r in rows if r["error"])
    print("errors:", errs.most_common(20))
    return verdicts


def read_results(path):
    """The rows of a ``results.csv`` by ``key``."""
    with open(path, newline="", encoding="utf-8") as f:
        return {r["key"]: r for r in csv.DictReader(f)}


def tally(rows, keys, near=NEAR):
    """``(verdicts, bad)`` over ``keys`` of ``rows`` (:func:`read_results`): the
    verdict counts (``missing`` for a key not in ``rows``) and the ``differs``
    counted by ``min_bad``, ``near + 1`` standing for more."""
    verdicts = Counter(rows[k]["verdict"] if k in rows else "missing" for k in keys)
    bad = Counter(
        min(int(rows[k].get("min_bad") or 0), near + 1)
        for k in keys
        if k in rows and rows[k]["verdict"] == "differs"
    )
    return verdicts, bad


def transitions(base, rows):
    """``(moves, lost)``: ``(before, after)`` verdict counts over the keys of ``rows``
    against ``base`` (``missing`` before for a new key), and the keys exact in
    ``base`` but not in ``rows``."""
    moves = Counter(
        (base[k]["verdict"] if k in base else "missing", r["verdict"])
        for k, r in rows.items()
    )
    lost = sorted(
        k
        for k, r in rows.items()
        if k in base and base[k]["verdict"] == "exact" and r["verdict"] != "exact"
    )
    return moves, lost


def changes(base, head):
    """``(key, before, after, first_before, first_after)`` of every subtune whose
    verdict or first-divergence frame differs between two ``results.csv`` row maps
    (:func:`read_results`); a key one side lacks reads ``missing``."""
    out = []
    for key in sorted(base.keys() | head.keys()):
        old, new = base.get(key, {}), head.get(key, {})
        move = (
            old.get("verdict", "missing"),
            new.get("verdict", "missing"),
            old.get("first_diff", ""),
            new.get("first_diff", ""),
        )
        if move[0] != move[1] or move[2] != move[3]:
            out.append((key, *move))
    return out


def changes_cli(base, head):
    """Print the subtunes whose verdict or first-divergence frame moved from the
    BASE to the HEAD ``results.csv``; returns the changes."""
    old, new = read_results(base), read_results(head)
    rows = changes(old, new)
    for key, before, after, first_before, first_after in rows:
        path = (new.get(key) or old.get(key))["path"]
        moved = f" first_diff {first_before or '-'} -> {first_after or '-'}"
        print(f"{key} {path}: {before} -> {after}{moved if before == after else ''}")
    print(f"{len(rows)} of {len(old.keys() | new.keys())} subtunes changed")
    return rows


def compare_results(results_csv, keylists=(), base=None, near=NEAR):
    """Print verdict tallies over key lists (files of ``<md5>_<song>`` keys) and,
    with ``base``, the verdict moves from that baseline ``results.csv`` and every
    subtune no longer exact; returns the moves."""
    rows = read_results(results_csv)
    moves = Counter()
    if base:
        moves, lost = transitions(read_results(base), rows)
        print("| before | after | subtunes |\n| --- | --- | ---: |")
        for (before, after), n in sorted(moves.items()):
            print(f"| {before} | {after} | {n:,} |")
        print(f"exact -> non-exact: {len(lost)}")
        for k in lost:
            print(f"  {k} {rows[k]['path']} {rows[k]['song']} {rows[k]['verdict']}")
    for path in keylists:
        keys = Path(path).read_text("utf-8").split()
        verdicts, bad = tally(rows, keys, near)
        print(f"{path}: {len(keys)} {dict(sorted(verdicts.items()))}")
        print(f"  differs by min_bad ({near + 1} = more): {dict(sorted(bad.items()))}")
    return moves


def diff(hvsc, tunes_csv, key, work, render, context=3):
    """Print oracle (O) and renderer (L) rows around one subtune's first difference."""
    work = Path(work)
    oracle_cache(work)
    md5, song = key.split("_")
    tune = next(t for t in load_tunes(tunes_csv) if t["md5"] == md5)
    job = next(j for j in jobs_for(hvsc, [tune], work, render) if j[2] == int(song))
    row = check(job)
    print(row)
    if row["verdict"] not in ("exact", "differs"):
        return row
    _, oracle = oracle_frames(work / "oracle" / f"{key}.csv.zst", job[3], tune["clock"])
    lib = read_grid(work / "library" / f"{key}.csv.zst", len(oracle) + LEAD)
    oracle, lib = aligned(oracle, lib, row["lead"])
    print(window_report(oracle, lib, row["first_diff"] or 0, context))
    return row


def _hex_addr(value):
    """A hex C64 address: ``1000``, ``0x1000``, or ``$1000``."""
    text = value[1:] if value[:1] == "$" else value
    try:
        return int(text, 16)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a hex address, got {value!r}"
        ) from exc


def _render_value(render, sid_path, song, frame, reg):
    """The library renderer's value of ``reg`` at ``frame`` (the side-by-side compare)."""
    lib, _info = render(Path(sid_path).read_bytes(), song, frame + LEAD + 1)
    grid = np.asarray(lib, dtype=np.int16)
    if grid.size == 0 or grid.ndim < 2:
        return None
    return int(grid[min(frame, len(grid) - 1)][reg])


def explain(
    hvsc,
    tunes_csv,
    key,
    work,
    *,
    frame,
    reg=None,
    watch=(),
    render=None,
    depth=8,
):
    """Run the tune's own 6502 play routine to ``frame`` and explain its SID writes.

    For each write in frame ``frame`` (or just ``reg``), prints the writing
    instruction with its disassembly, the load chain that produced the value, and the
    frame/cycle; with ``--watch`` also prints the chosen RAM cells per frame. When
    ``--reg`` and a ``render`` hook are given, the library's rendered value for that
    ``(frame, reg)`` appears next to the real player's write.
    """
    work = Path(work)
    md5, song = key.split("_")
    tune = next(t for t in load_tunes(tunes_csv) if t["md5"] == md5)
    sid_path = Path(hvsc) / tune["path"]
    image = SidImage.from_bytes(sid_path.read_bytes())
    from .provenance import explain as _explain
    from .provenance import format_explain, format_watch, trace_frames

    labels = entry_labels(image)
    ntsc = tune.get("clock", "PAL") == "NTSC"
    pro = trace_frames(image, frame + 1, watch=list(watch), labels=labels, ntsc=ntsc)
    if watch:
        print(format_watch(pro, list(watch)))
        print()
    print(format_explain(pro, frame, reg=reg, depth=depth))
    if render is not None and reg is not None:
        real = next((w.value for w in _explain(pro, frame, reg=reg)), None)
        libv = _render_value(render, sid_path, int(song), frame, int(reg))
        if libv is not None or real is not None:
            real_note = "" if real is None else f"   real ${real:02X}"
            lib_note = "n/a" if libv is None else f"${libv:02X}"
            print(f"library (frame {frame}, reg {reg}): {lib_note}{real_note}")
    return pro


def cached_grids(work, key, nframes):
    """``(oracle, lib)`` grids of a subtune's cached ``work`` traces."""
    _, oracle = oracle_frames(Path(work) / "oracle" / f"{key}.csv.zst", nframes=nframes)
    lib = read_grid(Path(work) / "library" / f"{key}.csv.zst", nframes + LEAD)
    return oracle, lib


def _diffs_at(oracle, lib, lead, frame):
    o, l = (g[frame] for g in aligned(oracle, lib, lead))
    return [(int(r), int(o[r]), int(l[r])) for r in np.flatnonzero(o != l)]


def _lead_rows(job):
    """``(summary, rows)``: one subtune's :func:`verdict_fields` and per-lead rows.

    Reuses the cached oracle trace and library grid, rendering only what is missing.
    """
    sid_path, tune, song, seconds, work, render, _ = job
    key = f"{tune['md5']}_{song}"
    ocsv, lcsv = (work / d / f"{key}.csv.zst" for d in ("oracle", "library"))
    nframes, oracle = oracle_frames(
        oracle_rows(sid_path, ocsv, seconds, song), seconds, tune["clock"]
    )
    if lcsv.exists():
        lib = read_grid(lcsv, nframes + LEAD)
    elif render is None:
        raise ValueError(f"{lcsv} is not cached and no --render was given")
    else:
        lib = np.asarray(
            render(Path(sid_path).read_bytes(), song, nframes + LEAD)[0], np.int16
        )
        write_grid(lib, lcsv)
    scan = scan_leads(oracle, lib)
    leads, ne, _, first, nbad = scan
    at_first = ne[np.arange(len(leads)), np.minimum(first, nframes - 1)]
    row = dict(key=key, path=tune["path"], song=song, frames=nframes)
    rows = [
        dict(row, lead=int(leads[i]), first_diff=int(first[i]) if nbad[i] else "")
        | dict(bad=int(nbad[i]), regs=reg_names(at_first[i]))
        for i in np.argsort(leads)
    ]
    return dict(row, **verdict_fields(scan)), rows


def leads(hvsc, tunes_csv, work, render=None, *, keys=(), verdicts=(), **kw):
    """Tabulate every lead's first mismatch, bad frames and first-frame registers.

    One key prints its table. Otherwise the keys, plus the ``results.csv``
    subtunes with a verdict in ``verdicts``, are scanned with ``kw['jobs']``
    processes into ``kw['out']`` (default ``work/leads.csv``).
    """
    work = Path(work)
    keys = list(keys)
    if verdicts:
        results = load_tunes(work / "results.csv")
        keys += [r["key"] for r in results if r["verdict"] in verdicts]
    wanted = {tuple(k.split("_")) for k in keys}
    tunes = [t for t in load_tunes(tunes_csv) if t["md5"] in {m for m, _ in wanted}]
    jobs_ = [
        j
        for j in jobs_for(hvsc, tunes, work, render)
        if (j[1]["md5"], str(j[2])) in wanted
    ]
    if len(keys) == 1 and not verdicts:
        summary, rows = _lead_rows(jobs_[0])
        print(
            f"{summary['key']} {summary['path']} song {summary['song']}: "
            f"{summary['frames']} frames, {summary['verdict']}, "
            f"lead {summary['lead']} (*, latest first mismatch), "
            f"min_lead {summary['min_lead']} (<, fewest bad frames)"
        )
        print(f"{'lead':>6} {'first':>6} {'bad':>6}  regs at first bad frame")
        for r in rows:
            mark = ("*" if r["lead"] == summary["lead"] else " ") + (
                "<" if r["lead"] == summary["min_lead"] else " "
            )
            print(
                f"{r['lead']:4d}{mark} {r['first_diff']:>6} {r['bad']:6d}  {r['regs']}"
            )
        return [summary]
    out = Path(kw.get("out") or work / "leads.csv")
    summaries = []
    with (
        pool(kw.get("jobs")) as ex,
        open(out, "w", newline="", encoding="utf-8") as f,
    ):
        w = csv.DictWriter(f, LEAD_FIELDS)
        w.writeheader()
        for summary, rows in tqdm(
            ex.map(_lead_rows, jobs_), total=len(jobs_), unit="song"
        ):
            w.writerows(rows)
            summaries.append(summary)
    moved = Counter(
        (s["lead"], s["min_lead"]) for s in summaries if s["lead"] != s["min_lead"]
    )
    print(f"{len(summaries)} subtunes, lead != min_lead:", moved.most_common(20))
    near_exact(summaries, kw.get("near", NEAR))
    return summaries


def first_difference(work, row):
    """``(frame, [(reg, oracle, renderer)])`` at a ``differs`` row's first mismatch."""
    first = int(row["first_diff"])
    grids = cached_grids(work, row["key"], int(row["frames"]))
    return first, _diffs_at(*grids, int(row["lead"]), first)


def _triage_one(args):
    work, row = args
    oracle, lib = cached_grids(work, row["key"], int(row["frames"]))
    frame = int(row["first_diff"])
    diffs = _diffs_at(oracle, lib, int(row["lead"]), frame)
    when, kind, roles = classify(frame, diffs, row)
    out = {k: row[k] for k in ("key", "path", "player", "song")}
    out.update(min_lead(scan_leads(oracle, lib)))
    return dict(out, first_diff=frame, when=when, kind=kind, roles=roles, diffs=diffs)


def info_rates(rows, max_values=20):
    """``{(key, value): (rendered, exact)}`` over each renderer ``info`` field."""
    tally = Counter()
    for r in rows:
        if r["verdict"] in ("exact", "differs"):
            for k, v in json.loads(r["info"] or "{}").items():
                tally[k, json.dumps(v), r["verdict"] == "exact"] += 1
    values = Counter(k for k, _ in {(k, v) for k, v, _ in tally})
    rates = {}
    for (k, v, exact), n in tally.items():
        if values[k] <= max_values:
            done, hit = rates.get((k, v), (0, 0))
            rates[k, v] = (done + n, hit + n * exact)
    return rates


def triage(work, jobs=None, top=15, max_values=20, near=NEAR):
    """Classify every ``differs`` first mismatch into ``triage.csv``; print tables."""
    work = Path(work)
    with open(work / "results.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    differs = [r for r in rows if r["verdict"] == "differs"]
    with (
        pool(jobs) as ex,
        open(work / "triage.csv", "w", newline="", encoding="utf-8") as f,
    ):
        w = csv.DictWriter(f, TRIAGE_FIELDS)
        w.writeheader()
        out = []
        hits = ex.map(_triage_one, ((work, r) for r in differs), chunksize=8)
        for t in tqdm(hits, total=len(differs), unit="song"):
            w.writerow(dict(t, diffs=json.dumps(t["diffs"])))
            out.append(t)
    print("verdicts:", Counter(r["verdict"] for r in rows).most_common())
    print("when:", Counter(t["when"] for t in out).most_common())
    print("kind:", Counter(t["kind"] for t in out).most_common(top))
    examples = {}
    for t in out:
        examples.setdefault((t["kind"], t["roles"]), []).append(t["path"])
    print("kind / roles at the first difference:")
    for (kind, roles), paths in sorted(examples.items(), key=lambda kv: -len(kv[1]))[
        :top
    ]:
        print(f"{len(paths):7d} {kind:14} {roles:28} e.g. {', '.join(paths[:2])}")
    near_exact([dict(t, verdict="differs") for t in out], near, top)
    rates = info_rates(rows, max_values)
    print("renderer info vs exact rate:")
    for (k, v), (n, hit) in sorted(rates.items(), key=lambda kv: -kv[1][0])[:top]:
        print(f"{n:7d} {100 * hit / n:5.1f}% exact  {k}={v}")
    return out, rates


def main(argv=None):
    """``python -m pysidtracker.survey``:
    ``find run scope report changes diff triage leads explain match variants file``."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("find", help=find.__doc__)
    p.add_argument("hvsc")
    p.add_argument("--sidid-cfg", help="sidid.cfg (default: the bundled one)")
    p.add_argument("--out", required=True)
    p.add_argument("--player", help="case-insensitive regex on the SIDId player name")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    p.add_argument("--threads", action="store_true", help=THREADS_HELP)
    for name, fn in (
        ("run", run),
        ("diff", diff),
        ("leads", leads),
        ("explain", explain),
    ):
        p = sub.add_parser(name, help=fn.__doc__)
        p.add_argument("hvsc")
        p.add_argument("tunes")
        p.add_argument("--work", required=True)
        p.add_argument(
            "--render",
            help="module:function renderer (leads: only for uncached grids)",
        )
        p.add_argument("--format", help="an installed format's renderer, or auto")
    sub.choices["diff"].add_argument("key")
    sub.choices["diff"].add_argument("--context", type=int, default=3)
    pe = sub.choices["explain"]
    pe.add_argument("key", help="<md5>_<song> subtune key")
    pe.add_argument("--frame", type=int, required=True, help="frame to explain")
    pe.add_argument("--reg", type=int, help="restrict to this SID register offset")
    pe.add_argument(
        "--watch",
        action="append",
        type=_hex_addr,
        default=[],
        help="a RAM cell ($XXXX) to watch per frame (repeatable)",
    )
    pe.add_argument("--depth", type=int, default=8, help="load-chain back-track depth")
    p = sub.add_parser("match", help=match.__doc__)
    p.add_argument("hvsc")
    p.add_argument("tunes")
    p.add_argument("natives", nargs="+", help="native files to find the HVSC tune of")
    p.add_argument(
        "--body",
        help="module:function body hook (default: the loaded image rebased "
        "to the load address)",
    )
    p.add_argument("--out", help="output TSV (default: stdout)")
    p.add_argument("--max-offset", type=int, default=MAX_OFFSET)
    pv = sub.add_parser("variants", help=variants_cli.__doc__)
    pv.add_argument(
        "paths", nargs="+", help=".sid/.prg tunes, or a find CSV (use --hvsc)"
    )
    pv.add_argument(
        "--ref", action="append", default=[], help="a reference build (repeatable)"
    )
    pv.add_argument("--hvsc", help="HVSC root resolving a find CSV's paths")
    pv.add_argument("--out", help="write the per-tune CSV here")
    pv.add_argument("--nframes", type=int, default=8)
    pv.add_argument("--threshold", type=float, default=0.8)
    p = sub.add_parser("file", help=file_cli.__doc__)
    p.add_argument(
        "render",
        nargs="?",
        help="renderer: oracle_hvsc.py[:function] or module:function",
    )
    p.add_argument("files", nargs="+", help=".sid or bare .prg native files")
    p.add_argument("--format", help="an installed format's renderer, or auto")
    p.add_argument("--frames", type=int, help="oracle frames (default: the whole song)")
    p.add_argument(
        "--song", default="1", help="1-based subtunes: N, N-M, comma lists, or all"
    )
    p.add_argument("--init", type=_hex_addr, help="bare PRG init (default: load)")
    p.add_argument("--play", type=_hex_addr, help="bare PRG play (default: init+3)")
    p.add_argument("--lead", type=int, default=LEAD, help="largest frame lead")
    p.add_argument("--work", help="sidtrace oracle cache dir (default: jennings)")
    p.add_argument("--out", help="output CSV (default: stdout)")
    p.add_argument(
        "--timeout", type=float, default=600, help="render seconds (0: none)"
    )
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    p = sub.add_parser("scope", help=scope.__doc__)
    p.add_argument("hvsc")
    p.add_argument("tunes")
    p.add_argument(
        "--render", help="renderer: oracle_hvsc.py[:function] or module:function"
    )
    p.add_argument("--format", help="an installed format's renderer, or auto")
    p.add_argument("--frames", type=int, default=1, help="frames rendered per subtune")
    p.add_argument("--out", help="output CSV (default: stdout)")
    for name in ("run", "scope"):
        p = sub.choices[name]
        p.add_argument(
            "--player", help="case-insensitive regex on the SIDId player name"
        )
        p.add_argument("--limit", type=int, help="only the first N tunes")
        p.add_argument(
            "--timeout", type=float, default=600, help="render seconds (0: none)"
        )
        p.add_argument("--jobs", type=int, default=os.cpu_count())
    p = sub.choices["leads"]
    p.add_argument("keys", nargs="*", help="<md5>_<song> subtune keys")
    p.add_argument(
        "--all", action="store_true", help="every exact/differs subtune in results.csv"
    )
    p.add_argument(
        "--verdict", action="append", default=[], help="results.csv verdict to scan"
    )
    p.add_argument("--out", help="leads CSV (default: WORK/leads.csv)")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    p = sub.add_parser("report", help=report.__doc__)
    p.add_argument("results")
    p = sub.add_parser("compare", help=compare_results.__doc__)
    p.add_argument("results")
    p.add_argument("keylists", nargs="*", help="files of <md5>_<song> keys")
    p.add_argument("--base", help="baseline results.csv to compare verdicts with")
    p = sub.add_parser("changes", help=changes_cli.__doc__)
    p.add_argument("base", help="baseline results.csv")
    p.add_argument("head", help="results.csv to compare with it")
    p = sub.add_parser("triage", help=triage.__doc__)
    p.add_argument("work")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    p.add_argument("--threads", action="store_true", help=THREADS_HELP)
    p.add_argument("--top", type=int, default=15)
    for name in ("leads", "report", "triage", "compare"):
        sub.choices[name].add_argument(
            "--near", type=int, default=NEAR, help="near-exact bad-frame limit"
        )
    a = ap.parse_args(argv)
    if a.cmd == "leads" and not (a.keys or a.all or a.verdict):
        ap.error("leads needs keys, --all or --verdict")
    if getattr(a, "format", None) and a.cmd == "file" and a.render:
        a.files, a.render = [a.render] + a.files, None
    if a.cmd in ("run", "diff", "file", "scope") and not (a.render or a.format):
        ap.error(f"{a.cmd} needs a renderer or --format")
    render = renderer(getattr(a, "render", None), getattr(a, "format", None))
    THREADS["on"] = getattr(a, "threads", False)
    if a.cmd == "find":
        out = find(a.hvsc, a.out, a.sidid_cfg, a.player, a.jobs)
    elif a.cmd == "run":
        out = run(
            a.hvsc,
            a.tunes,
            a.work,
            render,
            player=a.player,
            limit=a.limit,
            jobs=a.jobs,
            timeout=a.timeout,
        )
    elif a.cmd == "scope":
        out = scope(
            a.hvsc,
            a.tunes,
            render if a.format else a.render,
            out=a.out,
            player=a.player,
            limit=a.limit,
            frames=a.frames,
            jobs=a.jobs,
            timeout=a.timeout,
        )
        print(dict(out), file=sys.stderr)
    elif a.cmd == "diff":
        out = diff(a.hvsc, a.tunes, a.key, a.work, render, a.context)
    elif a.cmd == "explain":
        explain(
            a.hvsc,
            a.tunes,
            a.key,
            a.work,
            frame=a.frame,
            reg=a.reg,
            watch=a.watch,
            render=render,
            depth=a.depth,
        )
    elif a.cmd == "leads":
        leads(
            a.hvsc,
            a.tunes,
            a.work,
            render,
            keys=a.keys,
            verdicts=("exact", "differs") if a.all else a.verdict,
            out=a.out,
            jobs=a.jobs,
            near=a.near,
        )
    elif a.cmd == "triage":
        triage(a.work, a.jobs, a.top, near=a.near)
    elif a.cmd == "changes":
        changes_cli(a.base, a.head)
    elif a.cmd == "compare":
        compare_results(a.results, a.keylists, a.base, a.near)
    elif a.cmd == "match":
        match(
            a.hvsc,
            a.tunes,
            a.natives,
            load_render(a.body) if a.body else image_body,
            out=a.out,
            max_offset=a.max_offset,
        )
    elif a.cmd == "variants":
        out = variants_cli(
            a.paths,
            a.ref,
            hvsc=a.hvsc,
            out=a.out,
            nframes=a.nframes,
            threshold=a.threshold,
        )
    elif a.cmd == "file":
        out = file_cli(
            render if a.format else a.render,
            a.files,
            out=a.out,
            jobs=a.jobs,
            frames=a.frames,
            init=a.init,
            play=a.play,
            song=a.song,
            lead=a.lead,
            work=a.work,
            timeout=a.timeout,
        )
        print(dict(out), file=sys.stderr)
    else:
        out = report(a.results, a.near)
    if a.cmd in ("find", "run"):
        print(dict(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
