"""HVSC tune fetch/resolve helpers shared by the parsers' test suites.

HVSC ``.sid`` tunes are copyright works and are never committed. Each format
package's ``scripts/fetch_tunes.py`` / test conftest re-implemented the same
"resolve from a local HVSC tree, else a gitignored cache, else download from a
mirror" logic. This module is that shared core:

* :func:`fetch_tune` -- download one tune from an HVSC mirror into a cache,
  validating the PSID/RSID magic, retrying transient failures, writing
  atomically.
* :func:`resolve_tune` -- local HVSC tree (``$HVSC``) first, then the cache,
  then fetch; ``None`` only when genuinely unreachable.
* :func:`fetch_verified` / :func:`fetch_archive_member` -- a SHA-256-pinned
  download, or one SHA-256-pinned member of a pinned zip/tar release archive,
  cached while it verifies (copyright fixtures from an upstream release).
* :func:`default_cache` -- the one cache root (``$PYSID_TUNECACHE`` or
  ``.tunecache``) and its per-kind subdirectories, which every fetch defaults to
  (``docs/caching.md``).
* :func:`make_tune_fixtures` -- a pytest fixture factory for parametrized
  ``tune_id`` / ``tune_path`` fixtures (pytest is imported lazily, so importing
  this module without pytest still works).

Pure stdlib (plus an optional lazy pytest import), so it ships in the wheel.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import fcntl
import functools
import hashlib
import http.client
import io
import json
import os
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Mapping, Optional, Tuple

from .d64 import D64File, read_d64
from .errors import IncompleteComparison, SidError
from .formats import start_subtune
from .gridcmp import verdict_lead
from .oracle import (
    SIDTRACE_IMAGE,
    aligned_match,
    ShortTrace,
    cached_sidtrace,
    play_calls,
    play_seconds,
    register_grid,
    render_seconds,
    sidtrace_calls,
    sidtrace_grid,
)

# Public HVSC mirror. Override with ``$HVSC_MIRROR``.
DEFAULT_MIRROR = "https://hvsc.brona.dk/HVSC/C64Music"
# Retried download failures (an HTTPError other than 404 is a URLError).
_TRANSIENT = (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException)
_USER_AGENT = "pysidtracker/fetch"
# Hosts whose requests every process on the machine serialises, at least this
# many seconds apart: CSDB answers concurrent requests with empty bodies.
HOST_INTERVAL = {"csdb.dk": 0.5}
# Subdirectories of the cache root; HVSC tunes sit at the root by relpath.
CACHE_KINDS = ("archives", "csv", "disks", "native", "roms")


def default_tune_cache(root=None) -> Path:
    """The tune-cache directory: ``$PYSID_TUNECACHE`` or ``<root>/.tunecache``.

    ``root`` defaults to the current working directory. This is the location the
    :mod:`pysidtracker_pytest` fixtures and the ``pysid-tune-fetch`` CLI
    share, and what a GitHub ``actions/cache`` step persists across CI runs.
    """
    env = os.environ.get("PYSID_TUNECACHE")
    if env:
        return Path(env)
    return Path(root if root is not None else Path.cwd()) / ".tunecache"


def default_cache(kind: str = "", root=None) -> Path:
    """``kind``'s directory under :func:`default_tune_cache` (the root for ``""``).

    ``kind`` is one of :data:`CACHE_KINDS`: ``archives`` (pinned downloads),
    ``csv`` (sidtrace oracle traces), ``disks`` (disk images), ``native`` (CSDB
    editor files) and ``roms``. One root means one CI cache persists them all.
    """
    if kind and kind not in CACHE_KINDS:
        raise ValueError(f"cache kind {kind!r} not in {CACHE_KINDS}")
    return default_tune_cache(root) / kind


# HVSC's top-level directories: every tune relpath starts with one.
HVSC_ROOTS = ("DEMOS", "GAMES", "MUSICIANS")


def _is_relpath(text: str) -> bool:
    root, _, rest = text.lstrip("/").partition("/")
    return root in HVSC_ROOTS and rest.endswith(".sid")


def gather_tune_relpaths(paths: Iterable) -> List[str]:
    """Every ``*.sid`` relpath named by the files under ``paths``.

    A relpath starts with one of :data:`HVSC_ROOTS`, so a bare ``".sid"`` f-string
    fragment or a test's made-up path is not one. A Python file is parsed with :mod:`ast` (no import) for every
    such string constant; a ``.json`` file (a pathdiff manifest) for its cases'
    ``relpath`` fields. Any other file given directly is a manifest of words.
    """
    files: List[Path] = []
    for entry in paths:
        path = Path(entry)
        found = [f for g in ("*.py", "*.json") for f in path.rglob(g)]
        files.extend(sorted(found) if path.is_dir() else [path])
    rels = set()
    for file in files:
        text = file.read_text(encoding="utf-8")
        if file.suffix == ".json":
            cases = json.loads(text)
            cases = cases if isinstance(cases, list) else [cases]
            rels.update(
                c["relpath"]
                for c in cases
                if isinstance(c, dict) and _is_relpath(str(c.get("relpath", "")))
            )
        elif file.suffix != ".py":
            rels.update(word for word in text.split() if _is_relpath(word))
        else:
            tree = ast.parse(text, filename=str(file))
            rels.update(
                node.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and _is_relpath(node.value)
            )
    return sorted(rels)


class TuneFetchError(SidError):
    """A tune could not be fetched (mirror unreachable or the tune not found)."""


class ChecksumMismatch(TuneFetchError):
    """A pinned download's SHA-256 is not the pinned one: the content changed, so a
    test fails rather than skipping as for an unreachable download."""


def _mirror(mirror: str) -> str:
    return os.environ.get("HVSC_MIRROR", mirror).rstrip("/")


def _is_sid(data: bytes) -> bool:
    return data[:4] in (b"PSID", b"RSID")


def _atomic_write(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def fetch_tune(
    relpath: str,
    *,
    cache_dir=None,
    mirror: str = DEFAULT_MIRROR,
    retries: int = 4,
    force: bool = False,
) -> Path:
    """Fetch ``relpath`` from an HVSC mirror into ``cache_dir``; return its path.

    ``cache_dir`` defaults to :func:`default_tune_cache`. Returns the cached
    path unchanged when it already exists (unless ``force``). Otherwise downloads
    ``mirror``/``relpath`` (``$HVSC_MIRROR`` overrides ``mirror``) with
    :func:`_fetch`'s retries, validating the PSID/RSID magic. Raises
    :class:`TuneFetchError` on a genuine 404 or after ``retries`` attempts.
    """
    relpath = relpath.lstrip("/")
    dest = Path(cache_dir if cache_dir is not None else default_tune_cache()) / relpath
    if dest.exists() and not force:
        return dest
    url = f"{_mirror(mirror)}/{urllib.request.quote(relpath)}"
    data = _download(url, retries=retries)
    if not _is_sid(data):
        raise TuneFetchError(f"{relpath}: not a SID file (magic {data[:4]!r})")
    _atomic_write(dest, data)
    return dest


@contextlib.contextmanager
def _host_slot(url: str):
    """Hold ``url``'s host slot: requests to a :data:`HOST_INTERVAL` host from
    every process on this machine run one at a time, that interval apart."""
    host = (urllib.parse.urlparse(url).hostname or "").removeprefix("www.")
    interval = HOST_INTERVAL.get(host)
    if interval is None:
        yield
        return
    path = Path(tempfile.gettempdir()) / f"pysidtracker-{host}.lock"
    with open(path, "a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        lock.seek(0)
        try:
            last = float(lock.read() or 0)
        except ValueError:
            last = 0.0
        time.sleep(max(0.0, last + interval - time.time()))
        try:
            yield
        finally:
            lock.seek(0)
            lock.truncate()
            lock.write(repr(time.time()))


def _fetch(url: str, *, retries: int = 4, user_agent=_USER_AGENT) -> Tuple[str, bytes]:
    """``(final_url, body)`` of ``url``: the one download path every fetch uses.

    Transient failures retry with exponential backoff; an empty body is
    transient, as a throttling server (CSDB) answers 200 with none. Requests to
    a :data:`HOST_INTERVAL` host are serialised machine-wide (:func:`_host_slot`).
    """
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    last_err = None
    for attempt in range(retries):
        try:
            with (
                _host_slot(url),
                urllib.request.urlopen(req, timeout=60) as resp,  # nosec B310
            ):
                final, data = resp.geturl(), resp.read()
            if data:
                return final, data
            last_err = "an empty body"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise TuneFetchError(f"{url}: not found") from exc
            last_err = exc
        except _TRANSIENT as exc:
            last_err = exc
        if attempt + 1 < retries:
            time.sleep(min(2**attempt, 5))
    raise TuneFetchError(f"{url}: unreachable after {retries} attempts ({last_err})")


def _download(url: str, *, retries: int = 4) -> bytes:
    """The body of ``url`` (see :func:`_fetch`)."""
    return _fetch(url, retries=retries)[1]


def _sha256_file(path: Path) -> Optional[str]:
    """The SHA-256 hex digest of ``path``, or ``None`` when it is not a file."""
    if not path.is_file():
        return None
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _verified(data: bytes, sha256: str, what: str) -> bytes:
    """``data``, raising :class:`ChecksumMismatch` unless its SHA-256 is ``sha256``."""
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        raise ChecksumMismatch(f"{what}: SHA-256 {got}, expected {sha256}")
    return data


def fetch_verified(url: str, sha256: str, *, cache_dir=None, name=None) -> Path:
    """``url`` cached as ``cache_dir/name`` (default the URL's file name).

    ``cache_dir`` defaults to ``default_cache("archives")``.

    A cached copy is reused while its SHA-256 is ``sha256``; otherwise ``url`` is
    downloaded (retrying transient failures), verified and written atomically.
    Concurrent callers (xdist workers) serialise on a lock beside the file, so
    one downloads and the rest reuse its copy.
    """
    cache_dir = cache_dir if cache_dir is not None else default_cache("archives")
    dest = Path(cache_dir) / (
        name or PurePosixPath(urllib.parse.urlparse(url).path).name
    )
    if _sha256_file(dest) == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest.with_name(f".{dest.name}.lock"), "w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _sha256_file(dest) != sha256:
            _atomic_write(dest, _verified(_download(url), sha256, url))
    return dest


def fetch_archive_member(
    url: str,
    sha256: str,
    member: str,
    member_sha256: str,
    *,
    cache_dir=None,
    name=None,
    archive_name=None,
) -> Path:
    """``member`` of the zip or tar archive at ``url``, cached as ``cache_dir/name``.

    ``cache_dir`` defaults to ``default_cache("archives")`` and ``name`` to the
    member's file name. The archive is
    :func:`fetch_verified` (as ``archive_name``, for a URL whose path does not name
    the file) against ``sha256`` and the member against
    ``member_sha256``; a cached member that verifies is returned without the
    archive.
    """
    cache_dir = cache_dir if cache_dir is not None else default_cache("archives")
    dest = Path(cache_dir) / (name or PurePosixPath(member).name)
    if _sha256_file(dest) == member_sha256:
        return dest
    archive = fetch_verified(url, sha256, cache_dir=cache_dir, name=archive_name)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            data = zf.read(member)
    else:
        with tarfile.open(archive) as tf:
            src = tf.extractfile(member)
            if src is None:
                raise TuneFetchError(f"{member}: not a regular file in {archive.name}")
            data = src.read()
    _atomic_write(dest, _verified(data, member_sha256, member))
    return dest


def _d64_from_zip(blob: bytes, member: Optional[str]) -> bytes:
    """Extract a ``.d64`` from a zip ``blob`` (``member`` suffix, else the first)."""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = zf.namelist()
        if member is not None:
            cands = [n for n in names if n.endswith(member)]
        else:
            cands = [n for n in names if n.lower().endswith(".d64")]
        if not cands:
            raise TuneFetchError(f"no {member or '.d64'} in zip (contents: {names})")
        return zf.read(cands[0])


def fetch_disk(
    url: str,
    *,
    cache_dir=None,
    member: Optional[str] = None,
    name: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Fetch a ``.d64`` disk image into ``cache_dir``; return its cached path.

    ``url`` may be a raw ``.d64`` or a ``.zip`` containing one (``member`` selects
    the entry by filename suffix; with no ``member`` the first ``.d64`` is used).
    Cached like a tune -- returned unchanged when present unless ``force`` -- so a
    persisted CI cache avoids re-downloading. ``name`` overrides the cache filename;
    ``cache_dir`` defaults to ``default_cache("disks")``.
    """
    cache_dir = Path(cache_dir if cache_dir is not None else default_cache("disks"))
    is_zip = url.lower().endswith(".zip")
    if name is None:
        source = member if member else urllib.parse.urlparse(url).path
        name = Path(source).name
        if not name.lower().endswith(".d64"):
            name = f"{Path(name).stem or 'disk'}.d64"
    dest = cache_dir / name
    if dest.exists() and not force:
        return dest
    blob = _download(url)
    _atomic_write(dest, _d64_from_zip(blob, member) if is_zip else blob)
    return dest


def fetch_prgs(
    url: str,
    *,
    cache_dir=None,
    member: Optional[str] = None,
    name: Optional[str] = None,
    force: bool = False,
) -> List[D64File]:
    """Fetch a ``.d64`` (see :func:`fetch_disk`) and return its PRG files.

    Convenience over :func:`fetch_disk` + :func:`pysidtracker.d64.read_d64`, so a
    test suite gets ``[D64File(name, prg), ...]`` from a disk-image (or zipped
    disk-image) URL with one call, cached the same way tunes are.
    """
    disk = fetch_disk(url, cache_dir=cache_dir, member=member, name=name, force=force)
    return read_d64(disk.read_bytes())


def resolve_tune(relpath: str, *, cache_dir=None, local_env: str = "HVSC"):
    """Path to ``relpath``, or ``None`` if it is genuinely unreachable.

    Checks a local HVSC tree (``$HVSC`` by default, via ``local_env``) first,
    then ``cache_dir`` (default :func:`default_tune_cache`), then fetches from the
    mirror. Returns ``None`` only when the tune is neither local nor cached and
    the fetch fails (offline).
    """
    relpath = relpath.lstrip("/")
    cache_dir = cache_dir if cache_dir is not None else default_tune_cache()
    local = os.environ.get(local_env)
    if local:
        cand = Path(local) / relpath
        if cand.exists():
            return cand
    dest = Path(cache_dir) / relpath
    if dest.exists():
        return dest
    try:
        return fetch_tune(relpath, cache_dir=cache_dir)
    except TuneFetchError:
        return None


def make_tune_fixtures(
    tunes,
    cache_dir=None,
    *,
    local_env: str = "HVSC",
    skip_if_unavailable: bool = True,
):
    """Return ``(tune_id, tune_path)`` pytest fixtures for a ``tunes`` mapping.

    ``tunes`` maps a tune id to its HVSC relative path. The returned
    ``tune_id`` fixture is parametrized over the ids; ``tune_path`` resolves
    each via :func:`resolve_tune`, skipping (or raising
    :class:`TuneFetchError`) when a tune is unavailable. Assign them at module
    level in a conftest::

        tune_id, tune_path = make_tune_fixtures(TUNES, CACHE)

    pytest is imported here, lazily, so importing this module without pytest
    installed does not fail.
    """
    import pytest  # lazy: pytest is only a dev dependency

    ids = list(tunes)

    @pytest.fixture(params=ids)
    def tune_id(request):
        return request.param

    @pytest.fixture
    def tune_path(tune_id):  # pylint: disable=redefined-outer-name
        path = resolve_tune(tunes[tune_id], cache_dir=cache_dir, local_env=local_env)
        if path is None:
            if skip_if_unavailable:
                pytest.skip(f"tune {tune_id} unavailable (offline, not cached)")
            raise TuneFetchError(f"tune {tune_id} unavailable")
        return path

    return tune_id, tune_path


SONGLENGTHS = "DOCUMENTS/Songlengths.md5"


def songlengths_root(hvsc=None, local_env: str = "HVSC") -> Path:
    """An HVSC root holding :data:`SONGLENGTHS`: the local ``$HVSC`` tree (via
    ``local_env``) when it has one, else ``hvsc`` (default
    :func:`default_tune_cache`), fetched there from the mirror when missing."""
    local = os.environ.get(local_env) if local_env else None
    if local and (Path(local) / SONGLENGTHS).is_file():
        return Path(local)
    root = Path(hvsc if hvsc is not None else default_tune_cache())
    dest = root / SONGLENGTHS
    if not dest.is_file():
        _atomic_write(dest, _download(f"{_mirror(DEFAULT_MIRROR)}/{SONGLENGTHS}"))
    return root


@functools.lru_cache(maxsize=4)
def _songlengths(root: str, _mtime: int) -> Mapping[str, List[float]]:
    from .survey import songlengths  # survey imports this module

    return songlengths(root)


def song_length(
    tune, song: Optional[int] = 1, *, hvsc=None, local_env: str = "HVSC"
) -> float:
    """The seconds of 1-based ``song`` (``None``: the start song) of ``tune`` (a path
    or bytes): HVSC's ``Songlengths.md5`` entry for the file's MD5 (see
    :func:`songlengths_root` for ``hvsc``/``local_env``), else
    :data:`~pysidtracker.oracle.DEFAULT_SECONDS`."""
    from .survey import song_seconds

    data = (
        bytes(tune) if isinstance(tune, (bytes, bytearray)) else Path(tune).read_bytes()
    )
    if song is None:
        song = start_subtune(data) + 1
    root = songlengths_root(hvsc, local_env)
    table = _songlengths(str(root), (root / SONGLENGTHS).stat().st_mtime_ns)
    return song_seconds(table.get(hashlib.md5(data).hexdigest(), []), song)


def song_frames(
    tune, song: Optional[int] = 1, *, hvsc=None, local_env: str = "HVSC"
) -> int:
    """The play calls of 1-based ``song`` (``None``: the start song) of ``tune``
    (a path or bytes) over its whole :func:`song_length`, at the tune's
    :func:`~pysidtracker.cadence.playroutine_cadence`
    (:func:`~pysidtracker.oracle.play_calls`). Every render comparison covers this.
    """
    data = (
        bytes(tune) if isinstance(tune, (bytes, bytearray)) else Path(tune).read_bytes()
    )
    seconds = song_length(data, song, hvsc=hvsc, local_env=local_env)
    return play_calls(data, seconds, song)


def whole_song(length):
    """``length`` (frames or seconds), declared the whole length of a synthetic
    test song.

    The one way a test gives a render or comparison a literal length that
    :func:`pysidtracker.lint.truncated_comparisons` accepts: the number is the
    song's length, never a sampling window of a longer one.
    """
    return length


def require_whole(grid, frames: int, what: str = "reference") -> None:
    """Raise :class:`~pysidtracker.errors.IncompleteComparison` unless ``grid``
    covers ``frames`` (a song's whole length)."""
    if len(grid) < frames:
        raise IncompleteComparison(
            f"{what} covers {len(grid)} of the song's {frames} frames"
        )


def oracle_grid(
    tune_path,
    *,
    oracle_cache=None,
    seconds: Optional[float] = None,
    frames=None,
    image: str = SIDTRACE_IMAGE,
    chip: int = 0,
    reg_count: int = 25,
    cycles_per_frame: Optional[int] = None,
    force: bool = False,
    song: Optional[int] = None,
    hvsc=None,
    local_env: str = "HVSC",
):
    """Per-frame reference grid for ``tune_path`` from the sidtrace oracle.

    ``song`` is 1-based; ``None`` plays the header's start song. The CSV is
    cached at ``oracle_cache/<stem>.csv.zst`` (default ``default_cache("csv")``) (``<stem>_<song>`` for a given
    ``song``) while present (``force`` re-renders). ``cycles_per_frame`` fixes
    the cadence. Returns ``frames`` rows, by default the song's whole length
    (:func:`song_frames` with ``hvsc``/``local_env``), the last write repeated
    through the play calls after it. ``seconds`` defaults to the ``frames`` play
    calls at the tune's :func:`~pysidtracker.cadence.playroutine_cadence`; a cached
    trace covering fewer calls is re-rendered and
    :class:`~pysidtracker.oracle.ShortTrace` raised if it still falls short.
    """
    tune_path = Path(tune_path)
    stem = tune_path.stem if song is None else f"{tune_path.stem}_{song}"
    extra = {} if song is None else {"song": song}
    oracle_cache = oracle_cache if oracle_cache is not None else default_cache("csv")
    csv_path = Path(oracle_cache) / f"{stem}.csv.zst"
    if frames is None:
        frames = song_frames(tune_path, song, hvsc=hvsc, local_env=local_env)
    if seconds is None:
        seconds = render_seconds(play_seconds(tune_path, frames, song))
    rows = cached_sidtrace(
        tune_path,
        csv_path,
        calls=frames,
        force=force,
        seconds=seconds,
        image=image,
        **extra,
    )
    grid = sidtrace_grid(
        rows,
        chip=chip,
        reg_count=reg_count,
        cycles_per_frame=cycles_per_frame,
    )
    have = sidtrace_calls(rows, chip=chip, cycles_per_frame=cycles_per_frame)
    if have < frames:
        raise ShortTrace(
            f"{tune_path.name}: trace covers {have} of {frames} play calls"
        )
    return grid[:frames] + [grid[-1][:] for _ in range(frames - len(grid))]


def _register_render(data, nframes, song):
    return register_grid(data, nframes, subtune=song - 1)


def decode_fidelity(
    original,
    rebuilt,
    *,
    render=None,
    song: int = 1,
    max_lead: int = 0,
    hvsc=None,
    local_env: str = "HVSC",
):
    """The :class:`~pysidtracker.gridcmp.Verdict` of ``rebuilt``'s replay against ``original``'s.

    For formats whose playback emulates the tune's own code, the sidtrace oracle
    never exercises the decoded model: a wrong decode still renders exact. Rebuild
    a playable image from the decoded model (the format's writer, or the tracker's
    own packer under emulation) and pass both images here: each is rendered over
    1-based ``song``'s whole length (:func:`song_frames` of ``original``) and the
    grids compared, the rebuilt one allowed up to ``max_lead`` frames of silent
    lead-in. ``render`` is ``render(data, nframes, song) -> grid`` (default
    :func:`~pysidtracker.oracle.register_grid` on the image, so both sides run their
    own code).
    """
    render = render or _register_render
    frames = song_frames(original, song, hvsc=hvsc, local_env=local_env)
    want = render(original, frames, song)
    got = render(rebuilt, frames + max_lead, song)
    return verdict_lead(want, got, max_lead=max_lead, whole=frames)


def assert_decode_fidelity(
    data,
    rebuild,
    *,
    render=None,
    song: int = 1,
    max_lead: int = 0,
    name: str = "",
    hvsc=None,
    local_env: str = "HVSC",
) -> None:
    """Assert the model decoded from ``data`` rebuilds an image that replays as ``data`` does.

    ``rebuild(data) -> bytes`` decodes ``data`` and writes the playable image back
    out; see :func:`decode_fidelity` for the whole-song comparison and ``render``.
    """
    verdict = decode_fidelity(
        data,
        rebuild(data),
        render=render,
        song=song,
        max_lead=max_lead,
        hvsc=hvsc,
        local_env=local_env,
    )
    if not verdict.ok:
        raise AssertionError(
            f"{name or 'tune'}: rebuilt image replays differently from the "
            f"original: {verdict.line()}"
        )


def assert_matches(
    data,
    make_player,
    reference,
    *,
    song: int = 1,
    max_lead: int = 0,
    name: str = "",
    report_dir=None,
    hvsc=None,
    local_env: str = "HVSC",
) -> None:
    """Assert ``make_player(data)`` renders ``reference``; else fail with the explanation.

    ``reference`` (``None``: jennings' render) must cover the song's whole length
    (:func:`song_frames`), else :class:`~pysidtracker.errors.IncompleteComparison`.
    The failure message is :meth:`~pysidtracker.whydiff.Divergence.text`: both sides'
    dataflow of the first diverging cell, the state it read chased to where it parted,
    and full traced histories; saved to ``report_dir/<name>.txt``/``.json`` when given.
    """
    from .whydiff import explain_divergence

    name = name or "tune"
    frames = song_frames(data, song, hvsc=hvsc, local_env=local_env)
    if reference is not None:
        require_whole(reference, frames)
    why = explain_divergence(
        data,
        make_player,
        reference,
        song=song,
        max_lead=max_lead,
        frames=frames,
        name=name,
        state=True,
    )
    if why is None:
        return
    text = why.text()
    if report_dir is not None:
        out = Path(report_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = name.replace("/", "_").removesuffix(".sid")
        (out / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
        (out / f"{stem}.json").write_text(json.dumps(why.summary()), encoding="utf-8")
    raise AssertionError(f"{name}: player render diverges from the reference\n{text}")


def assert_oracle_match(
    relpath: str,
    make_player,
    *,
    hvsc_cache=None,
    oracle_cache=None,
    song: int = 1,
    max_lead: int = 0,
    local_env: str = "HVSC",
    report_dir=None,
) -> None:
    """:func:`assert_matches` of an HVSC tune's whole ``song`` against its sidtrace
    oracle grid.

    ``make_player(data)`` returns the library's
    :class:`~pysidtracker.player.MemPlayer` for the tune's bytes.
    """
    path = resolve_tune(relpath, cache_dir=hvsc_cache, local_env=local_env)
    if path is None:
        raise TuneFetchError(f"{relpath} unavailable (offline, not cached)")
    expected = oracle_grid(
        path,
        oracle_cache=oracle_cache,
        song=song,
        hvsc=hvsc_cache,
        local_env=local_env,
    )
    assert_matches(
        Path(path).read_bytes(),
        make_player,
        expected,
        song=song,
        max_lead=max_lead,
        name=relpath,
        report_dir=report_dir,
        hvsc=hvsc_cache,
        local_env=local_env,
    )


def make_oracle_fixtures(
    tunes,
    *,
    hvsc_cache=None,
    oracle_cache=None,
    render=register_grid,
    image: str = SIDTRACE_IMAGE,
    local_env: str = "HVSC",
    chip: int = 0,
    reg_count: int = 25,
    max_lead: int = 4,
    player=None,
    report_dir=None,
):
    """Return ``(tune_id, oracle_match)`` fixtures asserting a player == the oracle.

    ``tunes`` maps a tune id to its HVSC relative path. For each tune the
    ``oracle_match`` fixture resolves the ``.sid`` (local ``$HVSC`` tree, else
    ``hvsc_cache``, else download -- see :func:`resolve_tune`), renders the
    sidtrace oracle grid of the start song's whole length (:func:`song_frames`,
    cached in ``oracle_cache``), and returns a zero-arg callable that renders the
    library's player and asserts a frame-exact match::

        tune_id, oracle_match = make_oracle_fixtures(
            TUNES, hvsc_cache=HVSC, oracle_cache=CSV, render=my_render)

        def test_matches_oracle(oracle_match):
            oracle_match()

    ``render`` is ``render(tune_bytes, nframes) -> grid`` and defaults to
    :func:`~pysidtracker.oracle.register_grid` (the base jennings renderer); a format
    package passes its own :class:`~pysidtracker.player.MemPlayer` renderer, or
    ``player(tune_bytes) -> MemPlayer`` for a failure explained by :func:`assert_matches`.

    These tests are never skipped: an unavailable tune raises
    :class:`TuneFetchError` and a missing/failed oracle raises
    :class:`~pysidtracker.oracle.SidtraceUnavailable`, so a broken download or
    oracle fails the test rather than hiding a regression.
    """
    import pytest  # lazy: pytest is only a dev dependency

    ids = list(tunes)

    @pytest.fixture(params=ids)
    def tune_id(request):
        return request.param

    @pytest.fixture
    def oracle_match(tune_id):  # pylint: disable=redefined-outer-name
        path = resolve_tune(tunes[tune_id], cache_dir=hvsc_cache, local_env=local_env)
        if path is None:
            raise TuneFetchError(f"tune {tune_id} unavailable (offline, not cached)")
        data = Path(path).read_bytes()
        frames = song_frames(data, None, hvsc=hvsc_cache, local_env=local_env)
        expected = oracle_grid(
            path,
            oracle_cache=oracle_cache,
            frames=frames,
            image=image,
            chip=chip,
            reg_count=reg_count,
        )

        def _match():
            if player is not None:
                song = start_subtune(data) + 1
                assert_matches(
                    data,
                    player,
                    expected,
                    song=song,
                    max_lead=max_lead,
                    name=tunes[tune_id],
                    report_dir=report_dir,
                    hvsc=hvsc_cache,
                    local_env=local_env,
                )
                return
            rendered = render(data, frames + max_lead)
            require_whole(rendered, frames + max_lead, "render")
            assert aligned_match(
                expected, rendered, max_lead=max_lead
            ), f"tune {tune_id}: player render does not match the sidtrace oracle"

        return _match

    return tune_id, oracle_match


def corpus_params(tunes: Iterable, expected: Mapping[str, Optional[str]]) -> list:
    """One ``pytest.param(rel, expected.get(rel))`` per relative path of ``tunes``,
    its id the path; the message is the error a tune raises (None: it decodes),
    as :func:`~pysidtracker.snapshot.expected_errors` reads them."""
    import pytest  # lazy: pytest is only a dev dependency

    rels = (PurePosixPath(rel).as_posix() for rel in tunes)
    return [pytest.param(rel, expected.get(rel), id=rel) for rel in rels]


def check_outcome(load, path, expected: Optional[str], error_type: type = SidError):
    """``load(path)``, asserting it raises ``error_type`` with exactly the message
    ``expected``, or (``expected`` None) returns cleanly; the model, else None."""
    if expected is None:
        return load(path)
    try:
        load(path)
    except error_type as exc:
        assert str(exc) == expected, f"{path}: raised {exc!r}, not {expected!r}"
        return None
    raise AssertionError(f"{path}: decoded, expected {expected!r}")


def main(argv=None) -> int:
    """``pysid-tune-fetch``: pre-cache HVSC tunes into the shared tune cache.

    Fetches the given HVSC relpaths (and every ``*.sid`` literal found under any
    ``--tests`` path) from the mirror into the cache, so a CI job can populate a
    persisted ``actions/cache`` before the suite runs. Best-effort: an
    unreachable tune is warned about, not fatal (the test that needs it fails).
    """
    parser = argparse.ArgumentParser(
        prog="pysid-tune-fetch", description=main.__doc__.splitlines()[0]
    )
    parser.add_argument("relpaths", nargs="*", help="HVSC relative paths to fetch")
    parser.add_argument(
        "--tests",
        action="append",
        default=[],
        metavar="PATH",
        help="scan this dir/file for '*.sid' relpaths to fetch (repeatable)",
    )
    parser.add_argument(
        "--cache", help="cache dir (default: $PYSID_TUNECACHE or ./.tunecache)"
    )
    parser.add_argument("--force", action="store_true", help="re-download cached tunes")
    args = parser.parse_args(argv)

    cache = Path(args.cache) if args.cache else default_tune_cache()
    rels = set(args.relpaths)
    if args.tests:
        rels.update(gather_tune_relpaths(args.tests))
    got, missing = 0, []
    for rel in sorted(rels):
        try:
            fetch_tune(rel, cache_dir=cache, force=args.force)
            got += 1
        except TuneFetchError as exc:
            missing.append(rel)
            print(f"WARN: {rel}: {exc}", file=sys.stderr)
    print(f"tune cache: fetched/cached {got}/{len(rels)} tunes into {cache}")
    if missing:
        print(f"unreachable: {len(missing)} ({', '.join(missing)})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
