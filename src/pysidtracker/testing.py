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
from typing import Iterable, List, Optional, Tuple

from .d64 import D64File, read_d64
from .errors import SidError
from .oracle import (
    SIDTRACE_IMAGE,
    aligned_match,
    cached_sidtrace,
    register_grid,
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
    :mod:`pysidtracker.pytest_plugin` fixtures and the ``pysid-tune-fetch`` CLI
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


def _is_relpath(text: str) -> bool:
    return text.endswith(".sid") and "/" in text.strip("/")


def gather_tune_relpaths(paths: Iterable) -> List[str]:
    """Every ``*.sid`` relpath named by the files under ``paths``.

    A relpath has a directory (HVSC tunes all do), so a bare ``".sid"`` f-string
    fragment is not one. A Python file is parsed with :mod:`ast` (no import) for every
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


def oracle_grid(
    tune_path,
    *,
    oracle_cache=None,
    seconds: int = 60,
    frames=None,
    image: str = SIDTRACE_IMAGE,
    chip: int = 0,
    reg_count: int = 25,
    cycles_per_frame: Optional[int] = None,
    force: bool = False,
    song: Optional[int] = None,
):
    """Per-frame reference grid for ``tune_path`` from the sidtrace oracle.

    ``song`` is 1-based; ``None`` plays the header's start song. The CSV is
    cached at ``oracle_cache/<stem>.csv.zst`` (default ``default_cache("csv")``) (``<stem>_<song>`` for a given
    ``song``) while present (``force`` re-renders). ``cycles_per_frame`` fixes
    the cadence. Returns the first ``frames`` rows (all when ``None``).
    """
    tune_path = Path(tune_path)
    stem = tune_path.stem if song is None else f"{tune_path.stem}_{song}"
    extra = {} if song is None else {"song": song}
    oracle_cache = oracle_cache if oracle_cache is not None else default_cache("csv")
    csv_path = Path(oracle_cache) / f"{stem}.csv.zst"
    rows = cached_sidtrace(
        tune_path, csv_path, force=force, seconds=seconds, image=image, **extra
    )
    grid = sidtrace_grid(
        rows,
        chip=chip,
        reg_count=reg_count,
        cycles_per_frame=cycles_per_frame,
    )
    return grid[:frames] if frames else grid


def assert_matches(
    data,
    make_player,
    reference,
    *,
    song: int = 1,
    max_lead: int = 0,
    name: str = "",
    report_dir=None,
) -> None:
    """Assert ``make_player(data)`` renders ``reference``; else fail with the explanation.

    The failure message is :meth:`~pysidtracker.whydiff.Divergence.text`: both sides'
    dataflow of the first diverging cell, the state it read chased to where it parted,
    and full traced histories; saved to ``report_dir/<name>.txt``/``.json`` when given.
    """
    from .whydiff import explain_divergence

    name = name or "tune"
    why = explain_divergence(
        data,
        make_player,
        reference,
        song=song,
        max_lead=max_lead,
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
    frames: int = 250,
    song: int = 1,
    seconds=None,
    max_lead: int = 0,
    local_env: str = "HVSC",
    report_dir=None,
) -> None:
    """:func:`assert_matches` of an HVSC tune against its sidtrace oracle grid.

    ``make_player(data)`` returns the library's
    :class:`~pysidtracker.player.MemPlayer` for the tune's bytes.
    """
    path = resolve_tune(relpath, cache_dir=hvsc_cache, local_env=local_env)
    if path is None:
        raise TuneFetchError(f"{relpath} unavailable (offline, not cached)")
    expected = oracle_grid(
        path,
        oracle_cache=oracle_cache,
        seconds=seconds if seconds is not None else frames // 50 + 2,
        frames=frames,
        song=song,
    )
    assert_matches(
        Path(path).read_bytes(),
        make_player,
        expected,
        song=song,
        max_lead=max_lead,
        name=relpath,
        report_dir=report_dir,
    )


def make_oracle_fixtures(
    tunes,
    *,
    hvsc_cache=None,
    oracle_cache=None,
    render=register_grid,
    frames: int = 250,
    seconds=None,
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
    sidtrace oracle grid (cached in ``oracle_cache``), and returns a zero-arg
    callable that renders the library's player and asserts a frame-exact match::

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
    oracle fails the test rather than hiding a regression. ``seconds`` defaults
    to ``frames`` at 50 Hz plus a two-second margin.
    """
    import pytest  # lazy: pytest is only a dev dependency

    secs = seconds if seconds is not None else frames // 50 + 2
    ids = list(tunes)

    @pytest.fixture(params=ids)
    def tune_id(request):
        return request.param

    @pytest.fixture
    def oracle_match(tune_id):  # pylint: disable=redefined-outer-name
        path = resolve_tune(tunes[tune_id], cache_dir=hvsc_cache, local_env=local_env)
        if path is None:
            raise TuneFetchError(f"tune {tune_id} unavailable (offline, not cached)")
        expected = oracle_grid(
            path,
            oracle_cache=oracle_cache,
            seconds=secs,
            frames=frames,
            image=image,
            chip=chip,
            reg_count=reg_count,
        )

        def _match():
            if player is not None:
                assert_matches(
                    Path(path).read_bytes(),
                    player,
                    expected,
                    max_lead=max_lead,
                    name=tunes[tune_id],
                    report_dir=report_dir,
                )
                return
            rendered = render(Path(path).read_bytes(), len(expected))
            assert aligned_match(
                expected, rendered, max_lead=max_lead
            ), f"tune {tune_id}: player render does not match the sidtrace oracle"

        return _match

    return tune_id, oracle_match


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
