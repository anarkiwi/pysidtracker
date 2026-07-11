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
* :func:`make_tune_fixtures` -- a pytest fixture factory for parametrized
  ``tune_id`` / ``tune_path`` fixtures (pytest is imported lazily, so importing
  this module without pytest still works).

Pure stdlib (plus an optional lazy pytest import), so it ships in the wheel.
"""

from __future__ import annotations

import argparse
import ast
import io
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

from .d64 import D64File, read_d64
from .errors import SidError
from .oracle import (
    SIDTRACE_IMAGE,
    aligned_match,
    read_sidtrace,
    register_grid,
    run_sidtrace,
    sidtrace_grid,
)

# Public HVSC mirror. Override with ``$HVSC_MIRROR``.
DEFAULT_MIRROR = "https://hvsc.brona.dk/HVSC/C64Music"


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


def gather_tune_relpaths(paths: Iterable) -> List[str]:
    """Every ``*.sid`` string literal in the Python files under ``paths``.

    Parses each file with :mod:`ast` (no import, no side effects) and collects
    every string constant ending in ``.sid`` -- so a suite declares the tunes it
    needs as plain relpath literals (in dicts/lists/constants) and the fetch CLI
    pre-caches exactly those, with no hand-maintained manifest.
    """
    files: List[Path] = []
    for entry in paths:
        path = Path(entry)
        files.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    rels = set()
    for file in files:
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.endswith(".sid"):
                    rels.add(node.value)
    return sorted(rels)


class TuneFetchError(SidError):
    """A tune could not be fetched (mirror unreachable or the tune not found)."""


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
    cache_dir,
    mirror: str = DEFAULT_MIRROR,
    retries: int = 4,
    force: bool = False,
) -> Path:
    """Fetch ``relpath`` from an HVSC mirror into ``cache_dir``; return its path.

    Returns the cached path unchanged when it already exists (unless ``force``).
    Otherwise downloads ``mirror``/``relpath`` (``$HVSC_MIRROR`` overrides
    ``mirror``) with a User-Agent, validating the PSID/RSID magic and retrying
    transient failures with exponential backoff. Raises :class:`TuneFetchError`
    on a genuine 404 or after ``retries`` attempts.
    """
    relpath = relpath.lstrip("/")
    dest = Path(cache_dir) / relpath
    if dest.exists() and not force:
        return dest
    url = f"{_mirror(mirror)}/{urllib.request.quote(relpath)}"
    req = urllib.request.Request(url, headers={"User-Agent": "pysidtracker/fetch"})
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # nosec B310 (https)
                data = resp.read()
            if not _is_sid(data):
                raise TuneFetchError(f"{relpath}: not a SID file (magic {data[:4]!r})")
            _atomic_write(dest, data)
            return dest
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise TuneFetchError(f"{relpath}: not found on mirror") from exc
            last_err = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
        if attempt + 1 < retries:
            time.sleep(min(2**attempt, 5))
    raise TuneFetchError(
        f"{relpath}: mirror unreachable after {retries} attempts ({last_err})"
    )


def _download(url: str, *, retries: int = 4) -> bytes:
    """Download ``url``, retrying transient failures with exponential backoff."""
    req = urllib.request.Request(url, headers={"User-Agent": "pysidtracker/fetch"})
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # nosec B310
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise TuneFetchError(f"{url}: not found") from exc
            last_err = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
        if attempt + 1 < retries:
            time.sleep(min(2**attempt, 5))
    raise TuneFetchError(f"{url}: unreachable after {retries} attempts ({last_err})")


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
    cache_dir,
    member: Optional[str] = None,
    name: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Fetch a ``.d64`` disk image into ``cache_dir``; return its cached path.

    ``url`` may be a raw ``.d64`` or a ``.zip`` containing one (``member`` selects
    the entry by filename suffix; with no ``member`` the first ``.d64`` is used).
    Cached like a tune -- returned unchanged when present unless ``force`` -- so a
    persisted CI cache avoids re-downloading. ``name`` overrides the cache filename.
    """
    cache_dir = Path(cache_dir)
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
    cache_dir,
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


def resolve_tune(relpath: str, *, cache_dir, local_env: str = "HVSC"):
    """Path to ``relpath``, or ``None`` if it is genuinely unreachable.

    Checks a local HVSC tree (``$HVSC`` by default, via ``local_env``) first,
    then ``cache_dir``, then fetches from the mirror. Returns ``None`` only when
    the tune is neither local nor cached and the fetch fails (offline).
    """
    relpath = relpath.lstrip("/")
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
    cache_dir,
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
    oracle_cache,
    seconds: int = 60,
    frames=None,
    image: str = SIDTRACE_IMAGE,
    chip: int = 0,
    reg_count: int = 25,
    cycles_per_frame: Optional[int] = None,
    force: bool = False,
):
    """Per-frame reference grid for ``tune_path`` from the sidtrace oracle.

    The oracle CSV is cached at ``oracle_cache/<stem>.csv.zst`` and reused on the
    next call (so a CI cache -- or a developer's local dir -- avoids re-running
    Docker). Pass ``force=True`` to re-render. ``cycles_per_frame`` frames at a
    fixed cadence (e.g. PAL) instead of the sidtrace auto-cadence. Returns the
    first ``frames`` rows (all rows when ``frames`` is ``None``).
    """
    tune_path = Path(tune_path)
    oracle_cache = Path(oracle_cache)
    csv_path = oracle_cache / f"{tune_path.stem}.csv.zst"
    if force or not csv_path.exists():
        run_sidtrace(tune_path, csv_path, seconds=seconds, image=image)
    grid = sidtrace_grid(
        read_sidtrace(csv_path),
        chip=chip,
        reg_count=reg_count,
        cycles_per_frame=cycles_per_frame,
    )
    return grid[:frames] if frames else grid


def make_oracle_fixtures(
    tunes,
    *,
    hvsc_cache,
    oracle_cache,
    render=register_grid,
    frames: int = 250,
    seconds=None,
    image: str = SIDTRACE_IMAGE,
    local_env: str = "HVSC",
    chip: int = 0,
    reg_count: int = 25,
    max_lead: int = 4,
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
    :func:`~pysidtracker.oracle.register_grid` (the base py65 renderer); a format
    package passes its own :class:`~pysidtracker.player.MemPlayer` renderer.

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
