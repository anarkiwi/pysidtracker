"""Pytest plugin giving any ``py*`` parser's suite HVSC tunes, disk PRGs and ROMs.

Registered as a ``pytest11`` entry point, so merely installing ``pysidtracker``
exposes the :func:`hvsc`, :func:`sidtrace_oracle`, :func:`disk_prgs`,
:func:`c64_roms` and :func:`native` fixtures to every dependent test suite with
**no per-package conftest**. Each downstream
suite keeps only its *list* of HVSC relpaths / disk URLs (test data); all the
resolve -> local-tree / cache / mirror-fetch machinery lives in
:mod:`pysidtracker.testing`.

This is a standalone top-level module (not ``pysidtracker.pytest_plugin``): a
pytest ``pytest11`` entry point is imported at plugin-load time, before
``pytest-cov`` starts tracing, and importing a *submodule* would drag in the
whole ``pysidtracker`` package there and under-report its coverage. Keeping the
heavy imports lazy (inside the fixtures) leaves them to be traced normally when
the tests import ``pysidtracker`` themselves.

* :func:`hvsc` -- ``hvsc(relpath)`` -> cached local ``.sid`` path (local HVSC
  tree, else fetch-cache, else mirror download), raising ``TuneFetchError`` when
  genuinely unreachable (never a skip). ``hvsc.read(relpath)`` returns bytes.
* :func:`sidtrace_oracle` -- ``sidtrace_oracle(path, frames=N)`` -> the CSV-cached
  sidtrace register grid.
* :func:`disk_prgs` -- ``disk_prgs(url)`` -> the PRG files inside a cached ``.d64``
  (or zipped ``.d64``) disk image.
* :func:`c64_roms` -- ``c64_roms("kernal")`` -> the cached C64 ROM path,
  ``c64_roms.image()`` a flat C64 memory image (``MEM_SIZE``) with KERNAL + BASIC
  mapped, and ``c64_roms.available()`` for a suite that skips without them.
* :func:`native` -- ``native(sha256)`` -> the cached CSDB native-format file for
  that hash, or ``None`` (a suite skips); ``native.available(sha256)`` checks.

The cache directory defaults to ``<rootdir>/.tunecache`` (what GitHub
``actions/cache`` persists); ``$PYSID_TUNECACHE`` overrides it. Local HVSC trees
come from the env vars named by the ``hvsc_local_envs`` ini option (default
``HVSC``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``hvsc_local_envs`` ini option."""
    parser.addini(
        "hvsc_local_envs",
        "Env vars naming a local HVSC tree, checked before the fetch cache.",
        type="args",
        default=("HVSC",),
    )


def _cache_dir(config: pytest.Config) -> Path:
    from pysidtracker.testing import default_tune_cache

    return default_tune_cache(config.rootpath)


class HvscTunes:
    """Resolve HVSC relpaths to local files, fetching + caching on demand."""

    def __init__(self, cache_dir, local_envs):
        self._cache = Path(cache_dir)
        self._local_envs = tuple(local_envs)

    def path(self, relpath: str) -> Path:
        """Cached local path to ``relpath`` (raises if genuinely unreachable)."""
        from pysidtracker.testing import TuneFetchError, resolve_tune

        for env in self._local_envs:
            base = os.environ.get(env)
            if base:
                cand = Path(base) / relpath
                if cand.exists():
                    return cand
        resolved = resolve_tune(relpath, cache_dir=self._cache, local_env="")
        if resolved is None:
            raise TuneFetchError(f"{relpath}: unreachable (offline, not cached)")
        return Path(resolved)

    def read(self, relpath: str) -> bytes:
        """Raw bytes of ``relpath``."""
        return self.path(relpath).read_bytes()

    __call__ = path


class SidtraceOracle:
    """Session retriever for CSV-cached sidtrace register grids."""

    def __init__(self, cache_dir):
        self._csv = Path(cache_dir) / "csv"

    def grid(
        self,
        tune_path,
        *,
        frames,
        cycles_per_frame=None,
        seconds=None,
        reg_count: int = 25,
        chip: int = 0,
        force: bool = False,
    ):
        """The first ``frames`` rows of the sidtrace grid for ``tune_path``.

        ``cycles_per_frame`` frames at a fixed cadence (e.g. PAL) instead of the
        sidtrace auto-cadence. The CSV is cached; ``seconds`` defaults to
        ``frames`` at 50 Hz plus a two-second margin.
        """
        from pysidtracker.testing import oracle_grid

        secs = seconds if seconds is not None else frames // 50 + 2
        return oracle_grid(
            tune_path,
            oracle_cache=self._csv,
            seconds=secs,
            frames=frames,
            cycles_per_frame=cycles_per_frame,
            reg_count=reg_count,
            chip=chip,
            force=force,
        )

    __call__ = grid


class DiskFixtures:
    """Session retriever for PRG fixtures inside cached ``.d64`` disk images."""

    def __init__(self, cache_dir):
        self._cache = Path(cache_dir) / "disks"

    def prgs(self, url: str, *, member=None, name=None):
        """The PRG files (``[D64File, ...]``) from the disk image at ``url``."""
        from pysidtracker.testing import fetch_prgs

        return fetch_prgs(url, cache_dir=self._cache, member=member, name=name)

    __call__ = prgs


class NativeFixtures:
    """Session retriever for CSDB native-format files addressed by SHA-256."""

    def __init__(self, cache_dir):
        self._cache = Path(cache_dir)

    def file(self, sha256) -> Optional[Path]:
        """The cached file for ``sha256``, or ``None`` when the cache is cold."""
        from pysidtracker.fixtures import fixture

        return fixture(sha256, self._cache)

    def available(self, sha256: str) -> bool:
        """True when ``sha256`` is cached; a suite skips otherwise."""
        return self.file(sha256) is not None

    __call__ = file


class C64Roms:
    """Session retriever for the fetch-cached C64 KERNAL/BASIC/CHARGEN ROMs."""

    def __init__(self, cache_dir):
        self._cache = Path(cache_dir) / "roms"

    def path(self, name: str) -> Path:
        """Cached local path to the ``name`` ROM (raises when unreachable)."""
        from pysidtracker.roms import RomFetchError, resolve_rom

        resolved = resolve_rom(name, cache_dir=self._cache)
        if resolved is None:
            raise RomFetchError(f"{name}: unreachable (offline, not cached)")
        return Path(resolved)

    def available(self, names=None) -> bool:
        """True when every named ROM resolves; a suite skips on False."""
        from pysidtracker.roms import DEFAULT_ROMS, roms_available

        return roms_available(names or DEFAULT_ROMS, cache_dir=self._cache)

    def image(self, names=None) -> bytearray:
        """A flat C64 memory image (``MEM_SIZE``) with the named ROMs mapped.

        KERNAL + BASIC are mapped by default.
        """
        from pysidtracker.roms import DEFAULT_ROMS, rom_image

        return rom_image(names or DEFAULT_ROMS, cache_dir=self._cache)

    __call__ = path


@pytest.fixture(scope="session")
def hvsc(pytestconfig: pytest.Config) -> HvscTunes:
    """Session HVSC retriever: ``hvsc(relpath)`` -> cached local path."""
    return HvscTunes(_cache_dir(pytestconfig), pytestconfig.getini("hvsc_local_envs"))


@pytest.fixture(scope="session")
def sidtrace_oracle(pytestconfig: pytest.Config) -> SidtraceOracle:
    """Session sidtrace-oracle retriever: ``sidtrace_oracle(path, frames=N)``."""
    return SidtraceOracle(_cache_dir(pytestconfig))


@pytest.fixture(scope="session")
def disk_prgs(pytestconfig: pytest.Config) -> DiskFixtures:
    """Session PRG-fixture retriever: ``disk_prgs(url)`` -> cached ``[D64File]``."""
    return DiskFixtures(_cache_dir(pytestconfig))


@pytest.fixture(scope="session")
def native(pytestconfig: pytest.Config) -> NativeFixtures:
    """Session CSDB native-file retriever: ``native(sha256)`` -> cached path or None."""
    from pysidtracker.fixtures import default_native_cache

    return NativeFixtures(default_native_cache(pytestconfig.rootpath))


@pytest.fixture(scope="session")
def c64_roms(pytestconfig: pytest.Config) -> C64Roms:
    """Session C64 ROM retriever: ``c64_roms("kernal")``, ``.image()``, ``.available()``."""
    return C64Roms(_cache_dir(pytestconfig))
