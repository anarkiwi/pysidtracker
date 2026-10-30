"""Content-addressed cache of a format's whole-song library renders.

A render is keyed by a digest of everything it depends on: the format, its decoded
model, the subtune and the version of the player code (:func:`player_version`). The
frame count is not part of the key: a render is causal, so a cached grid of at least
``nframes`` rows serves any shorter request (and a longer one replaces it). A change
that alters a few decodes therefore renders only those subtunes, and
``survey run`` and ``snapshot build`` reuse each other's grids.

Entries are written atomically, so concurrent runs may share one cache.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.resources
import importlib.util
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .errors import SidParseError
from .formats import SidFormat, find_format, recognize_format

HIT = "hit"
MISS = "miss"
CORE_PLAYBACK = (
    "cadence",
    "emu",
    "image",
    "irqhost",
    "kernalhost",
    "loader",
    "mos6502",
    "player",
    "registers",
)


def digest(value: Any) -> str:
    """BLAKE2b digest of ``value``'s bytes (``bytes``-like) or ``repr``."""
    raw = value if isinstance(value, (bytes, bytearray)) else repr(value).encode()
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


@functools.lru_cache(maxsize=None)
def _modules_digest(names: Tuple[str, ...]) -> str:
    out = hashlib.blake2b(digest_size=16)
    for name in names:
        if ":" in name:  # a package data file, "pkg:path/to/file"
            package, _, resource = name.partition(":")
            try:
                data = (
                    importlib.resources.files(package).joinpath(resource).read_bytes()
                )
            except (ImportError, ValueError, OSError):
                continue
            out.update(name.encode() + data)
            continue
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        origin = spec and spec.origin
        if origin and os.path.isfile(origin):
            out.update(name.encode() + Path(origin).read_bytes())
    return out.hexdigest()


def player_version(fmt: SidFormat) -> str:
    """A digest of the playback code ``fmt``'s renders depend on: the core playback
    modules, the module of ``fmt.player`` and ``fmt.player_modules`` (a ``"pkg:path"``
    entry is a package data file, such as a ``.play`` spec). Decoding code
    is left out, so a decoder change invalidates only the models it alters."""
    own = (getattr(fmt.player, "__module__", None), *fmt.player_modules)
    core = tuple(f"{__package__}.{name}" for name in CORE_PLAYBACK)
    return _modules_digest(tuple(sorted({*core, *filter(None, own)})))


class RenderCache:
    """The grids under ``root``, one ``.npz`` per render key."""

    def __init__(self, root) -> None:
        self.root = Path(root)
        self.last = ""

    def path(self, key: str) -> Path:
        """Where the render of ``key`` is stored."""
        return self.root / f"{key}.npz"

    def get(self, key: str, nframes: int) -> Optional[np.ndarray]:
        """The first ``nframes`` rows cached under ``key``; None if absent, short
        or unreadable."""
        try:
            with np.load(self.path(key)) as f:
                grid = f["grid"]
        except Exception:  # pylint: disable=broad-except
            return None
        return grid[:nframes] if len(grid) >= nframes else None

    def put(self, key: str, grid: np.ndarray) -> None:
        """Store ``grid`` under ``key``, replacing any entry atomically."""
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=self.root, prefix=".tmp.", suffix=".npz", delete=False
        ) as tmp:
            np.savez_compressed(tmp, grid=np.asarray(grid, np.int16))
        os.replace(tmp.name, self.path(key))

    def grid(self, fmt: SidFormat, model: Any, song: int, nframes: int) -> np.ndarray:
        """The int16 grid of ``fmt.player(model)`` over ``nframes``, rendered only
        on a miss; ``last`` becomes :data:`HIT` or :data:`MISS`."""
        key = digest(f"{fmt.name}|{digest(model)}|{song}|{player_version(fmt)}")
        grid = self.get(key, nframes)
        self.last = HIT if grid is not None else MISS
        if grid is None:
            grid = np.asarray(fmt.player(model).render_grid(nframes), np.int16)
            self.put(key, grid)
        return grid


class CachedRender:
    """A picklable survey renderer for format ``name`` (or ``auto``) that reuses
    the :class:`RenderCache` at ``root``; ``last`` is the latest call's status."""

    def __init__(self, name: str, root) -> None:
        self.name = name
        self.cache = RenderCache(root)

    @property
    def last(self) -> str:
        """:data:`HIT` or :data:`MISS` for the latest render."""
        return self.cache.last

    def __call__(
        self, data: bytes, song: int, nframes: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        fmt = find_format(data, self.name)
        if not recognize_format([fmt], data):
            raise SidParseError(f"format {fmt.name!r} does not recognise this tune")
        model = fmt.load(data, song - 1)
        grid = self.cache.grid(fmt, model, song, nframes)
        return grid, {"format": fmt.name, **fmt.info(model, data, song)}
