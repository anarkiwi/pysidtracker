"""CSDB native-format fixtures: fetch csdb.dk editor files into a hash-indexed cache.

GoatTracker/DMCSID/JCH files are copyright works, so never committed: they are
fetched on demand into a gitignored cache and indexed by SHA-256, so a test
resolves a file by hash -- and skips when the cache is cold.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, NamedTuple, Optional

from .errors import SidError
from .testing import _atomic_write, default_tune_cache

DEFAULT_CSDB = "https://csdb.dk"
USER_AGENT = "pysidtracker/native-fixture"
RATE_LIMIT = 0.5  # seconds between network requests (be polite to csdb.dk)
SUMS_NAME = "SHA256SUMS.txt"


class CsdbFetchError(SidError):
    """A CSDB release file could not be fetched (page missing, or a 404)."""


class SumEntry(NamedTuple):
    """One ``SHA256SUMS.txt`` row: content hash, cache-relative path, source URL."""

    sha256: str
    relpath: str
    url: str


class NativeFixture(NamedTuple):
    """One committed-manifest line: what a fixture is and its expected hash."""

    release_id: str
    name: str
    sha256: str
    description: str = ""


def default_native_cache(root=None) -> Path:
    """Cache dir: ``$PYSID_NATIVECACHE`` or ``<tune cache>/native`` (gitignored)."""
    env = os.environ.get("PYSID_NATIVECACHE")
    if env:
        return Path(env)
    return default_tune_cache(root) / "native"


def _hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _http_get(url: str, *, retries: int = 4, rate_limit: float = RATE_LIMIT):
    """GET ``url`` with a User-Agent and retries; return ``(final_url, bytes)``.

    A short ``rate_limit`` pause precedes the call to stay polite; a 404 and
    repeated failures raise :class:`CsdbFetchError`.
    """
    if rate_limit:
        time.sleep(rate_limit)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:  # nosec B310 (https)
                return r.geturl(), r.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise CsdbFetchError(f"{url}: not found") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        if attempt + 1 < retries:
            time.sleep(min(2**attempt, 5))
    raise CsdbFetchError(f"{url}: unreachable after {retries} attempts ({last})")


def _scrape_release(
    rid, cache_dir, *, mirror: str, retries: int, force: bool
) -> List[int]:
    """A release's ``download.php`` ids; its page is cached under ``html/``."""
    page = Path(cache_dir) / "html" / f"rel_{rid}.html"
    if page.exists() and not force:
        html = page.read_text("latin-1")
    else:
        _url, data = _http_get(f"{mirror}/release/?id={rid}", retries=retries)
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_bytes(data)
        html = data.decode("latin-1", "replace")
    return sorted({int(m) for m in re.findall(r"download\.php\?id=(\d+)", html)})


def _file_name(final_url: str, fallback: str) -> str:
    """Basename of the percent-decoded final URL's path; ``fallback`` if empty."""
    path = urllib.parse.urlparse(urllib.parse.unquote(final_url)).path
    return os.path.basename(path) or fallback


def fetch_csdb(
    release_ids,
    cache_dir,
    *,
    mirror: str = DEFAULT_CSDB,
    retries: int = 4,
    force: bool = False,
) -> List[SumEntry]:
    """Download every file of ``release_ids`` into ``cache_dir``; return entries.

    Cached pages and files are reused unless ``force`` (resume). Each file is
    written to ``dl/<rid>/<did>_<name>`` and indexed in ``SHA256SUMS.txt`` as
    ``sha256  relpath  url``; a missing file is reported and skipped, not fatal.

    Args:
        release_ids: CSDB release ids to fetch.
        cache_dir: cache root holding ``html/``, ``dl/`` and the sums file.
        mirror: CSDB base URL; retries: transient-failure attempts; force: refetch.
    """
    cache_dir = Path(cache_dir)
    sums = cache_dir / SUMS_NAME
    known = (
        set(sums.read_text(encoding="utf-8").splitlines()) if sums.exists() else set()
    )
    new_lines: List[str] = []
    entries: List[SumEntry] = []
    for rid in release_ids:
        rid = str(rid)
        try:
            dids = _scrape_release(
                rid, cache_dir, mirror=mirror, retries=retries, force=force
            )
        except CsdbFetchError as exc:
            print(f"release {rid}: FAILED ({exc})", file=sys.stderr)
            continue
        out = cache_dir / "dl" / rid
        print(f"release {rid}: {len(dids)} file(s)")
        for did in dids:
            url = f"{mirror}/release/download.php?id={did}"
            out.mkdir(parents=True, exist_ok=True)
            present = [p for p in out.iterdir() if p.name.startswith(f"{did}_")]
            if present and not force:
                target = present[0]
            else:
                try:
                    final_url, data = _http_get(url, retries=retries)
                except CsdbFetchError as exc:
                    print(f"  {rid}/{did}: FAILED ({exc})", file=sys.stderr)
                    continue
                target = out / f"{did}_{_file_name(final_url, str(did))}"
                _atomic_write(target, data)
                print(f"  {rid}/{target.name}")
            sha = _hex(target.read_bytes())
            rel = f"dl/{rid}/{target.name}"
            line = f"{sha}  {rel}  {url}"
            if line not in known:
                known.add(line)
                new_lines.append(line)
            entries.append(SumEntry(sha, rel, url))
    if new_lines:
        sums.parent.mkdir(parents=True, exist_ok=True)
        with sums.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(new_lines) + "\n")
    return entries


_SUMS_LINE = re.compile(
    r"^(?P<sha>[0-9a-fA-F]{64})\s+(?P<rel>.+?)\s+(?P<url>https?://\S+)\s*$"
)


def sums_path(cache_dir) -> Path:
    """The cache's ``SHA256SUMS.txt`` path."""
    return Path(cache_dir) / SUMS_NAME


def read_sums(cache_dir) -> List[SumEntry]:
    """Parse the cache's ``SHA256SUMS.txt`` into ``SumEntry`` rows (blank-safe)."""
    path = sums_path(cache_dir)
    if not path.exists():
        return []
    out: List[SumEntry] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = _SUMS_LINE.match(raw.strip())
        if m:
            out.append(SumEntry(m["sha"].lower(), m["rel"], m["url"]))
    return out


def fixture(sha256, cache_dir) -> Optional[Path]:
    """Path to the cached file with this SHA-256 (re-digested), else ``None``.

    Resolved through ``SHA256SUMS.txt``; the file is re-verified so a missing or
    corrupt entry reads as absent rather than serving bad bytes.
    """
    sha = str(sha256).strip().lower()
    for entry in read_sums(cache_dir):
        if entry.sha256 != sha:
            continue
        path = Path(cache_dir) / entry.relpath
        if path.is_file() and _hex(path.read_bytes()) == sha:
            return path
    return None


def load_manifest(path) -> List[NativeFixture]:
    """Parse a committed manifest into ``NativeFixture`` rows.

    A line is ``release_id<TAB>name<TAB>sha256[<TAB>description]``; ``#`` lines and
    blanks are ignored; a short line raises :class:`CsdbFetchError`.
    """
    out: List[NativeFixture] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise CsdbFetchError(
                f"manifest {path}: need release_id, name, sha256 ({raw!r})"
            )
        out.append(
            NativeFixture(
                parts[0].strip(),
                parts[1].strip(),
                parts[2].strip().lower(),
                "\t".join(parts[3:]).strip(),
            )
        )
    return out


def write_manifest(entries, path) -> Path:
    """Write ``entries`` as a committed manifest; return the path written."""
    path = Path(path)
    lines = [
        "# pysidtracker native fixtures (committed; hashes only, never the files)",
        "# release_id<TAB>name<TAB>sha256<TAB>description",
    ]
    lines += [f"{e.release_id}\t{e.name}\t{e.sha256}\t{e.description}" for e in entries]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def manifest_from_cache(cache_dir, descriptions=None) -> List[NativeFixture]:
    """Build committed-manifest rows from a warm cache's ``SHA256SUMS.txt``.

    The release id and file name are recovered from each ``dl/<rid>/<file>`` path
    (the ``<did>_`` prefix stripped); ``descriptions`` maps a hash to a description.
    """
    descriptions = descriptions or {}
    out: List[NativeFixture] = []
    for sha, rel, _url in read_sums(cache_dir):
        parts = rel.split("/")
        if len(parts) >= 3 and parts[0] == "dl":
            release_id, name = parts[1], re.sub(r"^\d+_", "", parts[-1])
        else:
            release_id, name = "", parts[-1] if parts else rel
        out.append(NativeFixture(release_id, name, sha, descriptions.get(sha, "")))
    return sorted(out, key=lambda e: (e.release_id, e.name))


def make_native_fixtures(natives, cache_dir, *, skip_if_missing: bool = True):
    """Return ``(native_id, native_file)`` pytest fixtures for a ``natives`` map.

    ``natives`` maps an id to a SHA-256. ``native_id`` is parametrized over the
    ids; ``native_file`` resolves via :func:`fixture`, skipping (or raising
    :class:`CsdbFetchError`) when the cache is cold.
    """
    import pytest  # lazy: pytest is only a dev dependency

    ids = list(natives)

    @pytest.fixture(params=ids)
    def native_id(request):
        return request.param

    @pytest.fixture
    def native_file(native_id):  # pylint: disable=redefined-outer-name
        path = fixture(natives[native_id], cache_dir)
        if path is None:
            if skip_if_missing:
                pytest.skip(f"native fixture {native_id} not cached (fixtures fetch)")
            raise CsdbFetchError(f"native fixture {native_id} not cached")
        return path

    return native_id, native_file


def add_fixtures_commands(parser) -> None:
    """Add the ``fetch`` / ``manifest`` / ``verify`` subcommands onto ``parser``."""

    def _cache(dest_parser):
        dest_parser.add_argument(
            "--cache",
            help="cache dir (default $PYSID_NATIVECACHE or <tune cache>/native)",
        )

    sub = parser.add_subparsers(dest="fixtures_command", required=True)

    fetch_p = sub.add_parser(
        "fetch", help="recreate the cache from a committed manifest"
    )
    fetch_p.add_argument("manifest", help="committed fixture manifest")
    fetch_p.add_argument(
        "--force", action="store_true", help="re-download cached files"
    )
    _cache(fetch_p)
    fetch_p.set_defaults(func=_cmd_fetch)

    man_p = sub.add_parser(
        "manifest", help="write a committed manifest from a warm cache"
    )
    man_p.add_argument(
        "--out", default="native_fixtures.txt", help="manifest path to write"
    )
    _cache(man_p)
    man_p.set_defaults(func=_cmd_manifest)

    ver_p = sub.add_parser("verify", help="check a manifest's fixtures are all cached")
    ver_p.add_argument("manifest", help="committed fixture manifest")
    _cache(ver_p)
    ver_p.set_defaults(func=_cmd_verify)


def build_parser() -> argparse.ArgumentParser:
    """The standalone ``pysidtracker fixtures`` parser."""
    parser = argparse.ArgumentParser(
        prog="pysidtracker fixtures",
        description="Manage the CSDB native-fixture cache.",
    )
    add_fixtures_commands(parser)
    return parser


def main(argv=None) -> int:
    """Entry point for the ``pysidtracker fixtures`` command."""
    args = build_parser().parse_args(argv)
    return args.func(args)


def _report(expected, cache) -> int:
    """Print which ``expected`` fixtures are missing; return 1 if any are."""
    missing = [e for e in expected if fixture(e.sha256, cache) is None]
    for e in missing:
        print(f"  missing {e.sha256[:12]}  {e.name}", file=sys.stderr)
    print(
        f"{len(expected) - len(missing)}/{len(expected)} expected fixture(s) in {cache}"
    )
    return 1 if missing else 0


def _cmd_fetch(args) -> int:
    cache = Path(args.cache) if args.cache else default_native_cache()
    entries = load_manifest(args.manifest)
    release_ids = sorted({e.release_id for e in entries if e.release_id})
    if not release_ids:
        print(f"{args.manifest} declares no release ids; nothing to fetch")
        return 0
    fetch_csdb(release_ids, cache, force=args.force)
    return _report(entries, cache)


def _cmd_manifest(args) -> int:
    cache = Path(args.cache) if args.cache else default_native_cache()
    print(f"wrote {write_manifest(manifest_from_cache(cache), args.out)}")
    return 0


def _cmd_verify(args) -> int:
    cache = Path(args.cache) if args.cache else default_native_cache()
    return _report(load_manifest(args.manifest), cache)


__all__ = [
    "CsdbFetchError",
    "NativeFixture",
    "SumEntry",
    "add_fixtures_commands",
    "build_parser",
    "default_native_cache",
    "fetch_csdb",
    "fixture",
    "load_manifest",
    "make_native_fixtures",
    "main",
    "manifest_from_cache",
    "read_sums",
    "sums_path",
    "write_manifest",
]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
