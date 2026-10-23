# CSDB native-format fixtures: provenance and use

`pysidtracker.fixtures` fetches native-format editor files (GoatTracker `.sng`,
DMCSID `.sid`, JCH, ...) hosted on [CSDB](https://csdb.dk) into the gitignored
fixture cache and indexes each by SHA-256. A parser that supports such a format
needs the real files to decode against, but those files are copyright works and
must never be committed. This is the shared form of the `csdb_fetch.py` helper
native-format audits previously copy-pasted into each repository.

## Licensing

The native-format files are **copyright** their authors and are **never
committed** — no bytes, no base64, no fixtures, no `dist/`. They are fetched on
demand into `.tunecache/native/` (gitignored with the rest of the shared fixture
cache; `$PYSID_NATIVECACHE` overrides, `$PYSID_TUNECACHE` sets the shared root)
and indexed by SHA-256 in `SHA256SUMS.txt`. A test names the file it needs by that
hash and resolves to the cached copy — or **skips** — when the cache is cold,
rather than failing a suite for a file it cannot obtain. This is the same
fetch-don't-commit contract the HVSC `.sid` and C64 ROM fixtures use.

## Layout

```
.tunecache/native/
  html/rel_<release>.html          scraped release page (resume cache)
  dl/<release>/<did>_<file>        every downloaded file
  media/<release>/<did>_<file>/... a pinned file found inside release media
  SHA256SUMS.txt                   "sha256  dl/<release>/<file>  <source-url>"
```

## API

- `fetch_csdb(release_ids, cache_dir, *, mirror, retries, force)` — scrape each
  release page (cached under `html/`), download every file to
  `dl/<release>/<did>_<file>`, and append its `sha256  path  url` to
  `SHA256SUMS.txt`. Cached pages/files are reused unless `force`; a missing file
  is reported and skipped, not fatal. Returns one `SumEntry` per file.
- `index_media(entries, wanted, cache_dir)` — walk those downloads through any
  zip/gzip, d64/d71/d81, t64 and p00 nesting (`iter_prgs`) and pin each file whose
  SHA-256 is in `wanted` under `media/`, indexed with its download's URL, so a
  manifest can name an editor file that ships on a release disk.
  `fixtures fetch` does this for every manifest hash the downloads lack.
- `fixture(sha256, cache_dir)` — the cached file for a SHA-256 (re-digested for
  integrity), or `None` when absent.
- `make_native_fixtures(natives, cache_dir)` — a pytest factory: parametrizes
  `native_id` over the ids and returns `native_file` (the cached path) or skips.
- `load_manifest(path)` / `write_manifest(entries, path)` /
  `manifest_from_cache(cache_dir)` — the committed manifest, one
  `release_id<TAB>name<TAB>sha256[<TAB>description]` per file. **Hashes only**,
  never content, so a library can commit it safely.
- The `native` pytest fixture: `native(sha256)` -> cached path or `None`;
  `native.available(sha256)` for a suite that skips.

## Committing a manifest

1. Populate the cache once, where you have network:
   `pysidtracker fixtures fetch <manifest>` (or `fetch_csdb([ids], cache)`).
2. Emit the manifest from the warm cache and commit it:
   `pysidtracker fixtures manifest --out native_fixtures.txt`.
3. In CI, warm the cache (fetch) then run the suite; `pysidtracker fixtures
   verify <manifest>` asserts every declared hash is present.

## CLI

```
pysidtracker fixtures fetch <manifest> [--cache DIR] [--force]
pysidtracker fixtures manifest [--cache DIR] [--out PATH]
pysidtracker fixtures verify <manifest> [--cache DIR]
```

`pysid-fixtures` is a thin alias for the same command.
