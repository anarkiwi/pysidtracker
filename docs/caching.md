# Caching

Every download pysidtracker or a dependent suite makes lands under one cache
root, so one CI cache persists it all and nothing is fetched twice.

## Layout

The root is `$PYSID_TUNECACHE`, else `<root>/.tunecache` (`default_tune_cache`).
`default_cache(kind)` names each subdirectory:

| path | contents | written by |
|---|---|---|
| `.tunecache/<HVSC relpath>` | HVSC `.sid` tunes | `fetch_tune`, `resolve_tune`, the `hvsc` fixture, `pysid-tune-fetch` |
| `.tunecache/DOCUMENTS/Songlengths.md5` | HVSC song lengths | `song_frames` |
| `.tunecache/native/` | CSDB release files + `SHA256SUMS.txt` | `fetch_csdb`, `pysid-fixtures fetch`, the `native` fixture |
| `.tunecache/archives/` | SHA-256-pinned downloads and archive members | `fetch_verified`, `fetch_archive_member` |
| `.tunecache/disks/` | `.d64` images | `fetch_disk`, `fetch_prgs`, the `disk_prgs` fixture |
| `.tunecache/roms/` | C64 ROMs | `roms`, the `c64_roms` fixture |
| `.tunecache/csv/` | sidtrace oracle traces | `oracle_grid`, `assert_oracle_match`, `make_oracle_fixtures`, the `sidtrace_oracle` fixture |

Every fetch defaults its `cache_dir` to its row; pass one only to isolate a
test. `$HVSC` (a local HVSC tree) and `$C64_ROMS` are read-only sources checked
before the cache, not cache locations.

## Downloads

All fetches share one path (`testing._fetch`): transient failures and empty
`200` bodies retry with backoff, and requests to a `HOST_INTERVAL` host (CSDB)
are serialised machine-wide, spaced by its interval, so parallel test workers
never trip CSDB's throttling. A pinned file downloads once under a lock beside
it; `fetch_csdb` holds the native cache's lock.

## CI

The `anarkiwi/pysidtracker/.github/actions/cache` composite action restores and
saves `.tunecache` (oracle traces apart, keyed on the sidtrace image digest) and
optionally pre-fetches before the suite:

```yaml
- run: pip install -e ".[dev]"
- uses: anarkiwi/pysidtracker/.github/actions/cache@v0.30.0
  with:
    key: ${{ hashFiles('tests/fixtures/*.txt', 'tests/test_oracle_hvsc.py') }}
    sidtrace: "true"                     # oracle jobs only
    tunes: tests                         # pysid-tune-fetch --tests tests
    native: tests/fixtures/native.txt    # pysid-fixtures fetch <manifest>
- run: pytest -n auto
```

`key` hashes the files that name what the suite fetches; a change restores the
previous cache and saves a superset. Pin the action to a pysidtracker release
tag, which dependabot's `github-actions` ecosystem keeps current.
