# Shared lint: dead symbols and unresolved doc references

`black` and `pylint` miss unused module constants / re-exports, docstrings or
docs naming symbols that no longer exist, and tests comparing a window of a song.
`pysidtracker.lint` checks all three with the standard library only. Each format repo's `tests/test_lint.py` is:

```python
from pysidtracker.lint import check_project

def test_lint():
    assert check_project("src/mypkg", tests=["tests"], docs=["docs"]) == []
```

(or use the `lint_project` fixture from the pytest plugin, or run
`pysidtracker lint src/mypkg`).

* **Unused symbols**: a module-level function, class, constant or `__init__`
  re-export that nothing in the package or `tests` reads. `__all__` does not count
  as a use.
* **Unresolved references**: Sphinx class / func / meth / attr / data / const / exc
  roles (and dotted mod roles), plus double-backtick dotted names that start with
  the package name, in docstrings and `docs/*.md`. A reference
  resolves when it equals, or is a dotted suffix of, a name the package defines.
  References whose dotted head is external (numpy.ndarray), builtins, names
  imported by the docstring's module, and file names are skipped.

* **Truncated comparisons** (`truncated_comparisons`): in the `tests` files, a
  render or compare call given a hard-coded frame or second count -- a number
  literal, a name every assignment in the test files binds to one (a module
  constant, also when imported), or arithmetic of those:

  | call | length arguments |
  |---|---|
  | `render_grid` | 1st positional, `nframes=` |
  | `iter_frames` | 1st positional, `max_frames=` |
  | `register_grid` | 2nd positional, `nframes=` |
  | `oracle_grid`, `sidtrace_oracle` | `frames=`, `seconds=` |
  | `survey.oracle_rows` | 3rd and 5th positional, `seconds=`, `calls=` |
  | `decode_fidelity`, `assert_decode_fidelity`, `assert_matches`, `assert_oracle_match`, `make_oracle_fixtures`, `explain_divergence`, `fidelity_row`, `survey.compare_file`, `survey.file_cli` | `frames=` |

  Compare whole songs: `song_frames(tune, song)` ([oracle-testing.md](oracle-testing.md)).
  The one opt-out is `pysidtracker.testing.whole_song(n)`, which returns `n` and
  declares it the whole length of a synthetic test song, never a window of a
  longer one: `player.render_grid(whole_song(8))`. Grep `whole_song(` to review them.

Pass `whitelist=[...]` (`--whitelist ...`) for public API kept for downstream
users or external names named in prose (the first two checks).
