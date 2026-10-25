# Shared lint: dead symbols and unresolved doc references

`black` and `pylint` miss unused module constants / re-exports and docstrings or
docs naming symbols that no longer exist. `pysidtracker.lint` checks both with the
standard library only. Each format repo's `tests/test_lint.py` is:

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

Pass `whitelist=[...]` (`--whitelist ...`) for public API kept for downstream
users or external names named in prose.
