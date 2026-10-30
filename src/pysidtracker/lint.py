"""Shared lint for dead symbols and unresolved doc references.

``black`` and ``pylint`` miss two kinds of rot that every format repo
accumulates: module-level constants / functions / re-exports nothing uses any
more, and docstrings or ``docs/*.md`` naming symbols that no longer exist.
:func:`check_project` finds both with the standard library only (no vulture), so
each dependent repo's ``test_lint.py`` is one call::

    from pysidtracker.lint import check_project

    def test_lint():
        assert check_project("src/mypkg", tests=["tests"], docs=["docs"]) == []

Also available as ``pysidtracker lint`` and the ``lint_project`` pytest fixture.

* **Unused symbols** -- a module-level definition (function, class, constant, or
  an ``__init__`` re-export) whose name is never read anywhere in the package or
  the ``tests`` directories. Being listed in ``__all__`` does not count as a use.
* **Unresolved references** -- Sphinx roles (``:class:`` / ``:func:`` /
  ``:meth:`` / ``:attr:`` / ``:mod:`` / ``:data:`` / ``:const:`` / ``:exc:``) and
  double-backtick dotted names in docstrings and Markdown that match no module,
  class, function, method or attribute of the package. A reference resolves when
  it equals, or is a dotted suffix of, a qualified name the package defines.

* **Truncated comparisons** -- a render or compare call in the ``tests``
  directories given a hard-coded frame or second count
  (:func:`truncated_comparisons`): comparisons cover whole songs.

The first two checks take a ``whitelist`` of names to ignore (public API kept for
downstream users, external symbols named in prose).
"""

from __future__ import annotations

import argparse
import ast
import builtins
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .errors import SidError

_ROLE = re.compile(r":(?:py:)?(class|func|meth|attr|mod|data|const|exc):`([^`]+)`")
_LITERAL = re.compile(r"``([A-Za-z_][\w]*(?:\.[A-Za-z_]\w*)+)(?:\(\))?``")
_BUILTINS = frozenset(dir(builtins))
_FILE_SUFFIXES = frozenset({"py", "md", "csv", "json", "cfg", "txt", "toml", "yml"})
# Names the interpreter / tooling reads without a visible reference.
_IMPLICIT = frozenset({"main", "__version__", "__all__"})


def _py_files(roots: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        root = Path(root)
        files.extend([root] if root.is_file() else sorted(root.rglob("*.py")))
    return files


def _parse(path: Path) -> Optional[ast.Module]:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), str(path))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None


def _targets(node: ast.AST) -> List[str]:
    """Names bound by a module-level assignment statement."""
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [node.target]
    else:
        return []
    return [t.id for t in targets if isinstance(t, ast.Name)]


def _definitions(tree: ast.Module, is_init: bool) -> List[Tuple[str, int]]:
    """Module-level ``(name, line)`` definitions (plus ``__init__`` re-exports)."""
    found: List[Tuple[str, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.append((node.name, node.lineno))
        elif is_init and isinstance(node, ast.ImportFrom):
            found.extend(
                (a.asname or a.name, node.lineno) for a in node.names if a.name != "*"
            )
        else:
            found.extend((name, node.lineno) for name in _targets(node))
    return [
        (n, line) for n, line in found if not (n.startswith("__") and n[-2:] == "__")
    ]


def _read_names(tree: ast.Module) -> Set[str]:
    """Every identifier a module reads, calls, accesses or imports by name."""
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            names.update(a.name.split(".")[-1] for a in node.names)
    return names


def unused_symbols(
    package: Path,
    tests: Sequence[Path] = (),
    whitelist: Iterable[str] = (),
) -> List[str]:
    """Problems for module-level symbols in ``package`` that nothing reads."""
    package = Path(package)
    skip = set(whitelist) | _IMPLICIT
    trees = {p: _parse(p) for p in _py_files([package, *tests])}
    reads: Dict[Path, Set[str]] = {
        p: _read_names(t) for p, t in trees.items() if t is not None
    }
    problems = []
    for path in _py_files([package]):
        tree = trees.get(path)
        if tree is None:
            continue
        for name, line in _definitions(tree, path.name == "__init__.py"):
            if name in skip or name.startswith("test_"):
                continue
            used = any(name in names for p, names in reads.items() if p != path)
            # A read in its own module counts, except for re-exports (which are
            # defined by an import, not used by one).
            if not used and path.name != "__init__.py":
                used = name in reads.get(path, set())
            if not used:
                problems.append(f"{path}:{line}: unused symbol {name!r}")
    return problems


def _qualified_names(package: Path) -> Set[str]:
    """Dotted names the package defines: modules, symbols, methods, attributes."""
    root = package.name
    names: Set[str] = {root}

    def add_class(prefix: str, cls: ast.ClassDef) -> None:
        for node in cls.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(f"{prefix}.{node.name}")
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Attribute) and (
                        isinstance(sub.value, ast.Name) and sub.value.id == "self"
                    ):
                        names.add(f"{prefix}.{sub.attr}")
            elif isinstance(node, ast.ClassDef):
                names.add(f"{prefix}.{node.name}")
                add_class(f"{prefix}.{node.name}", node)
            else:
                names.update(f"{prefix}.{n}" for n in _targets(node))

    for path in _py_files([package]):
        tree = _parse(path)
        if tree is None:
            continue
        rel = path.relative_to(package).with_suffix("").parts
        parts = rel[:-1] if rel[-1] == "__init__" else rel
        module = ".".join((root, *parts))
        names.add(module)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(f"{module}.{node.name}")
            elif isinstance(node, ast.ClassDef):
                names.add(f"{module}.{node.name}")
                add_class(f"{module}.{node.name}", node)
            elif isinstance(node, ast.ImportFrom):
                names.update(f"{module}.{a.asname or a.name}" for a in node.names)
            else:
                names.update(f"{module}.{n}" for n in _targets(node))
    return names


def _resolves(ref: str, qualified: Set[str], suffixes: Set[str]) -> bool:
    ref = ref.lstrip("~.").removesuffix("()")
    return ref in qualified or ref in suffixes


def _docstrings(tree: ast.Module) -> List[Tuple[int, str]]:
    found = []
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                body = node.body[0]
                found.append((getattr(body, "lineno", 1), doc))
    return found


def _references(text: str) -> List[Tuple[int, str, bool]]:
    """``(line_offset, ref, from_role)`` for every checkable name in ``text``."""
    refs = []
    for pattern, from_role in ((_ROLE, True), (_LITERAL, False)):
        for match in pattern.finditer(text):
            ref = match.group(match.lastindex).split("<")[-1].rstrip(">").strip()
            if match.re is _ROLE and match.group(1) == "mod" and "." not in ref:
                continue  # a sibling distribution, not a module of this package
            refs.append((text.count("\n", 0, match.start()), ref, from_role))
    return refs


def unresolved_references(
    package: Path,
    docs: Sequence[Path] = (),
    whitelist: Iterable[str] = (),
) -> List[str]:
    """Problems for doc references (docstrings in ``package``, ``docs`` Markdown)
    that name nothing the package defines."""
    package = Path(package)
    skip = set(whitelist)
    qualified = _qualified_names(package)
    suffixes = {
        ".".join(parts[i:])
        for name in qualified
        for parts in [name.split(".")]
        for i in range(len(parts))
    }
    heads = {name.split(".")[0] for name in qualified} | {
        name.split(".")[1] for name in qualified if "." in name
    }
    heads |= {s.split(".")[0] for s in suffixes}

    sources: List[Tuple[str, int, str, Set[str]]] = []
    for path in _py_files([package]):
        tree = _parse(path)
        if tree is not None:
            imported = {
                a.asname or a.name.split(".")[0]
                for n in ast.walk(tree)
                if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in n.names
            }
            for line, doc in _docstrings(tree):
                sources.append((str(path), line, doc, imported))
    for root in docs:
        root = Path(root)
        for path in [root] if root.is_file() else sorted(root.rglob("*.md")):
            sources.append((str(path), 1, path.read_text(encoding="utf-8"), set()))

    problems = []
    for where, first, text, imported in sources:
        for offset, ref, from_role in _references(text):
            bare = ref.lstrip("~.").removesuffix("()")
            head = bare.split(".")[0]
            if bare in skip or head in skip or head in imported:
                continue
            if bare.rsplit(".", 1)[-1] in _FILE_SUFFIXES:
                continue  # a file name, not a symbol
            if not from_role:
                # Prose literals are only trusted when they name the package.
                if head != package.name:
                    continue
            elif "." in bare:
                if head not in heads:
                    continue  # external symbol (numpy.ndarray, ...)
            elif bare in _BUILTINS:
                continue
            if not _resolves(ref, qualified, suffixes):
                problems.append(
                    f"{where}:{first + offset}: unresolved reference {ref!r}"
                )
    return problems


# Render/compare calls: positions and keywords of their frame or second counts.
LENGTH_ARGS: Dict[str, Tuple[Tuple[int, ...], Tuple[str, ...]]] = {
    "render_grid": ((0,), ("nframes",)),
    "iter_frames": ((0,), ("max_frames",)),
    "register_grid": ((1,), ("nframes",)),
    "oracle_grid": ((), ("frames", "seconds")),
    "sidtrace_oracle": ((), ("frames", "seconds")),
    "oracle_rows": ((2, 4), ("seconds", "calls")),
    **{
        name: ((), ("frames",))
        for name in (
            "decode_fidelity",
            "assert_decode_fidelity",
            "assert_matches",
            "assert_oracle_match",
            "make_oracle_fixtures",
            "explain_divergence",
            "fidelity_row",
            "compare_file",
            "file_cli",
        )
    },
}


def _is_number(node: ast.AST, constants: Set[str]) -> bool:
    """Whether ``node`` is a number literal, a name bound only to one, or arithmetic
    of them."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.Name):
        return node.id in constants
    if isinstance(node, ast.UnaryOp):
        return _is_number(node.operand, constants)
    if isinstance(node, ast.BinOp):
        return _is_number(node.left, constants) and _is_number(node.right, constants)
    return False


def _number_names(trees: Iterable[ast.Module]) -> Set[str]:
    """Names every assignment in ``trees`` binds to a number expression."""
    bound: Dict[str, List[ast.AST]] = {}
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value:
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name):
                        bound.setdefault(target.id, []).append(node.value)
            elif isinstance(node, ast.arg):
                bound.setdefault(node.arg, []).append(node)
            elif isinstance(node, (ast.AugAssign, ast.For, ast.comprehension)):
                for target in ast.walk(node.target):
                    if isinstance(target, ast.Name):
                        bound.setdefault(target.id, []).append(node)
    numbers: Set[str] = set()
    while True:
        more = {
            name
            for name, values in bound.items()
            if name not in numbers and all(_is_number(v, numbers) for v in values)
        }
        if not more:
            return numbers
        numbers |= more


def truncated_comparisons(tests: Sequence[Path]) -> List[str]:
    """Problems for render/compare calls in ``tests`` given a hard-coded length.

    A call named in :data:`LENGTH_ARGS` whose frame or second count is a number
    literal, a name every assignment binds to one (across ``tests``), or arithmetic
    of those cuts the comparison to a window of the song. Compare whole songs
    (:func:`pysidtracker.testing.song_frames`); a synthetic song whose whole length
    is the number says so with :func:`pysidtracker.testing.whole_song`.
    """
    trees = {p: t for p in _py_files(tests) if (t := _parse(p)) is not None}
    numbers = _number_names(trees.values())
    problems = []
    for path, tree in trees.items():
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        for node in sorted(calls, key=lambda n: (n.lineno, n.col_offset)):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else func
            name = name.id if isinstance(name, ast.Name) else name
            if not isinstance(name, str) or name not in LENGTH_ARGS:
                continue
            spots, keywords = LENGTH_ARGS[name]
            args = [node.args[i] for i in spots if i < len(node.args)]
            args += [k.value for k in node.keywords if k.arg in keywords]
            if any(_is_number(arg, numbers) for arg in args):
                problems.append(
                    f"{path}:{node.lineno}: {name}() compares a hard-coded length, "
                    "not the whole song"
                )
    return problems


def check_project(
    package: str,
    tests: Sequence[str] = ("tests",),
    docs: Sequence[str] = ("docs",),
    whitelist: Iterable[str] = (),
) -> List[str]:
    """All lint problems for the package directory ``package`` (empty when clean).

    Missing ``tests`` / ``docs`` directories are ignored.
    """
    whitelist = list(whitelist)
    test_dirs = [Path(t) for t in tests if Path(t).exists()]
    doc_dirs = [Path(d) for d in docs if Path(d).exists()]
    return (
        unused_symbols(Path(package), test_dirs, whitelist)
        + unresolved_references(Path(package), doc_dirs, whitelist)
        + truncated_comparisons(test_dirs)
    )


def add_lint_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments of ``pysidtracker lint`` (shared with the main CLI)."""
    parser.add_argument("package", help="package directory, e.g. src/mypkg")
    parser.add_argument("--tests", nargs="*", default=["tests"])
    parser.add_argument("--docs", nargs="*", default=["docs"])
    parser.add_argument("--whitelist", nargs="*", default=[], help="names to ignore")


def run_lint(args: argparse.Namespace) -> None:
    """Print the problems for parsed ``args``; raise :class:`SidError` when any."""
    problems = check_project(args.package, args.tests, args.docs, args.whitelist)
    for problem in problems:
        print(problem)
    if problems:
        raise SidError(f"{len(problems)} lint problem(s)")
