"""Structural comparison of two Lark grammars.

Each grammar is compiled as Lark compiles it and normalised to ``{rule: {alternative}}``
and ``{terminal: pattern}``; EBNF helper rules and anonymous terminals are inlined
by structure, so two spellings of the same language-level rule compare equal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from lark.exceptions import LarkError
from lark.lexer import PatternStr
from lark.load_grammar import load_grammar

from .bytegrammar import FRAGMENTS
from .errors import SidError

Rules = Dict[str, FrozenSet[str]]
Terminals = Dict[str, str]
_HELPER, _ANON, _SELF = "__", "__ANON", "@"


def _pattern(pattern) -> str:
    flags = "".join(sorted(pattern.flags))
    if isinstance(pattern, PatternStr):
        return repr(pattern.value) + flags
    return f"/{pattern.value}/{flags}"


def normalise(
    text: str, path: str = "<grammar>", start: Optional[Sequence[str]] = None
) -> Tuple[Rules, Terminals]:
    """``({rule: alternatives}, {terminal: pattern})`` of grammar ``text``, its
    rules those reachable from ``start`` (default: every rule it defines)."""
    imports = [FRAGMENTS, str(Path(path).parent)]
    grammar, _ = load_grammar(text, path, imports, False)
    if start is None:
        start = [name for name, params, _, _ in grammar.rule_defs if not params]
    terminals, rules, _ = grammar.compile(list(start), ())
    anon = {t.name: _pattern(t.pattern) for t in terminals if t.name[:6] == _ANON}
    alts: Dict[str, List[Tuple[str, ...]]] = {}
    for rule in rules:
        names = tuple(sym.name for sym in rule.expansion)
        alts.setdefault(rule.origin.name, []).append(
            names + ((f"-> {rule.alias}",) if rule.alias else ())
        )
    helpers: Dict[str, str] = {}

    def symbol(name: str) -> str:
        if name in anon:
            return anon[name]
        if not name.startswith(_HELPER) or name not in alts:
            return name
        if name not in helpers:
            helpers[name] = _SELF
            body = sorted(
                " ".join(_SELF if s == name else symbol(s) for s in alt)
                for alt in alts[name]
            )
            helpers[name] = "{" + " | ".join(body) + "}"
        return helpers[name]

    out = {
        name: frozenset(" ".join(map(symbol, alt)) for alt in rule_alts)
        for name, rule_alts in alts.items()
        if not name.startswith(_HELPER)
    }
    terms = {t.name: _pattern(t.pattern) for t in terminals if t.name not in anon}
    return out, terms


def compare(
    a: Tuple[Rules, Terminals], b: Tuple[Rules, Terminals], names=("A", "B")
) -> List[str]:
    """The differences of two :func:`normalise` results, one line each."""
    (rules_a, terms_a), (rules_b, terms_b) = a, b
    out = [f"rule {r} only in {names[0]}" for r in sorted(rules_a.keys() - rules_b)]
    out += [f"rule {r} only in {names[1]}" for r in sorted(rules_b.keys() - rules_a)]
    for rule in sorted(rules_a.keys() & rules_b):
        out += [
            f"rule {rule}: - {alt}" for alt in sorted(rules_a[rule] - rules_b[rule])
        ]
        out += [
            f"rule {rule}: + {alt}" for alt in sorted(rules_b[rule] - rules_a[rule])
        ]
    for term in sorted(terms_a.keys() | terms_b):
        pat_a, pat_b = terms_a.get(term), terms_b.get(term)
        if pat_a is None or pat_b is None:
            out.append(f"terminal {term} only in {names[pat_a is None]}")
        elif pat_a != pat_b:
            out.append(f"terminal {term}: {pat_a} != {pat_b}")
    return out


def grammar_diff(path_a, path_b, start: Optional[Sequence[str]] = None) -> List[str]:
    """The differences between the grammar files ``path_a`` and ``path_b``."""
    try:
        sides = [
            normalise(Path(p).read_text("utf-8"), str(p), start)
            for p in (path_a, path_b)
        ]
    except LarkError as exc:
        raise SidError(f"grammar does not load: {exc}") from exc
    return compare(*sides, names=(str(path_a), str(path_b)))


def run_grammardiff(args) -> None:
    """Print the differences; raise :class:`SidError` when there are any."""
    lines = grammar_diff(args.a, args.b, args.start)
    for line in lines:
        print(line)
    if lines:
        raise SidError(f"{len(lines)} grammar difference(s)")


def add_grammardiff_arguments(parser) -> None:
    """Arguments of ``pysidtracker grammardiff``."""
    parser.add_argument("a", help="first .lark grammar")
    parser.add_argument("b", help="second .lark grammar")
    parser.add_argument(
        "--start",
        action="append",
        help="start rule (repeatable; default: every rule each grammar defines)",
    )
    parser.set_defaults(func=run_grammardiff)


__all__ = [
    "add_grammardiff_arguments",
    "compare",
    "grammar_diff",
    "normalise",
    "run_grammardiff",
]
