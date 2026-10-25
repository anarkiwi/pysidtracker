"""Facet registry: per-value provenance for a player build's variant axes.

A format package declares its build/release/layout/init axes as :class:`Facet`
objects in a :class:`FacetRegistry`. Detecting an image yields a :class:`Variant`:
for every facet a value and its :class:`Source` -- ``detected@$addr``, ``default``
or ``inferred(evidence)``. ``Variant.resolve()`` materialises a plain frozen
dataclass so players keep plain attribute access, and ``Variant.as_info()``
flattens values plus a separate ``<facet>.src`` key for the survey ``info`` dict.

(Not to be confused with :mod:`pysidtracker.variants`, which clusters unknown
builds by execution fingerprint.)
"""

from __future__ import annotations

from dataclasses import dataclass, field, make_dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Tuple

from .codescan import CodePattern, Match, find_code_first

KINDS = ("build", "release", "layout", "init")

DETECTED = "detected"
DEFAULT = "default"
INFERRED = "inferred"


@dataclass(frozen=True)
class Source:
    """Where a facet value came from.

    ``kind`` is ``detected`` (read at ``addr``), ``default`` or ``inferred``
    (``evidence`` says from what).
    """

    kind: str
    addr: Optional[int] = None
    evidence: str = ""

    def __str__(self) -> str:
        if self.kind == DETECTED:
            return f"detected@${self.addr:04x}" if self.addr is not None else DETECTED
        if self.kind == INFERRED:
            return f"inferred({self.evidence})"
        return self.kind


def detected(addr: Optional[int] = None) -> Source:
    """A :class:`Source` for a value read from code or data at ``addr``."""
    return Source(DETECTED, addr)


def inferred(evidence: str) -> Source:
    """A :class:`Source` for a value deduced from ``evidence``."""
    return Source(INFERRED, evidence=evidence)


DEFAULT_SOURCE = Source(DEFAULT)


@dataclass(frozen=True)
class Finding:
    """What a detector reports: a facet ``value`` and its ``source``."""

    value: Any
    source: Source


Detector = Callable[[Any], Optional[Finding]]


@dataclass(frozen=True)
class Facet:
    """One variant axis.

    ``domain`` (optional) is the set of legal values. ``default`` is either a plain
    value or, with ``per_family=True``, a mapping of family -> value (a driver
    byte, a release) used when no detector fires. ``detector`` maps an image to a
    :class:`Finding` or ``None``.
    """

    name: str
    kind: str
    domain: Optional[Tuple[Any, ...]] = None
    default: Any = None
    detector: Optional[Detector] = field(default=None, compare=False)
    per_family: bool = False

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"facet name is not an identifier: {self.name!r}")
        if self.kind not in KINDS:
            raise ValueError(f"facet {self.name}: kind {self.kind!r} not in {KINDS}")
        if self.domain is not None:
            object.__setattr__(self, "domain", tuple(self.domain))

    def default_for(self, family: Optional[str] = None) -> Any:
        """The default value, for ``family`` when the default is per-family."""
        if self.per_family:
            return self.default.get(family)
        return self.default

    def check(self, value: Any) -> Any:
        """``value`` if it is in the domain (or the domain is open), else ``ValueError``."""
        if self.domain is not None and value not in self.domain:
            raise ValueError(
                f"facet {self.name}: {value!r} not in domain {self.domain!r}"
            )
        return value

    def detect(self, image: Any, family: Optional[str] = None) -> "Entry":
        """Run the detector on ``image``; fall back to the default."""
        found = self.detector(image) if self.detector is not None else None
        if found is None:
            return Entry(self.default_for(family), DEFAULT_SOURCE)
        return Entry(self.check(found.value), found.source)


@dataclass(frozen=True)
class Entry:
    """A facet's value with its provenance."""

    value: Any
    source: Source


@dataclass(frozen=True)
class Variant:
    """The detected facet values of one image, each with its source."""

    entries: Mapping[str, Entry]
    family: Optional[str] = None

    def __getitem__(self, name: str) -> Any:
        return self.entries[name].value

    def __contains__(self, name: object) -> bool:
        return name in self.entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.entries)

    def source(self, name: str) -> Source:
        """The :class:`Source` of facet ``name``."""
        return self.entries[name].source

    def replace(
        self, name: str, value: Any, source: Optional[Source] = None
    ) -> "Variant":
        """A copy with ``name`` overridden (an explicit override by default)."""
        if name not in self.entries:
            raise KeyError(name)
        entries = dict(self.entries)
        entries[name] = Entry(value, source or inferred("override"))
        return Variant(entries, self.family)

    def resolve(self, name: str = "Resolved") -> Any:
        """A plain frozen dataclass instance of the values (attribute access)."""
        cls = make_dataclass(
            name, [(n, Any) for n in self.entries], frozen=True, slots=True
        )
        return cls(**{n: e.value for n, e in self.entries.items()})

    def as_info(self) -> Dict[str, Any]:
        """Values plus a separate ``<facet>.src`` string key, for the survey info dict."""
        info: Dict[str, Any] = {}
        for name, entry in self.entries.items():
            info[name] = entry.value
            info[f"{name}.src"] = str(entry.source)
        return info


class FacetRegistry:
    """An ordered, name-unique collection of :class:`Facet`."""

    def __init__(self, facets: Iterable[Facet] = ()) -> None:
        self._facets: Dict[str, Facet] = {}
        for facet in facets:
            self.add(facet)

    def add(self, facet: Facet) -> Facet:
        """Register ``facet``; a duplicate name is an error."""
        if facet.name in self._facets:
            raise ValueError(f"duplicate facet: {facet.name}")
        self._facets[facet.name] = facet
        return facet

    def merge(self, other: "FacetRegistry") -> "FacetRegistry":
        """A new registry holding this one's facets then ``other``'s."""
        return FacetRegistry([*self, *other])

    def __iter__(self) -> Iterator[Facet]:
        return iter(self._facets.values())

    def __len__(self) -> int:
        return len(self._facets)

    def __contains__(self, name: object) -> bool:
        return name in self._facets

    def __getitem__(self, name: str) -> Facet:
        return self._facets[name]

    def of_kind(self, kind: str) -> Tuple[Facet, ...]:
        """The facets of one ``kind``, in registration order."""
        return tuple(f for f in self if f.kind == kind)

    def detect(self, image: Any, family: Optional[str] = None) -> Variant:
        """Detect every facet on ``image`` into a :class:`Variant`."""
        return Variant({f.name: f.detect(image, family) for f in self}, family)


def pattern_detector(
    pattern: CodePattern | str,
    value: Any = True,
    *,
    capture: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> Detector:
    """Detect by a code pattern: its first match gives ``value`` (or a capture).

    The source is ``detected@$addr`` of the match.
    """
    pat = pattern if isinstance(pattern, CodePattern) else CodePattern(pattern)

    def detect(image: Any) -> Optional[Finding]:
        match: Optional[Match] = find_code_first(image, pat, start=start, end=end)
        if match is None:
            return None
        got = match.captures[capture] if capture is not None else value
        return Finding(got, detected(match.addr))

    return detect


def on_off_detector(
    on: CodePattern | str,
    off: CodePattern | str,
    *,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> Detector:
    """Detect a feature switch by its enabled/disabled code site (True/False)."""
    on_det = pattern_detector(on, True, start=start, end=end)
    off_det = pattern_detector(off, False, start=start, end=end)

    def detect(image: Any) -> Optional[Finding]:
        return on_det(image) or off_det(image)

    return detect


def reference_detector(
    key: Callable[[Any], Any], table: Mapping[Any, Any], label: str = "reference"
) -> Detector:
    """Detect by looking ``key(image)`` up in a reference -> value table.

    The source is ``inferred(<label> <key>)``; an unknown key reports nothing.
    """

    def detect(image: Any) -> Optional[Finding]:
        ref = key(image)
        if ref is None or ref not in table:
            return None
        return Finding(table[ref], inferred(f"{label} {ref}"))

    return detect


def probe_detector(
    probe: Callable[[Any], Optional[Tuple[Any, str]]],
) -> Detector:
    """Detect by a format-local behavioural probe returning ``(value, evidence)``."""

    def detect(image: Any) -> Optional[Finding]:
        got = probe(image)
        if got is None:
            return None
        value, evidence = got
        return Finding(value, inferred(evidence))

    return detect
