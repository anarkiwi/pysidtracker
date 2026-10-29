"""Byte streams parsed as Lark grammars, one latin-1 character per byte.

A :class:`ByteGrammar` wraps a package's ``.lark`` resource (LALR, contextual
lexer, several start rules) and an inline transformer whose start rules return
:class:`Parsed`. Grammars ``%import bytes (BYTE, _TAIL)`` from
``pysidtracker/grammars/bytes.lark``.
"""

from __future__ import annotations

from typing import (
    Any,
    Collection,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

from lark import Lark, Token, Transformer, Tree
from lark.exceptions import LarkError, UnexpectedInput, UnexpectedToken
from lark.load_grammar import FromPackageLoader

from .errors import SidParseError
from .facets import Variant
from .image import SidImage

REACH = 0x100

FRAGMENTS = FromPackageLoader(__package__, ("grammars",))

Stream = Tuple[str, Any, str]


class Parsed(NamedTuple):
    """A parsed stream: its model and the bytes it spans, terminator included."""

    model: Any
    size: int


def _describe(exc: LarkError, data: bytes, rule: str) -> str:
    if isinstance(exc, UnexpectedToken) and exc.token.type == "$END":
        return f"{rule} has no terminator within {len(data)} bytes"
    if isinstance(exc, UnexpectedInput) and 0 <= exc.pos_in_stream < len(data):
        pos = exc.pos_in_stream
        return f"{rule}: unexpected byte ${data[pos]:02x} at offset {pos}"
    return f"{rule} does not parse: {exc}"


def _byte(token: Token) -> str:
    return f"{token.type}=${ord(token):02x}"


def render(node: Tree, depth: int = 0) -> Iterator[str]:
    """Indented lines of ``node`` in stream order; a rule holding only bytes
    is one line of ``TERMINAL=$xx`` items."""
    pad = "  " * depth
    if not any(isinstance(child, Tree) for child in node.children):
        yield " ".join([pad + node.data] + [_byte(tok) for tok in node.children])
        return
    yield pad + node.data
    for child in node.children:
        if isinstance(child, Tree):
            yield from render(child, depth + 1)
        else:
            yield pad + "  " + _byte(child)


class ByteGrammar:
    """The ``start`` rules of ``package``'s ``resource`` grammar over raw bytes.

    ``transformer`` builds each start rule's :class:`Parsed` while parsing;
    a stream that does not parse raises ``error`` (a :class:`SidParseError`).
    """

    def __init__(
        self,
        package: str,
        resource: str,
        start: Sequence[str],
        transformer: Optional[Transformer] = None,
        error: type = SidParseError,
    ):
        self.package = package
        self.resource = resource
        self.start = tuple(start)
        self.transformer = transformer
        self.error = error
        self._lark: Dict[bool, Lark] = {}

    def parser(self, trees: bool = False) -> Lark:
        """The (cached) LALR parser: transformer results, else parse trees."""
        if trees not in self._lark:
            self._lark[trees] = Lark.open_from_package(
                self.package,
                self.resource,
                parser="lalr",
                lexer="contextual",
                start=list(self.start),
                transformer=None if trees else self.transformer,
                import_paths=[FRAGMENTS],
            )
        return self._lark[trees]

    def _parse(self, data: bytes, rule: str, trees: bool = False):
        data = bytes(data)
        try:
            return self.parser(trees).parse(data.decode("latin-1"), start=rule)
        except LarkError as exc:
            raise self.error(_describe(exc, data, rule)) from exc

    def parse_prefix(self, data: bytes, rule: str) -> Parsed:
        """The ``rule`` stream at the start of ``data``; the rest is never read."""
        return self._parse(data, rule)

    def parse_exact(self, data: bytes, rule: str) -> Any:
        """The model of ``data``, which must be exactly one ``rule`` stream."""
        model, size = self._parse(data, rule)
        if size != len(data):
            raise self.error(f"{rule} ends after {size} of its {len(data)} bytes")
        return model

    def tree(self, data: bytes, rule: str) -> Tree:
        """The parse tree of the ``rule`` stream at the start of ``data``."""
        return self._parse(data, rule, trees=True)

    def read(
        self,
        img: SidImage,
        addr: int,
        rule: str,
        what: Optional[str] = None,
        size: int = REACH,
    ) -> Parsed:
        """The ``rule`` stream a ``size``-byte cursor reads from ``addr``."""
        try:
            return self.parse_prefix(img.window(addr, size), rule)
        except self.error as exc:
            raise self.error(f"{what or rule} at {addr:#06x}: {exc}") from exc

    def entry(
        self,
        img: SidImage,
        lo: int,
        hi: int,
        index: int,
        rule: str,
        what: Optional[str] = None,
        size: int = REACH,
    ) -> Parsed:
        """The ``rule`` stream entry ``index`` of the ``lo``/``hi`` table points at."""
        what = what or f"{rule} {index}"
        return self.read(img, img.ptr(lo, hi, index), rule, what, size)

    def stored(
        self,
        img: SidImage,
        lo: int,
        hi: int,
        index: int,
        rule: str,
        region: range,
        size: int = REACH,
    ) -> Any:
        """The model entry ``index`` points at in ``region``, else ``None``.

        The stream is read no further than ``region.stop``; a pointer outside
        ``region`` or to bytes that are not one ``rule`` stream gives ``None``.
        """
        addr = img.ptr(lo, hi, index)
        if addr not in region:
            return None
        try:
            return self.read(img, addr, rule, size=min(size, region.stop - addr)).model
        except self.error:
            return None

    def check(
        self,
        streams: Iterable[Stream],
        error: type,
        skip_equality: Collection[str] = (),
    ) -> None:
        """Raise ``error`` unless every ``(name, model, rule)``'s
        ``model.to_bytes()`` parses back as exactly one stream equal to ``model``
        (only one stream for a rule in ``skip_equality``)."""
        for name, model, rule in streams:
            try:
                back = self.parse_exact(model.to_bytes(), rule)
            except self.error as exc:
                raise error(f"{name} is not one {rule} stream: {exc}") from exc
            if rule not in skip_equality and back != model:
                raise error(f"{name} reads back as {back}, not {model}")

    def listing(self, streams: Iterable[Stream]) -> Iterator[str]:
        """``name:`` then the indented :func:`render` of each stream's tree."""
        for name, model, rule in streams:
            yield f"{name}:"
            for line in render(self.tree(model.to_bytes(), rule)):
                yield "  " + line


def _facet(variant: Any, name: str) -> Any:
    if isinstance(variant, (Variant, Mapping)):
        return variant[name] if name in variant else None
    return getattr(variant, name, None)


def select_rule(variant: Any, rules: Sequence[Tuple[str, str]], default: str) -> str:
    """The rule of the first ``(facet, rule)`` whose facet ``variant`` sets.

    ``variant`` is a :class:`~pysidtracker.facets.Variant` or mapping, else
    read by attribute (a resolved variant); ``default`` when none is set.
    """
    return next((rule for facet, rule in rules if _facet(variant, facet)), default)


__all__ = [
    "ByteGrammar",
    "FRAGMENTS",
    "Parsed",
    "REACH",
    "render",
    "select_rule",
]
