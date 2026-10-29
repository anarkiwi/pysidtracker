"""Typed models as text: one Lark constructor rule per dataclass.

A grammar names each constructor rule after its dataclass (snake_case) and takes
``%import text (INT, STRING, COMMENT, WS)`` from ``grammars/text.lark``;
:class:`ModelTransformer` builds the models and :func:`term` prints them back.
"""

from __future__ import annotations

import dataclasses
import json
import re
from functools import cached_property
from typing import Any, ClassVar, Dict, Iterable, Tuple

from lark import Lark, Token, Transformer, Tree
from lark.exceptions import LarkError, VisitError

from .bytegrammar import FRAGMENTS
from .errors import SidParseError

_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")


def rule_name(cls: type) -> str:
    """The constructor rule of ``cls``: its name in snake_case (``SetAD``: ``set_ad``)."""
    return _CAMEL.sub("_", cls.__name__).lower()


def classes(module, *extra: type) -> Dict[str, type]:
    """``{rule: dataclass}`` of ``module``'s own dataclasses and ``extra``."""
    own = (
        v
        for v in vars(module).values()
        if dataclasses.is_dataclass(v) and v.__module__ == module.__name__
    )
    return {rule_name(cls): cls for cls in (*own, *extra)}


def _is_flag(field: dataclasses.Field) -> bool:
    return field.type in (bool, "bool")


class ModelTransformer(Transformer):
    """Each rule in :attr:`classes` to its dataclass: non-token children fill the
    other fields in order, and a ``bool`` field is True when a token of its
    upper-cased name is a child; a failed build raises ``VisitError``. ``INT``,
    ``STRING`` and ``none`` are values."""

    # pylint: disable=invalid-name
    classes: ClassVar[Dict[str, type]] = {}

    def __default__(self, data, children, meta):
        cls = self.classes.get(data)
        if cls is None:
            return super().__default__(data, children, meta)
        fields = [f for f in dataclasses.fields(cls) if f.init]
        tokens = {c.type for c in children if isinstance(c, Token)}
        values = [c for c in children if not isinstance(c, Token)]
        names = [f.name for f in fields if not _is_flag(f)]
        flags = {f.name: f.name.upper() in tokens for f in fields if _is_flag(f)}
        try:
            if len(values) > len(names):
                raise TypeError(f"{cls.__name__} takes {len(names)} values")
            return cls(**dict(zip(names, values)), **flags)
        except Exception as exc:
            raise VisitError(data, Tree(data, children, meta), exc) from exc

    def INT(self, token: Token) -> int:
        """``$hex`` or decimal, optionally negative."""
        digits = token.lstrip("-")
        value = int(digits[1:], 16) if digits.startswith("$") else int(digits)
        return -value if token.startswith("-") else value

    def STRING(self, token: Token) -> str:
        """A JSON string."""
        return json.loads(token)

    def none(self, _children) -> None:
        """The absent value."""
        return None


def hexint(value: int) -> str:
    """``value`` as ``$XX`` or ``-$XX``."""
    return f"-${-value:02X}" if value < 0 else f"${value:02X}"


def term(value: Any) -> str:
    """The constructor term of ``value``: ``Name(arg, ...)`` of its init fields
    with true ``bool`` ones by name, ``Name`` without arguments, ints by
    :func:`hexint`, strings as JSON and None as ``none``."""
    if value is None:
        return "none"
    if isinstance(value, int):
        return hexint(value)
    if isinstance(value, str):
        return json.dumps(value)
    args = []
    for field in (f for f in dataclasses.fields(value) if f.init):
        arg = getattr(value, field.name)
        if isinstance(arg, bool):
            args += [field.name] if arg else []
        else:
            args.append(term(arg))
    name = type(value).__name__
    return f"{name}({', '.join(args)})" if args else name


def seq(items: Iterable[Tuple[int, Any]], base: int, what: str, error: type) -> tuple:
    """The values of ``(index, value)`` items, raising ``error`` unless the
    indices count up from ``base``."""
    items = list(items)
    for want, (got, _) in enumerate(items, base):
        if got != want:
            raise error(f"{what} {got} where {want} is due")
    return tuple(value for _, value in items)


class TextGrammar:
    """``package``'s ``resource`` grammar (LALR) and the ``transformer_cls``
    building its models; text that does not parse or build raises ``error``."""

    def __init__(
        self,
        package: str,
        resource: str,
        transformer_cls: type,
        error: type = SidParseError,
    ):
        self.package, self.resource = package, resource
        self.transformer_cls, self.error = transformer_cls, error

    @cached_property
    def parser(self) -> Lark:
        """The (cached) LALR parser."""
        return Lark.open_from_package(
            self.package, self.resource, parser="lalr", import_paths=[FRAGMENTS]
        )

    def loads(self, text: str) -> Any:
        """The model of ``text``."""
        try:
            return self.transformer_cls().transform(self.parser.parse(text))
        except VisitError as exc:
            raise self.error(str(exc.orig_exc)) from exc.orig_exc
        except LarkError as exc:
            raise self.error(str(exc)) from exc


__all__ = [
    "ModelTransformer",
    "TextGrammar",
    "classes",
    "hexint",
    "rule_name",
    "seq",
    "term",
]
