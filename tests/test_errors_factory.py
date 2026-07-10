"""Tests proving make_package_errors products are caught by base except."""

import pytest

from pysidtracker import (
    SidError,
    SidFormatError,
    SidParseError,
    make_package_errors,
)


def test_products_caught_by_base():
    errs = make_package_errors("Foo")
    assert errs.error.__name__ == "FooError"
    assert errs.parse_error.__name__ == "FooParseError"
    assert errs.format_error.__name__ == "FooFormatError"

    # The leaf parse error is caught by the base SidParseError.
    with pytest.raises(SidParseError):
        raise errs.parse_error("boom")
    # And by the package root and SidError.
    with pytest.raises(errs.error):
        raise errs.parse_error("boom")
    with pytest.raises(SidError):
        raise errs.parse_error("boom")
    # Format error is a base SidFormatError (and thus SidParseError).
    with pytest.raises(SidFormatError):
        raise errs.format_error("bad")
    with pytest.raises(SidParseError):
        raise errs.format_error("bad")


def test_unpackable():
    root, parse_error, format_error = make_package_errors("Bar")
    assert issubclass(parse_error, root)
    assert issubclass(format_error, root)
    assert issubclass(parse_error, SidParseError)
    assert issubclass(format_error, SidFormatError)
