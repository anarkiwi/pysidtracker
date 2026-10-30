"""Shared test configuration."""

import hashlib

import pytest

from pysidtracker.testing import SONGLENGTHS

pytest_plugins = ["pytester"]


class Songlengths:
    """A synthetic HVSC ``Songlengths.md5`` under ``root``, the ``$HVSC`` tree."""

    def __init__(self, root):
        self.path = root / SONGLENGTHS
        self.path.parent.mkdir(parents=True)
        self.path.write_text("[Database]\n", "latin-1")

    def add(self, data: bytes, *seconds: float) -> None:
        """List ``data``'s songs as ``seconds`` long."""
        lens = " ".join(f"{int(s // 60)}:{s % 60:06.3f}" for s in seconds)
        with self.path.open("a", encoding="latin-1") as out:
            out.write(f"{hashlib.md5(data).hexdigest()}={lens}\n")


@pytest.fixture(autouse=True)
def songlengths(request, tmp_path_factory, monkeypatch):
    """Song lengths from a synthetic ``$HVSC`` tree, so no test fetches HVSC's;
    ``oracle`` tests read the real one."""
    if request.node.get_closest_marker("oracle"):
        return None
    lengths = Songlengths(tmp_path_factory.mktemp("hvsc"))
    monkeypatch.setenv("HVSC", str(lengths.path.parent.parent))
    return lengths
