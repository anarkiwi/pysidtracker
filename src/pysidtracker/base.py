"""The base class every ``py*`` SID parser subclasses for a consistent API.

A concrete parser implements :meth:`BaseSidParser.parse` (bytes -> its own song
model) and, to gain packed/relocating detection for free, the cheap
:meth:`BaseSidParser.recognize` predicate. In return it inherits a uniform
``read``/``parse``/``detect`` surface identical across every format.
"""

from __future__ import annotations

import abc
from typing import Any

from .detect import Detection, detect_playroutine
from .errors import SidParseError
from .image import SidImage
from .source import Source, read_bytes


class BaseSidParser(abc.ABC):
    """Abstract base giving every SID parser the same entry points.

    Subclass responsibilities:
      * :meth:`parse` -- decode raw container bytes into the format's song model.
      * :meth:`recognize` (optional) -- return a truthy anchor when the format's
        signature is present in a loaded image, enabling :meth:`detect`.

    Inherited for free: :meth:`read` (path/bytes/file dispatch) and
    :meth:`detect` (untrustworthy-header handling via
    :func:`pysidtracker.detect.detect_playroutine`).
    """

    #: Exception type raised for parse failures; subclasses may narrow this to
    #: their own :class:`~pysidtracker.errors.SidParseError` subclass.
    error_class: type = SidParseError

    @abc.abstractmethod
    def parse(self, data: bytes, **kwargs: Any) -> Any:
        """Decode raw ``.sid``/``.prg`` ``data`` into the format's song model."""

    def read(self, src: Source, **kwargs: Any) -> Any:
        """Read ``src`` (path, ``bytes``, or file-like) and :meth:`parse` it."""
        return self.parse(read_bytes(src), **kwargs)

    def load_image(self, data: bytes) -> SidImage:
        """Load ``data`` into a :class:`~pysidtracker.image.SidImage`."""
        return SidImage.from_bytes(data)

    def recognize(self, image: SidImage) -> object:  # pylint: disable=unused-argument
        """Return a truthy anchor if this format is present in ``image``.

        The default returns ``None`` (no static recogniser). Override to enable
        :meth:`detect`.
        """
        return None

    def detect(
        self,
        src: Source,
        *,
        init: bool = True,
        subtune: int = 0,
    ) -> Detection:
        """Classify how ``src`` presents its playroutine data.

        See :func:`pysidtracker.detect.detect_playroutine`. Uses
        :meth:`recognize`; returns
        :attr:`~pysidtracker.detect.PlayroutineKind.UNKNOWN` for parsers that do
        not override it.
        """
        image = self.load_image(read_bytes(src))
        return detect_playroutine(image, self.recognize, init=init, subtune=subtune)
