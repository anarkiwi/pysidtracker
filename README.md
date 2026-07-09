# pysidtracker

Shared base for the pure-Python C64 SID tracker parsers (pygoattracker,
pysidwizard, pydmcsid, pyfuturecomposer, pymusicassembler, pydefmon, pyjch).

Provides one implementation of the pieces every format parser duplicated:

- **`parse_sid_header` / `SidHeader`** — PSID/RSID container header parsing.
- **`SidImage`** — a loaded 64 KiB C64 memory image with absolute-addressed
  accessors, from a `.sid` container or a bare `.prg`.
- **`read_bytes`** — path / `bytes` / file-like source dispatch.
- **`SidError` hierarchy** — `SidParseError`, `SidFormatError`,
  `EmulatorUnavailable`.
- **`detect_playroutine` / `PlayroutineKind`** — the untrustworthy-header
  detector: static signature recognition first, then an emulated init run to
  classify `DIRECT` / `RELOCATED` / `PACKED` / `UNKNOWN` playroutines.
- **`BaseSidParser`** — the class each format subclasses for a consistent
  `read` / `parse` / `detect` API.

## Install

```
pip install pysidtracker          # core (stdlib only)
pip install pysidtracker[emu]     # + py65, to unpack packed/relocating tunes
pip install pysidtracker[fast]    # + numpy, to accelerate the image scan
```

## Usage

```python
from pysidtracker import BaseSidParser, PlayroutineKind

class MyParser(BaseSidParser):
    def recognize(self, image):
        return image.find(b"MYSIG")          # truthy anchor when found
    def parse(self, data, **kw):
        image = self.load_image(data)
        ...                                    # decode image.mem into a model

det = MyParser().detect("tune.sid")
if det.kind is PlayroutineKind.PACKED:
    ...                                        # header was not trustworthy
```

See [docs/design.md](docs/design.md) for the detection model and how the
format packages consume this base.

## Development

```
pip install -e ".[dev]"
pytest --cov=pysidtracker
```

Apache-2.0 licensed.
