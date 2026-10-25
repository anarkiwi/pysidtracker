"""Pure-Python SIDId: playroutine identification by ``sidid.cfg`` byte signatures.

Faithful to cadaver's SIDId semantics (players in cfg order, first match wins;
``??`` any byte, ``AND`` skip any bytes). The bundled cfg is SIDId's (BSD-3).
"""

import argparse
import functools
import os
import re
import sys
from collections import Counter
from importlib import resources
from pathlib import Path

DEFAULT_CFG = resources.files(__package__) / "data" / "sidid.cfg"
UNIDENTIFIED = "*Unidentified*"
_HEX = re.compile(r"[0-9A-Fa-f]{2}")
_WILD = {"??": b".", "AND": b".*?"}


def sidid_signatures(cfg=None):
    """SIDId ``cfg`` players in file order, as ``[(name, [signature regex])]``."""
    text = Path(cfg).read_text("latin-1") if cfg else DEFAULT_CFG.read_text("latin-1")
    players, regs, sig = [], None, []
    for tok in text.split():
        up = tok.upper()
        if _HEX.fullmatch(tok):
            sig.append(re.escape(bytes([int(tok, 16)])))
        elif up in _WILD:
            sig.append(_WILD[up])
        elif up == "END":
            if sig and regs is not None:
                regs.append(re.compile(b"".join(sig), re.S))
            sig = []
        else:
            regs, sig = [], []
            players.append((tok, regs))
    return [(name, regs) for name, regs in players if regs]


@functools.lru_cache(maxsize=None)
def default_signatures():
    """The bundled cfg's signatures, parsed once per process."""
    return sidid_signatures()


def identify_all(data, sigs=None):
    """Every player (cfg order) with a signature in ``data``: SIDId ``-m``."""
    sigs = default_signatures() if sigs is None else sigs
    return [n for n, regs in sigs if any(r.search(data) for r in regs)]


def identify(data, sigs=None):
    """SIDId's report: the first player in cfg order with a signature in ``data``."""
    sigs = default_signatures() if sigs is None else sigs
    return next((n for n, regs in sigs if any(r.search(data) for r in regs)), None)


def sid_paths(paths):
    """``.sid`` files under ``paths`` (files kept as given, directories recursed)."""
    for p in map(Path, paths):
        yield from sorted(p.rglob("*.sid")) if p.is_dir() else [p]


_SIGS = {}


def _init(cfg):
    _SIGS["sigs"] = sidid_signatures(cfg)


def _scan(args):
    path, multi = args
    data = Path(path).read_bytes()
    if multi:
        return identify_all(data, _SIGS["sigs"])
    name = identify(data, _SIGS["sigs"])
    return [name] if name else []


def scan(paths, jobs, cfg=None, multi=False):
    """Identify ``paths`` in parallel: yield ``(path, names)`` in input order.

    ``names`` is every matching player when ``multi`` else at most the first;
    empty when unidentified.
    """
    from .survey import pool  # pylint: disable=import-outside-toplevel

    paths = list(paths)
    with pool(jobs, _init, (cfg,)) as ex:
        yield from zip(paths, ex.map(_scan, ((p, multi) for p in paths), chunksize=64))


def main(argv=None):
    """``python -m pysidtracker.sidid PATH...``: ``path<TAB>player`` per tune."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    ap.add_argument("paths", nargs="+", help=".sid files or directories")
    ap.add_argument("--cfg", help="sidid.cfg (default: the bundled one)")
    ap.add_argument("--player", help="only players matching this regex")
    ap.add_argument("-m", "--multi", action="store_true", help="every match")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args(argv)
    # pylint: disable-next=import-outside-toplevel
    from tqdm import tqdm

    paths = list(sid_paths(a.paths))
    want = re.compile(a.player or "", re.I)
    tally = Counter()
    hits = scan(paths, a.jobs, a.cfg, a.multi)
    for path, names in tqdm(hits, total=len(paths), unit="sid"):
        for name in [n for n in names if want.search(n)] or (
            [] if a.player else [UNIDENTIFIED]
        ):
            tally[name] += 1
            print(f"{path}\t{name}")
    for name, n in tally.most_common():
        print(f"{n:7d} {name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
