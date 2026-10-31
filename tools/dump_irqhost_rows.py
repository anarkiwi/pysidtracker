"""Pickle InterruptHost.raw for every fixture tune, to compare two implementations.

PYTHONPATH=src python tools/dump_irqhost_rows.py OUT.pkl [--seconds 1] [--backend native|python]
"""

import argparse
import pickle
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# pylint: disable=wrong-import-position
from pysidtracker import irqhost
from tests import irqhost_tunes


def main() -> None:
    """Dump ``{tune: raw}`` for every tune in ``irqhost_tunes.TUNES``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()
    irqhost.poweron_writes = lambda: None
    out = {}
    for name in tqdm(irqhost_tunes.TUNES, unit="tune"):
        host = irqhost.InterruptHost(irqhost_tunes.build(name), backend=args.backend)
        host.run(int(args.seconds * host.clock_hz))
        out[name] = list(host.raw)
    Path(args.out).write_bytes(pickle.dumps(out))


if __name__ == "__main__":
    main()
