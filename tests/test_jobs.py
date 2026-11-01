"""A job's deadline is enforced from outside: a job blocked in C or waiting on a child pool is
killed with every process it started, and the other jobs run on."""

import hashlib
import multiprocessing
import os
import time
from pathlib import Path

import pytest

from pysidtracker import jobs

FOREVER = (1 << 31) - 1


def spin(seconds, fallback=None):
    """Hash for ``FOREVER`` rounds in C under a deadline of ``seconds``."""
    with jobs.deadline(seconds, fallback):
        return hashlib.pbkdf2_hmac("sha256", b"x", b"y", FOREVER)


def nested(pids, seconds):
    """Wait under a deadline on a child pool whose job spins in C, its pid written to
    ``pids``."""
    with jobs.deadline(seconds):
        with multiprocessing.get_context("spawn").Pool(1) as pool:
            Path(pids).write_text(str(pool.apply(os.getpid)), encoding="utf-8")
            return pool.apply(hashlib.pbkdf2_hmac, ("sha256", b"x", b"y", FOREVER))


def square(x):
    """``x * x``, or a ValueError for a negative ``x``."""
    if x < 0:
        raise ValueError(x)
    with jobs.deadline(60):
        return x * x


def alive(pid: int) -> bool:
    """Whether ``pid`` runs (a zombie does not)."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[2]
    except FileNotFoundError:
        return False
    return state != "Z"


def test_a_job_spinning_in_c_ends_at_its_deadline_and_the_others_run_on():
    start = time.monotonic()
    got = sorted(jobs.imap(spin, [(1,), (1, "late")], jobs=2), key=lambda r: r[0])
    assert time.monotonic() - start < 60
    assert isinstance(got[0][2], TimeoutError) and got[0][1] is None
    assert got[1][1:] == ("late", None)
    got = sorted(jobs.imap(square, [(3,), (-1,), (4,)], jobs=2), key=lambda r: r[0])
    assert [r[1] for r in got] == [9, None, 16] and isinstance(got[1][2], ValueError)


def test_a_dead_worker_is_reported_and_replaced():
    got = sorted(jobs.imap(os._exit, [(3,), (4,)], jobs=1), key=lambda r: r[0])
    assert [str(r[2]) for r in got] == ["worker exited with 3", "worker exited with 4"]


def test_a_deadline_outside_a_pool_raises_between_bytecodes():
    with pytest.raises(TimeoutError):
        with jobs.deadline(0.2):
            while True:
                pass
    with jobs.deadline(0):
        pass


def test_results_stand_a_failed_item_in_or_raise():
    got = list(
        jobs.results(square, [2, -2], 1, lambda item, e: (item, type(e).__name__))
    )
    assert got == [4, (-2, "ValueError")]
    with pytest.raises(ValueError):
        list(jobs.results(square, [-3], 1))
    assert not list(jobs.imap(square, []))
