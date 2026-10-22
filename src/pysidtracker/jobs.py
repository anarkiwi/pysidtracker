"""Process pools whose parent enforces each job's deadline from outside.

A job's :func:`deadline` arms its supervising parent, which kills the worker's whole process
group (the worker and every process it started, such as a nested pool) once the budget runs
out, wherever the job is: in Python, in a C extension or waiting on a child.
"""

import multiprocessing
import os
import signal
import time
from collections import deque
from contextlib import contextmanager
from multiprocessing.connection import wait

_PARENT = None


@contextmanager
def deadline(seconds, fallback=None):
    """End the enclosed work after ``seconds`` (0: never).

    In an :func:`imap` worker the parent kills the worker at the deadline and takes
    ``fallback`` as the job's result (a :class:`TimeoutError` without one). Elsewhere
    ``SIGALRM`` raises :class:`TimeoutError`, which Python delivers only between bytecodes.
    """
    if not seconds:
        yield
        return
    if _PARENT is not None:
        _PARENT.send(("deadline", float(seconds), fallback))
        try:
            yield
        finally:
            _PARENT.send(("deadline", 0.0, None))
        return

    def _expire(*_):
        raise TimeoutError(f"exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _serve(conn, own_group, initializer, initargs):
    """A worker: run each ``(fn, args)`` received and send back ``("done", value, error)``."""
    global _PARENT  # pylint: disable=global-statement
    if own_group:
        os.setsid()
    _PARENT = conn
    if initializer is not None:
        initializer(*initargs)
    while (task := conn.recv()) is not None:
        fn, args = task
        try:
            out = ("done", fn(*args), None)
        except Exception as exc:  # pylint: disable=broad-except
            out = ("done", None, exc)
        try:
            conn.send(out)
        except Exception as exc:  # pylint: disable=broad-except
            conn.send(("done", None, RuntimeError(f"unpicklable job result: {exc!r}")))


class _Worker:
    """One worker process and the job it runs, with that job's deadline and fallback."""

    def __init__(self, ctx, initializer, initargs):
        self.conn, child = ctx.Pipe()
        self.group = _PARENT is None
        self.process = ctx.Process(
            target=_serve, args=(child, self.group, initializer, initargs)
        )
        self.process.start()
        child.close()
        self.job = self.until = self.fallback = None

    def start(self, job, fn, args):
        """Send job ``job``: ``fn(*args)``."""
        self.job, self.until, self.fallback = job, None, None
        self.conn.send((fn, args))

    def kill(self):
        """Kill the worker and, when it leads one, its process group."""
        if self.group:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self.process.kill()
        self.process.join()
        self.conn.close()

    def stop(self):
        """Ask an idle worker to exit, then reap it."""
        try:
            self.conn.send(None)
        except OSError:
            pass
        self.process.join()
        self.conn.close()


def _ended(worker):
    """``(value, error)`` of the busy ``worker``'s job once it ends, None while it runs; a
    ``deadline`` message arms or disarms the worker's kill."""
    while worker.conn.poll():
        try:
            kind, value, error = worker.conn.recv()
        except (EOFError, OSError):
            break
        if kind == "done":
            return value, error
        worker.until = time.monotonic() + value if value else None
        worker.fallback = error
    if worker.process.is_alive():
        if worker.until is None or time.monotonic() < worker.until:
            return None
        worker.kill()
        if worker.fallback is not None:
            return worker.fallback, None
        return None, TimeoutError("the job ran past its deadline")
    worker.kill()
    return None, ChildProcessError(f"worker exited with {worker.process.exitcode}")


def imap(fn, tasks, jobs=None, initializer=None, initargs=()):
    """Yield ``(index, value, error)`` of ``fn(*task)`` per task as it ends, over ``jobs``
    spawned workers: ``error`` what ``fn`` raised, a :class:`TimeoutError` where its
    :func:`deadline` passed (``value`` its fallback, if any) or a :class:`ChildProcessError`
    where its worker died. A killed or dead worker is replaced."""
    todo = deque(enumerate(tasks))
    if not todo:
        return
    ctx = multiprocessing.get_context("spawn")
    count = max(1, min(jobs or len(os.sched_getaffinity(0)), len(todo)))
    idle, busy = [_Worker(ctx, initializer, initargs) for _ in range(count)], []
    try:
        while todo or busy:
            while idle and todo:
                index, args = todo.popleft()
                worker = idle.pop()
                worker.start(index, fn, args)
                busy.append(worker)
            armed = [w.until for w in busy if w.until is not None]
            wait(
                [w.conn for w in busy] + [w.process.sentinel for w in busy],
                max(0.0, min(armed) - time.monotonic()) if armed else None,
            )
            for worker in list(busy):
                ended = _ended(worker)
                if ended is None:
                    continue
                busy.remove(worker)
                yield (worker.job, *ended)
                idle.append(
                    worker
                    if worker.process.is_alive()
                    else _Worker(ctx, initializer, initargs)
                )
    finally:
        for worker in busy:
            worker.kill()
        for worker in idle:
            worker.stop()


def results(fn, items, jobs=None, failed=None):
    """``fn(item)`` of each item as it ends (:func:`imap`); ``failed(item, error)`` stands
    for one that raised, ran past its deadline without a fallback or lost its worker."""
    items = list(items)
    for index, value, error in imap(fn, [(item,) for item in items], jobs):
        if error is None:
            yield value
        elif failed is None:
            raise error
        else:
            yield failed(items[index], error)
