"""The features a diverging register's value depends on, by data and control dependence.

One frame per side is sliced back from its last store to the register (``whydiff``),
each step keeping the branch it ran under, and the slice's branch outcomes are mapped
to :mod:`~pysidtracker.paths` ``edge:`` feature ids.
"""

from __future__ import annotations

import functools
from typing import Callable, Dict, Hashable, Iterable, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .paths import PathTrace, PySubject, source_of
from .whydiff import cpu_side, python_side

EXIT = -1
_BRANCHES = ("BPL", "BMI", "BVC", "BVS", "BCC", "BCS", "BNE", "BEQ")


def postdominators(edges: Iterable[Tuple[Hashable, Hashable]], exit_=EXIT):
    """``(ipdom, branches)`` of a control-flow graph: each node's immediate
    post-dominator (nodes that never reach ``exit_`` are absent) and the nodes with
    more than one successor."""
    g = nx.DiGraph()
    g.add_node(exit_)
    g.add_edges_from(edges)
    branches = {n for n, k in g.out_degree() if k > 1}
    return nx.immediate_dominators(g.reverse(copy=False), exit_), branches


def control_parents(
    nodes: Sequence[Optional[Hashable]],
    depth: Sequence[int],
    leave: Sequence[bool],
    ipdom: Dict,
    branches,
    exit_=EXIT,
) -> np.ndarray:
    """Each step's dynamic control parent: the innermost branch step whose region (up
    to its immediate post-dominator in the same call) it ran in, -1 none.

    ``nodes`` are the steps' CFG nodes (None: none), ``depth`` their call depth,
    ``leave`` the steps returning from it; a branch re-run in its own region replaces it.
    """
    out = np.full(len(nodes), -1, np.int64)
    stack: list = []
    for i, (node, d, back) in enumerate(zip(nodes, depth, leave)):
        while stack and (stack[-1][2] > d or stack[-1][1:] == (node, d)):
            stack.pop()
        if stack:
            out[i] = stack[-1][0]
        if node in branches:
            entry = (i, ipdom.get(node, exit_), d)
            if stack and stack[-1][1:] == entry[1:]:
                stack[-1] = entry
            else:
                stack.append(entry)
        if back:
            while stack and stack[-1][2] >= d:
                stack.pop()
    return out


@functools.lru_cache(maxsize=1)
def _mnemonics() -> np.ndarray:
    from jennings.opcodes import OPCODES

    return np.array([OPCODES[op][0] for op in range(256)])


def cpu_control(code=None) -> Callable:
    """``control(steps)``: the dynamic control parents of ``CpuStep`` s over ``code``'s
    (a ``StaticCode``) static edges, both arms of each branch and the executed edges;
    a call continues at its return site and a return leaves to the exit."""

    def control(steps) -> np.ndarray:
        if not steps:
            return np.zeros(0, np.int64)
        pcs = np.array([s.pc for s in steps], np.int64)
        mn = _mnemonics()[[s.raw[0] for s in steps]]
        call = mn == "JSR"
        leave = np.isin(mn, ("RTS", "RTI"))
        depth = np.maximum(np.cumsum(call.astype(np.int64) - leave) - call + leave, 0)
        dst = np.where(call, pcs + 3, np.append(pcs[1:], EXIT))
        dst[leave] = EXIT
        edges = list(zip(pcs.tolist(), dst.tolist()))
        rel = np.flatnonzero(np.isin(mn, _BRANCHES))
        off = np.array([steps[i].raw[1] for i in rel], np.int64)
        arms = pcs[rel] + 2 + np.where(off & 0x80, off - 0x100, off)
        edges += list(zip(pcs[rel].tolist(), (arms & 0xFFFF).tolist()))
        edges += list(zip(pcs[rel].tolist(), (pcs[rel] + 2).tolist()))
        if code is not None:
            edges += code.cfg(EXIT)
        ipdom, branches = postdominators(edges)
        return control_parents(pcs.tolist(), depth.tolist(), leave, ipdom, branches)

    return control


def next_lines(events) -> np.ndarray:
    """Per Python ``line`` event, the next line event of the same call (-1: none)."""
    out = np.full(len(events), -1, np.int64)
    ahead: Dict[int, int] = {}
    for i in range(len(events) - 1, -1, -1):
        event = events[i]
        if event.kind == "line":
            out[i] = ahead.get(event.depth, -1)
            ahead[event.depth] = i
        elif event.kind in ("call", "return"):
            ahead[event.depth] = -1
    return out


def py_control(sources: Optional[Dict[str, str]] = None) -> Callable:
    """``control(events)``: the dynamic control parents of ``PyEvent`` s over
    coverage.py's static line arcs of each traced file (``sources`` for exec'd code)
    and the executed ones."""

    def control(events) -> np.ndarray:
        files = {e.path: source_of(e.path, sources) for e in events if e.kind == "line"}
        nodes = [
            (
                (e.path, files[e.path].first.get(e.line, e.line))
                if e.kind == "line" and files[e.path] is not None
                else None
            )
            for e in events
        ]
        edges = [
            (
                (path, src.first.get(a, a)),
                (path, src.first.get(b, b)) if b > 0 else EXIT,
            )
            for path, src in files.items()
            if src is not None
            for a, b in src.flow
            if a > 0
        ]
        nxt = next_lines(events)
        edges += [
            (n, nodes[j] if j >= 0 else EXIT)
            for n, j in zip(nodes, nxt)
            if n is not None and (j < 0 or nodes[j] is not None)
        ]
        ipdom, branches = postdominators(edges)
        depth = [e.depth for e in events]
        leave = [e.kind == "return" for e in events]
        return control_parents(nodes, depth, leave, ipdom, branches)

    return control


def cpu_relevant(
    trace: PathTrace, data, frame: int, register: int, subtune: int = 0
) -> np.ndarray:
    """``trace``'s (``cpu_paths``) edge feature ids the register's value at play call
    ``frame`` depends on: the edge each step of its slice took."""
    side = cpu_side(
        data,
        frame,
        register,
        subtune=subtune,
        grid=trace.grid,
        control=cpu_control(trace.code),
    )
    ids = {e: i for i, e in enumerate(trace.edges)}
    code = trace.code
    pcs = [s.pc for s in side.events] + [EXIT]
    pcs = [pc if code is None or code.inspan(pc) else EXIT for pc in pcs]
    got = {ids.get((pcs[i], pcs[i + 1])) for i in side.keep}
    return np.array(sorted(got - {None}), np.int64)


def py_relevant(
    trace: PathTrace, subject: PySubject, frame: int, register: int
) -> np.ndarray:
    """``trace``'s (``py_paths``) arc feature ids the register's value at ``frame``
    depends on: the arc each branch line of its slice took."""
    sources = dict(subject.sources or {})
    side = python_side(
        subject.make,
        frame,
        register,
        label=subject.label,
        cell=subject.cell,
        roots=subject.roots,
        sources=sources,
        grid=trace.grid,
        control=py_control(sources),
    )
    events = side.events
    nxt = next_lines(events)
    index = {n: i for i, n in enumerate(trace.names)}
    got = set()
    for i in side.keep:
        event = events[i]
        src = source_of(event.path, sources) if event.kind == "line" else None
        if src is not None and event.line in src.branch:
            to = events[nxt[i]].line if nxt[i] >= 0 else 0
            got.add(index.get(src.arc_key(event.line, to)))
    return np.array(sorted(got - {None}), np.int64)


__all__ = [
    "control_parents",
    "cpu_control",
    "cpu_relevant",
    "next_lines",
    "postdominators",
    "py_control",
    "py_relevant",
]
