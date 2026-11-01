"""Tests for program-table automata (pysidtracker.automaton)."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pysidtracker.automaton import layout_chains, minimise, number_chains
from pysidtracker.errors import SidFormatError


@st.composite
def graphs(draw, max_nodes=8, alphabet=3):
    """``(labels, succ)`` of a random functional graph over ``0..n-1``."""
    n = draw(st.integers(1, max_nodes))
    labels = {i: "abc"[draw(st.integers(0, alphabet - 1))] for i in range(n)}
    targets = draw(st.lists(st.integers(-1, n - 1), min_size=n, max_size=n))
    order = draw(st.permutations(range(n)))
    succ = {i: (None if targets[i] < 0 else targets[i]) for i in order}
    return labels, succ


def _trace(labels, succ, node, steps):
    out = []
    for _ in range(steps):
        if node is None:
            out.append(None)
            break
        out.append(labels[node])
        node = succ.get(node)
    return tuple(out)


@settings(max_examples=200, deadline=None)
@given(graphs())
def test_minimise_is_stable(graph):
    labels, succ = graph
    classes = minimise(labels, succ)
    for x in labels:
        for y in labels:
            if classes[x] == classes[y]:
                sx, sy = succ.get(x), succ.get(y)
                assert labels[x] == labels[y]
                assert (sx is None) == (sy is None)
                assert sx is None or classes[sx] == classes[sy]


def test_number_chains_example():
    succ = {"a": "b", "b": None, "c": "b", "d": "e", "e": "d"}
    assert number_chains(["c", None, "a", "d"], succ, 1) == {
        "c": 1,
        "b": 2,
        "a": 3,
        "d": 4,
        "e": 5,
    }


def _check_layout(succ, slots, jumps, size, base):
    assert set(slots) == set(succ)
    used = list(slots.values()) + list(jumps)
    assert len(set(used)) == len(used)
    assert all(base <= s < size for s in used)
    for node, target in succ.items():
        if target is not None:
            after = slots[node] + 1
            assert slots[target] == after or jumps.get(after) == target


def test_layout_concrete_and_fallback():
    succ = {0: 1, 1: 2, 2: 0, 3: 1, 4: None}
    slots, jumps = layout_chains(succ, 7, 1)
    assert slots == {3: 1, 1: 2, 2: 3, 0: 4, 4: 6}
    assert jumps == {5: 1}
    slots, jumps = layout_chains(succ, 6, 1)
    assert min(slots.values()) == 0
    _check_layout(succ, slots, jumps, 6, 0)
    with pytest.raises(SidFormatError, match="need 6 table slots"):
        layout_chains(succ, 5, 1)
    with pytest.raises(ValueError):
        layout_chains({"x": "x"}, 1, error=ValueError)
