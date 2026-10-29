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


@settings(max_examples=400, deadline=None)
@given(graphs())
def test_minimise_is_label_sequence_equivalence(graph):
    labels, succ = graph
    classes = minimise(labels, succ)
    steps = 2 * len(labels) + 1
    traces = {n: _trace(labels, succ, n, steps) for n in labels}
    for x in labels:
        for y in labels:
            assert (classes[x] == classes[y]) == (traces[x] == traces[y])
    assert list(dict.fromkeys(classes[n] for n in labels)) == list(
        range(len(set(classes.values())))
    )


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


def test_minimise_cycles_and_tails():
    labels = dict(enumerate("abab" "ab" "xab" "aab"))
    succ = {0: 1, 1: 2, 2: 3, 3: 0, 4: 5, 5: 4, 6: 7, 7: 8, 8: 7, 9: 10, 10: 11}
    classes = minimise(labels, succ)
    assert classes[0] == classes[2] == classes[4] == classes[7]
    assert classes[1] == classes[3] == classes[5] == classes[8]
    assert classes[6] not in {classes[0], classes[1]}
    assert len({classes[9], classes[10], classes[11], classes[0]}) == 4
    assert minimise({"p": 1, "q": 1}, {"p": "q"}) == {"p": 0, "q": 1}
    assert minimise({"p": 1, "q": 1}, {"p": "q", "q": "p"}) == {"p": 0, "q": 0}


@settings(max_examples=300, deadline=None)
@given(graphs(), st.lists(st.integers(-1, 7), max_size=6), st.integers(0, 3))
def test_number_chains(graph, picks, start):
    labels, succ = graph
    entries = [None if p < 0 or p >= len(labels) else p for p in picks]
    numbers = number_chains(entries, succ, start)
    assert sorted(numbers.values()) == list(range(start, start + len(numbers)))
    reach = set()
    for node in entries:
        while node is not None and node not in reach:
            reach.add(node)
            node = succ.get(node)
    assert set(numbers) == reach
    for node, num in numbers.items():
        nxt = succ.get(node)
        if nxt is not None and numbers[nxt] != num + 1:
            assert numbers[nxt] <= num


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


@settings(max_examples=400, deadline=None)
@given(graphs(), st.integers(0, 2))
def test_layout_preserves_every_successor(graph, first):
    _, succ = graph
    slots, jumps = layout_chains(succ, 64, first)
    _check_layout(succ, slots, jumps, 64, first)
    assert len(jumps) <= sum(t is not None for t in succ.values())


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
