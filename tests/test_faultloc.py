"""Spectrum building, suspiciousness, correspondence, candidates and the minimum cover."""

import itertools

import numpy as np
import pytest
from scipy import sparse

from pysidtracker import faultloc as fl


def _sample(tune, dense, names, onset, extra=()):
    return (tune, sparse.csr_matrix(np.asarray(dense, bool)), names, onset, extra)


def _corpus(seed=0, ntunes=200, nframes=40, nbg=60, nbugs=3, per_tune=20):
    """Tunes over shared background code plus one planted bug and one noise path each.

    Background column ``i`` is ``6502:e{i}`` / ``py:e{perm[i]}`` and both sides run
    it on the same frames; bug and noise columns co-run on a few passing frames
    and only the 6502 side runs them on the onset frame.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(nbg)
    bug_of, samples = {}, []
    for t in range(ntunes):
        cols = rng.choice(nbg, per_tune, replace=False)
        bg = rng.random((nframes, per_tune)) < 0.4
        bug = t % nbugs
        onset = int(rng.integers(10, nframes))
        tag = np.zeros((nframes, 2), bool)
        tag[rng.choice(onset, 3, replace=False)] = True
        names = [f"6502:e{i}" for i in cols] + [f"py:e{perm[i]}" for i in cols]
        names += [f"6502:bug{bug}", f"py:bug{bug}", f"6502:n{t}", f"py:n{t}"]
        py_tag = tag.copy()
        py_tag[onset], tag[onset] = False, True
        dense = np.hstack(
            [bg, bg, tag[:, :1], py_tag[:, :1], tag[:, 1:], py_tag[:, 1:]]
        )
        samples.append(_sample(f"t{t}", dense, names, onset, (f"info:opt{t % 2}",)))
        bug_of[f"t{t}"] = bug
    return samples, perm, bug_of


@pytest.fixture(scope="module")
def corpus():
    samples, perm, bug_of = _corpus()
    spec = fl.build_spectrum(samples)
    return spec, fl.correspond(spec), perm, bug_of


def test_onset_slicing_and_dedup_weights():
    dense = np.array([[1, 0], [1, 0], [0, 1], [1, 0], [1, 1], [0, 1]])
    spec = fl.build_spectrum(
        [
            _sample("a", dense, ["x", "y"], 3, ("info:k",)),
            _sample("b", dense, ["y", "z"], None),
            _sample("c", dense, ["x", "y"], 99),
            _sample("d", dense[:0], ["x"], 0),
        ]
    )
    assert spec.names == ["x", "y", "info:k", "z"]
    assert spec.tunes == ["a", "b", "c", "d"]
    assert spec.onset.tolist() == [3, -1, -1, -1]
    assert spec.extra.tolist() == [False, False, True, False]
    rows = {
        (spec.tunes[t], f, tuple(spec.x[i].indices.tolist())): w
        for i, (t, f, w) in enumerate(zip(spec.tune, spec.fail, spec.weight))
    }
    assert rows == {
        ("a", False, (0, 2)): 2,
        ("a", False, (1, 2)): 1,
        ("a", True, (0, 2)): 1,
        ("b", False, (1,)): 3,
        ("b", False, (3,)): 2,
        ("b", False, (1, 3)): 1,
        ("c", False, (0,)): 3,
        ("c", False, (1,)): 2,
        ("c", False, (0, 1)): 1,
    }
    assert spec.fail_rows.tolist() == [int(np.flatnonzero(spec.fail)[0])]


def test_corpus_dedup_weights_sum_to_kept_frames():
    samples, _, _ = _corpus(ntunes=20)
    spec = fl.build_spectrum(samples, progress=True)
    per_tune = np.bincount(spec.tune, weights=spec.weight)
    assert per_tune.tolist() == [s[3] + 1 for s in samples]
    assert spec.fail.sum() == 20
    keys = {
        (t, f, spec.x[i].indices.tobytes())
        for i, (t, f) in enumerate(zip(spec.tune, spec.fail))
    }
    assert len(keys) == spec.x.shape[0]


def test_ochiai_and_dstar_match_hand_counts():
    dense = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 1]])
    spec = fl.build_spectrum(
        [
            _sample("a", dense, ["f", "g", "h"], 3),
            _sample("b", dense, ["f", "g", "h"], 2),
        ]
    )
    ef = np.array([1, 1, 1])
    ep = np.array([4, 3, 0])
    assert np.allclose(fl.ochiai(spec), ef / np.sqrt(2 * (ef + ep)))
    assert np.allclose(fl.dstar(spec), ef**2 / (ep + 1))
    assert np.allclose(fl.dstar(spec, star=3), ef**3 / (ep + 1))
    only = fl.build_spectrum([_sample("a", [[0, 0], [1, 0]], ["f", "g"], 1)])
    assert fl.dstar(only).tolist() == [np.inf, 0.0]
    assert fl.ochiai(only).tolist() == [1.0, 0.0]


def test_correspond_recovers_the_permutation(corpus):
    spec, corr, perm, _ = corpus
    for i, j in enumerate(perm):
        col = spec.index[f"6502:e{i}"]
        assert [spec.names[p] for p in corr.partners(col)] == [f"py:e{j}"]
        assert [spec.names[p] for p in corr.partners(spec.index[f"py:e{j}"])] == [
            f"6502:e{i}"
        ]
    assert corr.partners(spec.index["info:opt0"]).size == 0
    assert all(u.size == 0 for u in corr.unmatched.values())


def test_correspond_jaccard_fallback_and_threshold():
    rng = np.random.default_rng(1)
    base = rng.random((60, 2)) < 0.5
    base[:, 0] = True
    near = base[:, 0].copy()
    near[0] = False
    far = ~base[:, 1]
    dense = np.column_stack([base, near, far, np.zeros(60, bool)])
    spec = fl.build_spectrum(
        [_sample("a", dense, ["6502:a", "6502:b", "py:a", "py:b", "py:never"], None)]
    )
    corr = fl.correspond(spec, min_jaccard=0.9)
    assert [spec.names[p] for p in corr.partners(spec.index["6502:a"])] == ["py:a"]
    assert [spec.names[p] for p in corr.partners(spec.index["py:a"])] == ["6502:a"]
    names = {s: [spec.names[c] for c in v] for s, v in corr.unmatched.items()}
    assert names == {"6502:": ["6502:b"], "py:": ["py:b", "py:never"]}
    loose = fl.correspond(spec, min_jaccard=0.0)
    assert loose.partners(spec.index["6502:b"]).size > 0
    empty = fl.correspond(fl.build_spectrum([_sample("a", [[1]], ["x"], None)]))
    assert empty.adj.nnz == 0


def test_ochiai_ranks_planted_bugs_top(corpus):
    spec, _, _, _ = corpus
    top = np.argsort(-fl.ochiai(spec), kind="stable")[:3]
    assert sorted(spec.names[i] for i in top) == ["6502:bug0", "6502:bug1", "6502:bug2"]


def test_min_cover_finds_exactly_the_planted_bugs(corpus):
    spec, corr, _, bug_of = corpus
    scores = fl.ochiai(spec)
    cands = fl.candidates(spec, corr, scores=scores)
    assert all(len(c) == 2 for c in cands)
    bugs = [("path", f"6502:bug{b}", (f"py:bug{b}",)) for b in range(3)]
    assert set(fl.min_cover(cands)) == set(bugs)
    fixes = fl.rank_fixes(spec, cands, scores)
    assert {f.key for f in fixes} == set(bugs)
    assert sum(f.n_tunes for f in fixes) == len(spec.tunes)
    for f in fixes:
        bug = int(f.key[1][-1])
        assert all(bug_of[t] == bug for t in f.tunes)
        onsets = {spec.tunes[t]: spec.onset[t] for t in range(len(spec.tunes))}
        assert onsets[f.representative] == min(onsets[t] for t in f.tunes)
        assert f.score == pytest.approx(scores[spec.index[f.key[1]]])
    table = fl.fixes_table(fixes)
    assert [r["rank"] for r in table] == [1, 2, 3]
    assert table[0]["kind"] == "path"
    assert table[0]["n_tunes"] == fixes[0].n_tunes


def test_greedy_matches_exact_on_planted_corpus(corpus):
    spec, corr, _, _ = corpus
    cands = fl.candidates(spec, corr)
    assert set(fl.min_cover(cands, exact_limit=0)) == set(fl.min_cover(cands))


def test_milp_beats_greedy_on_the_classic_counterexample():
    cands = [{"A", "C"}, {"A", "C"}, {"A"}, {"B", "C"}, {"B", "C"}, {"B"}, set()]
    assert fl.min_cover(cands) == ["A", "B"]
    assert set(fl.min_cover(cands, exact_limit=0)) == {"A", "B", "C"}
    pair = [{"X", "Y"}, {"X", "Z"}]
    assert fl.min_cover(pair) == ["X"]
    assert sorted(fl.min_cover(pair, weights={"X": 3})) == ["Y", "Z"]
    assert fl.min_cover([set(), set()]) == []


def test_exact_cover_is_optimal_on_random_instances():
    rng = np.random.default_rng(3)
    for _ in range(25):
        nkeys = int(rng.integers(3, 9))
        cands = [
            set(rng.choice(nkeys, int(rng.integers(1, 4)), replace=False).tolist())
            for _ in range(int(rng.integers(4, 12)))
        ]
        keys = sorted(set().union(*cands))
        best = next(
            k
            for k in range(1, len(keys) + 1)
            if any(
                all(c & set(s) for c in cands) for s in itertools.combinations(keys, k)
            )
        )
        exact, greedy = fl.min_cover(cands), fl.min_cover(cands, exact_limit=0)
        for cover in (exact, greedy):
            assert all(c & set(cover) for c in cands)
        assert len(exact) == best <= len(greedy)
        hits = [sum(k in c for c in cands) for k in exact]
        assert hits == sorted(hits, reverse=True)


def test_candidates_fall_back_to_value_unexplained_and_info():
    names = ["6502:a", "py:a", "6502:solo", "6502:seen"]
    ok = np.array([[1, 1, 0, 1], [1, 1, 0, 0]] * 2)
    novel = np.vstack([ok, [1, 1, 1, 1]])
    same = np.vstack([ok, [1, 1, 0, 1]])
    lost = np.vstack([ok, [1, 0, 0, 1]])
    spec = fl.build_spectrum(
        [
            _sample("novel", novel, names, 4, ("info:x",)),
            _sample("value", same, names, 4),
            _sample("none", same, names, 4),
            _sample("lost", lost, names, 4, ("info:y",)),
        ]
    )
    corr = fl.correspond(spec)
    scores = np.zeros(len(spec.names))
    scores[spec.index["info:y"]] = 1.0
    cands = fl.candidates(
        spec, corr, store_pairs=[None, ("6502:s", "py:s"), None, None], scores=scores
    )
    assert cands == [
        {("path", "6502:solo", ()), ("info", "info:x")},
        {("value", "6502:s", "py:s")},
        {("unexplained",)},
        {("path", "6502:a", ("py:a",)), ("info", "info:y")},
    ]
    assert fl.candidates(spec, corr)[3] == {("path", "6502:a", ("py:a",))}
    labels = {"lost": {"6502:a": "LDA $1000", "py:a": "voice.freq"}}
    fixes = fl.rank_fixes(spec, cands, scores, labels=labels)
    keys = [f.key for f in fixes]
    assert len(keys) == 4 and ("info", "info:y") in keys
    assert all(set(keys) & c for c in cands)
    path = ("path", "6502:a", ("py:a",))
    lost = fl.rank_fixes(spec, cands, scores, cover=[path], labels=labels)[0]
    assert lost.labels == ("LDA $1000", "voice.freq")
    assert lost.tunes == ["lost"]
    info = fl.rank_fixes(spec, cands, scores, cover=[("info", "info:y")])
    assert info[0].score == 1.0 and info[0].tunes == ["lost"]
    assert fl.key_names(("unexplained",)) == ()
    assert fl.key_score(spec, scores, ("value", "6502:s", "py:s")) == 0.0
    unhit = fl.rank_fixes(spec, cands, scores, cover=[("nothing",)])
    assert unhit[0].representative == "" and unhit[0].n_tunes == 0


def test_candidates_without_failures_are_empty():
    spec = fl.build_spectrum([_sample("a", [[1, 0], [0, 1]], ["6502:a", "py:a"], None)])
    assert fl.candidates(spec, fl.correspond(spec)) == []
    assert fl.rank_fixes(spec, [], fl.ochiai(spec)) == []


def test_correspond_pairs_per_build_code_over_shared_tunes():
    rng = np.random.default_rng(5)
    samples = []
    for t in range(6):
        run = rng.random((30, 1)) < 0.5
        noise = rng.random((30, 1)) < 0.5
        names = [f"6502:build{t % 2}", "6502:noise", "py:shared"]
        samples.append(_sample(f"t{t}", np.hstack([run, noise, run]), names, None))
    spec = fl.build_spectrum(samples)
    corr = fl.correspond(spec)
    partners = sorted(spec.names[p] for p in corr.partners(spec.index["py:shared"]))
    assert partners == ["6502:build0", "6502:build1"]
    assert corr.partners(spec.index["6502:noise"]).size == 0
    assert (
        fl._shared_mass(
            sparse.csr_matrix((0, 3)), np.zeros(0, int), np.zeros(0, int)
        ).size
        == 0
    )
