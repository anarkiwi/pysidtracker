"""Corpus-wide spectrum-based fault localisation over executed-feature traces.

Pre-onset frames pass, the onset frame fails; features are scored (Ochiai,
DStar), paired across sides, turned into per-failure candidate causes (the branch
where the two executions first part, when the onset frame's execution order is
known) and reduced to a minimum hitting set of fixes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Dict,
    Hashable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.optimize import Bounds, LinearConstraint, milp
from tqdm import tqdm

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


@dataclass
class Spectrum:
    """Deduplicated pass/fail rows over a global feature vocabulary.

    ``x`` is (unique rows x features), ``weight`` each row's multiplicity,
    ``tune`` its index into ``tunes``; ``onset`` is per tune (-1: none) and
    ``extra`` flags the per-tune constant columns. ``order`` is (failing rows x
    features), aligned to :attr:`fail_rows`: 1 + each feature's first-occurrence rank
    on its side in the onset frame, an empty row where the order is not known.
    """

    names: List[str]
    x: sparse.csr_matrix
    weight: np.ndarray
    fail: np.ndarray
    tune: np.ndarray
    tunes: List[str]
    onset: np.ndarray
    extra: np.ndarray
    order: Optional[sparse.csr_matrix] = None
    index: Dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        self.index = {n: i for i, n in enumerate(self.names)}

    @property
    def fail_rows(self) -> np.ndarray:
        """Row indices of the failing rows, in tune order."""
        return np.flatnonzero(self.fail)

    def counts(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Per-feature weighted ``(ef, ep)`` and the total failing weight."""
        w = self.weight.astype(np.float64)
        wf = np.where(self.fail, w, 0.0)
        xt = self.x.T.astype(np.float64)
        return xt @ wf, xt @ (w - wf), float(wf.sum())


def _ids(vocab, names):
    return np.fromiter(
        (vocab.setdefault(n, len(vocab)) for n in names), np.int64, len(names)
    )


def _cat(parts, dtype):
    return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype)


def build_spectrum(samples, progress: bool = False) -> Spectrum:
    """A :class:`Spectrum` from ``(tune, matrix, names, onset[, extra[, rank]])`` samples.

    Frames before ``onset`` pass, frame ``onset`` fails, later frames drop (None:
    all pass); ``extra`` names tune-constant features; ``rank`` (aligned with
    ``names``, -1: not run) is each feature's first-occurrence rank on its side in
    the onset frame. Identical ``(features, fail, tune)`` rows collapse into one
    weighted row.
    """
    vocab: Dict[str, int] = {}
    tunes, onsets, extra_cols = [], [], []
    rows, cols, weights, fails, tix = [], [], [], [], []
    ocols, oranks, olens = [], [], []
    nrow = 0
    for t, (tune, mat, names, onset, *rest) in enumerate(
        tqdm(samples, desc="spectrum", unit="tune", disable=not progress)
    ):
        lut = _ids(vocab, names)
        ext = np.unique(_ids(vocab, rest[0] if rest else ()))
        rank = rest[1] if len(rest) > 1 and rest[1] is not None else None
        extra_cols.append(ext)
        tunes.append(tune)
        mat = sparse.csr_matrix(mat, dtype=bool)
        nframes = mat.shape[0]
        hit = onset is not None and 0 <= onset < nframes
        onsets.append(onset if hit else -1)
        mat = mat[: onset + 1 if hit else nframes]
        if not mat.shape[0]:
            continue
        if hit:
            ran = np.flatnonzero(np.asarray(rank) >= 0) if rank is not None else []
            ocols.append(lut[ran])
            oranks.append(np.asarray(rank)[ran] + 1 if len(ran) else np.zeros(0))
            olens.append(len(ran))
        fail = np.zeros(mat.shape[0], np.uint8)
        fail[-1] = hit
        key = np.ascontiguousarray(
            np.column_stack([fail, np.packbits(mat.toarray(), axis=1)])
        )
        _, first, count = np.unique(
            key.view(np.dtype((np.void, key.shape[1]))).ravel(),
            return_index=True,
            return_counts=True,
        )
        uniq = mat[first]
        r, c = uniq.nonzero()
        nu = len(first)
        rows += [r + nrow, np.repeat(np.arange(nu) + nrow, len(ext))]
        cols += [lut[c], np.tile(ext, nu)]
        weights.append(count)
        fails.append(fail[first].astype(bool))
        tix.append(np.full(nu, t))
        nrow += nu
    r, c = _cat(rows, np.int64), _cat(cols, np.int64)
    x = sparse.csr_matrix(
        (np.ones(len(r), np.int32), (r, c)), shape=(nrow, len(vocab))
    ).astype(bool)
    extra = np.zeros(len(vocab), bool)
    extra[_cat(extra_cols, np.int64)] = True
    indptr = np.zeros(len(olens) + 1, np.int64)
    np.cumsum(olens, out=indptr[1:])
    order = sparse.csr_matrix(
        (_cat(oranks, np.int64), _cat(ocols, np.int64), indptr),
        shape=(len(olens), len(vocab)),
    )
    return Spectrum(
        names=list(vocab),
        x=x,
        weight=_cat(weights, np.int64),
        fail=_cat(fails, bool),
        tune=_cat(tix, np.int64),
        tunes=tunes,
        onset=np.array(onsets, np.int64),
        extra=extra,
        order=order,
    )


def ochiai(spec: Spectrum) -> np.ndarray:
    """Per-feature Ochiai suspiciousness ``ef / sqrt(F * (ef + ep))``."""
    ef, ep, nfail = spec.counts()
    den = np.sqrt(nfail * (ef + ep))
    return np.divide(ef, den, out=np.zeros_like(ef), where=den > 0)


def dstar(spec: Spectrum, star: float = 2) -> np.ndarray:
    """Per-feature DStar ``ef^star / (ep + nf)``, inf for failing-only features."""
    ef, ep, nfail = spec.counts()
    num, den = ef**star, ep + (nfail - ef)
    out = np.divide(num, den, out=np.zeros_like(ef), where=den > 0)
    out[(den <= 0) & (num > 0)] = np.inf
    return out


def _splitmix(v, seed):
    z = (v.astype(np.uint64) + np.uint64(seed)) * _GOLDEN
    z = (z ^ (z >> np.uint64(30))) * _MIX1
    z = (z ^ (z >> np.uint64(27))) * _MIX2
    return z ^ (z >> np.uint64(31))


def _column_keys(m: sparse.csc_matrix) -> np.ndarray:
    """``(nnz, h1, h2)`` per column: a 128-bit set hash of its row indices."""
    out = [np.diff(m.indptr).astype(np.uint64)]
    for seed in (1, 2):
        cs = np.zeros(m.nnz + 1, np.uint64)
        np.cumsum(_splitmix(m.indices, seed), dtype=np.uint64, out=cs[1:])
        out.append(cs[m.indptr[1:]] - cs[m.indptr[:-1]])
    return np.column_stack(out)


@dataclass
class Correspondence:
    """Partner columns across the two sides, as a directed adjacency.

    ``classes`` (default: the connected components of ``adj``) numbers each partnered
    column's class of mutually corresponding columns, -1 for an unpartnered one.
    """

    names: List[str]
    side: np.ndarray
    adj: sparse.csr_matrix
    prefixes: Tuple[str, str]
    classes: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.classes is None:
            _, comp = connected_components(self.adj, directed=False)
            self.classes = np.where(self.matched, comp, -1)

    def partners(self, col: int) -> np.ndarray:
        """The other side's columns paired with ``col``."""
        return self.adj.indices[self.adj.indptr[col] : self.adj.indptr[col + 1]]

    @property
    def matched(self) -> np.ndarray:
        """Per column whether it has any partner."""
        return np.diff(self.adj.indptr) > 0

    @property
    def unmatched(self) -> Dict[str, np.ndarray]:
        """Per side prefix, its columns with no partner."""
        m = self.matched
        return {
            p: np.flatnonzero((self.side == s) & ~m)
            for s, p in enumerate(self.prefixes, 1)
        }


def _shared_mass(per_tune: sparse.csr_matrix, a: np.ndarray, b: np.ndarray):
    """Per pair ``(a, b)``: both rows' summed ``per_tune`` mass over the tunes both have."""
    small = np.where(np.diff(per_tune.indptr)[a] <= np.diff(per_tune.indptr)[b], a, b)
    big = a + b - small
    lens = np.diff(per_tune.indptr)[small]
    pair = np.repeat(np.arange(len(a)), lens)
    at = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    at += per_tune.indptr[small][pair]
    ntunes = per_tune.shape[1]
    keys = np.repeat(np.arange(per_tune.shape[0]), np.diff(per_tune.indptr))
    keys = keys * ntunes + per_tune.indices
    want = big[pair] * ntunes + per_tune.indices[at]
    pos = np.minimum(np.searchsorted(keys, want), max(len(keys) - 1, 0))
    hit = keys[pos] == want
    return np.bincount(
        pair[hit],
        weights=per_tune.data[at[hit]] + per_tune.data[pos[hit]],
        minlength=len(a),
    )


def _best_jaccard(qw, q, per_tune, src, dst, min_jaccard):
    """``(src, dst)`` class pairs: each ``src`` class's best ``dst`` by weighted Jaccard
    over the tunes both classes run in."""
    m = (qw[:, src].T @ q[:, dst]).tocoo()
    union = _shared_mass(per_tune, src[m.row], dst[m.col]) - m.data
    jac = m.data / union
    best = np.full(len(src), -np.inf)
    np.maximum.at(best, m.row, jac)
    keep = (jac >= min_jaccard) & (jac == best[m.row])
    return src[m.row[keep]], dst[m.col[keep]]


def correspond(
    spec: Spectrum, left: str = "6502:", right: str = "py:", min_jaccard: float = 0.9
) -> Correspondence:
    """Pair ``left`` and ``right`` columns by their passing-row indicators.

    Identical indicators are exact partners; a column with none takes the best
    weighted-Jaccard partners when ``>= min_jaccard``, over the passing rows of the
    tunes both run in (one side's code may differ by tune, e.g. a player's builds).
    Columns never active in a passing row have no partner.
    """
    n = len(spec.names)
    side = np.zeros(n, np.int8)
    side[[i for i, s in enumerate(spec.names) if s.startswith(left)]] = 1
    side[[i for i, s in enumerate(spec.names) if s.startswith(right)]] = 2
    passing = ~spec.fail
    p = spec.x[passing].tocsc()
    p.sort_indices()
    w = spec.weight[passing].astype(np.float64)
    live = np.flatnonzero((side > 0) & (np.diff(p.indptr) > 0))
    adj = sparse.csr_matrix((n, n), dtype=bool)
    if len(live):
        sub = p[:, live].tocsc()
        _, cls = np.unique(_column_keys(sub), axis=0, return_inverse=True)
        cls = cls.ravel()
        ncls = int(cls.max()) + 1
        has = np.zeros((ncls, 3), bool)
        has[cls, side[live]] = True
        rep = np.zeros(ncls, np.int64)
        rep[cls] = np.arange(len(live))
        q = sub[:, rep].astype(np.float64).tocsc()
        qw = (sparse.diags(w) @ q).tocsc()
        tunes = spec.tune[passing]
        per_tune = (
            qw.T
            @ sparse.csr_matrix(
                (np.ones(len(tunes)), (np.arange(len(tunes)), tunes)),
                shape=(len(tunes), len(spec.tunes)),
            )
        ).tocsr()
        per_tune.sort_indices()
        exact = np.flatnonzero(has[:, 1] & has[:, 2])
        pairs = [(exact, exact)] + [
            _best_jaccard(
                qw,
                q,
                per_tune,
                np.flatnonzero(has[:, a] & ~has[:, b]),
                np.flatnonzero(has[:, b]),
                min_jaccard,
            )
            for a, b in ((1, 2), (2, 1))
        ]
        src, dst = (np.concatenate(v) for v in zip(*pairs))
        e = sparse.csr_matrix((np.ones(len(src)), (src, dst)), shape=(ncls, ncls))
        member = sparse.csr_matrix((np.ones(len(live)), (live, cls)), shape=(n, ncls))
        a = (member @ e @ member.T).tocoo()
        keep = side[a.row] != side[a.col]
        adj = sparse.csr_matrix(
            (np.ones(keep.sum(), bool), (a.row[keep], a.col[keep])), shape=(n, n)
        )
    return Correspondence(spec.names, side, adj, (left, right))


class Parting(NamedTuple):
    """Where the two sides' onset-frame executions first part (feature names, "" none).

    ``left``/``right`` run at the parting, ``after`` is the last agreed pair before it.
    """

    left: str
    right: str
    after: Tuple[str, str]


def _part(a: np.ndarray, b: np.ndarray, ina: np.ndarray, inb: np.ndarray):
    """Positions ``(i, j, ci, cj)`` where class sequences ``a`` and ``b`` first part.

    Heads that are equal, or both run somewhere by the other side (an interleaving),
    are consumed together and later repeats of consumed classes skipped; the first
    head the other side never runs parts them (``len`` when a side has ended).
    ``(ci, cj)`` are the last equal heads (-1: none); None when nothing parts them.
    """
    done = np.zeros(len(ina), np.bool_)
    i = j = 0
    ci = cj = -1
    while True:
        while i < len(a) and done[a[i]]:
            i += 1
        while j < len(b) and done[b[j]]:
            j += 1
        if i == len(a) and j == len(b):
            return None
        if i == len(a) or j == len(b) or not (inb[a[i]] and ina[b[j]]):
            return i, j, ci, cj
        if a[i] == b[j]:
            ci, cj = i, j
        done[a[i]] = done[b[j]] = True
        i += 1
        j += 1


def partings(spec: Spectrum, corr: Correspondence) -> List[Optional[Parting]]:
    """Per failing row (aligned to :attr:`Spectrum.fail_rows`) its :class:`Parting`.

    Each side's partnered non-``store:`` features of the onset frame, in their
    first-occurrence order, are mapped to their correspondence classes and walked
    together (:func:`_part`): an unpartnered feature is helper code the other side
    need not mirror, and an interleaving (voices taken in another order) is not a
    parting. None where the order is unknown or nothing parts the sides.
    """
    nf = len(spec.fail_rows)
    order = spec.order
    if order is None or not order.nnz:
        return [None] * nf
    plen = np.array([len(p) for p in corr.prefixes])
    side = corr.side.astype(np.int64)
    store = np.array(
        [
            s > 0 and n.startswith("store:", plen[s - 1])
            for n, s in zip(spec.names, side)
        ]
    )
    use = (corr.classes >= 0) & ~store
    out: List[Optional[Parting]] = []
    for r in range(nf):
        lo, hi = order.indptr[r], order.indptr[r + 1]
        col, rank = order.indices[lo:hi], order.data[lo:hi]
        keep = use[col]
        col, rank = col[keep], rank[keep]
        seqs = [col[side[col] == s] for s in (1, 2)]
        seqs = [
            c[np.argsort(rank[side[col] == s], kind="stable")]
            for c, s in zip(seqs, (1, 2))
        ]
        cls, local = np.unique(corr.classes[np.concatenate(seqs)], return_inverse=True)
        a, b = local[: len(seqs[0])], local[len(seqs[0]) :]
        ina, inb = (np.isin(np.arange(len(cls)), x) for x in (a, b))
        got = _part(a, b, ina, inb) if len(col) else None
        if got is None:
            out.append(None)
            continue
        i, j, ci, cj = got

        def name(seq, k):
            return spec.names[seq[k]] if 0 <= k < len(seq) else ""

        out.append(
            Parting(
                name(seqs[0], i),
                name(seqs[1], j),
                (name(seqs[0], ci), name(seqs[1], cj)),
            )
        )
    return out


def key_names(key: Hashable) -> Tuple[str, ...]:
    """The feature names a candidate key mentions."""
    kind = key[0]
    if kind == "path":
        return (key[1],) + tuple(key[2])
    if kind in ("value", "info", "first"):
        return tuple(n for n in key[1:] if n)
    return ()


def key_score(spec: Spectrum, scores: np.ndarray, key: Hashable) -> float:
    """The highest score among the features a key mentions (0 when none are known)."""
    ids = [spec.index[n] for n in key_names(key) if n in spec.index]
    return float(np.max(scores[ids])) if ids else 0.0


def candidates(
    spec: Spectrum,
    corr: Correspondence,
    store_pairs: Optional[Sequence[Optional[Tuple[str, str]]]] = None,
    scores: Optional[np.ndarray] = None,
    parts: Optional[Sequence[Optional[Parting]]] = None,
) -> List[Set[Hashable]]:
    """Candidate cause keys per failing row (aligned to :attr:`Spectrum.fail_rows`).

    ``("first", left, right)`` where the sides part (``parts``, default
    :func:`partings`); else ``("path", name, partners)`` per side column active with
    all partners off (unpartnered: unseen in its tune's passing rows), else
    ``("value", ..)``, else ``("unexplained",)``; plus ``("info", name)`` extras
    scoring >= them.
    """
    scores = ochiai(spec) if scores is None else scores
    fr = spec.fail_rows
    if not len(fr):
        return []
    parts = partings(spec, corr) if parts is None else parts
    f = spec.x[fr].astype(np.float64).tocsr()
    pt = np.flatnonzero(~spec.fail)
    seen = (
        sparse.csr_matrix(
            (np.ones(len(pt)), (spec.tune[pt], np.arange(len(pt)))),
            shape=(len(spec.tunes), len(pt)),
        )
        @ spec.x[pt].astype(np.float64)
    ).tocsr()
    hit = (f @ corr.adj.T.astype(np.float64)).tocsr()
    r, c = f.nonzero()
    onside = corr.side[c] > 0
    parted = np.array([p is not None for p in parts], bool)
    keep = (
        onside
        & ~parted[r]
        & (np.asarray(hit[r, c]).ravel() == 0)
        & (corr.matched[c] | (np.asarray(seen[spec.tune[fr][r], c]).ravel() == 0))
    )
    nf = len(fr)
    best = np.full(nf, -np.inf)
    np.maximum.at(best, r[keep], scores[c[keep]])
    out: List[Set[Hashable]] = [set() for _ in range(nf)]
    for i in np.flatnonzero(parted):
        key = ("first", parts[i].left, parts[i].right)
        out[i].add(key)
        best[i] = key_score(spec, scores, key)
    fallback = np.full(nf, -np.inf)
    np.maximum.at(fallback, r[onside], scores[c[onside]])
    thr = np.where(np.isfinite(best), best, fallback)
    info = spec.extra[c] & (scores[c] >= thr[r])
    keys: Dict[int, Tuple] = {}
    for i, col in zip(r[keep], c[keep]):
        if col not in keys:
            keys[col] = (
                "path",
                spec.names[col],
                tuple(sorted(spec.names[p] for p in corr.partners(col))),
            )
        out[i].add(keys[col])
    for i in range(nf):
        if not out[i]:
            pair = store_pairs[i] if store_pairs is not None else None
            out[i].add(("value",) + tuple(pair) if pair else ("unexplained",))
    for i, col in zip(r[info], c[info]):
        out[i].add(("info", spec.names[col]))
    return out


def _incidence(cands):
    keys = sorted({k for s in cands for k in s}, key=repr)
    col = {k: i for i, k in enumerate(keys)}
    rows = [i for i, s in enumerate(cands) for _ in s]
    cols = [col[k] for s in cands for k in s]
    a = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(len(cands), len(keys))
    )
    return keys, a[np.diff(a.indptr) > 0].tocsc()


def _milp_cover(a, cost, score):
    """The optimal 0/1 cover, of maximum total score among the minimum-cost ones."""
    nk = a.shape[1]
    kw = {"integrality": np.ones(nk), "bounds": Bounds(0, 1)}
    cover = LinearConstraint(a, lb=1, ub=np.inf)
    x = milp(cost, constraints=cover, **kw).x
    if np.ptp(score) > 0:
        best = float(cost @ np.round(x))
        budget = LinearConstraint(cost[None], ub=best)
        x = milp(-score, constraints=[cover, budget], **kw).x
    return np.flatnonzero(x > 0.5)


def min_cover(
    cands: Sequence[Set[Hashable]],
    weights: Optional[Mapping[Hashable, float]] = None,
    exact_limit: int = 2000,
    scores: Optional[Mapping[Hashable, float]] = None,
) -> List[Hashable]:
    """A minimum-weight set of keys hitting every non-empty candidate set.

    Exact (``milp``, highest total ``scores`` among optima) up to ``exact_limit``
    keys, else greedy coverage per weight; ordered by sets hit, then score.
    """
    keys, a = _incidence(cands)
    if not keys:
        return []
    nk = len(keys)
    cost = np.array([(weights or {}).get(k, 1.0) for k in keys], np.float64)
    score = np.array([(scores or {}).get(k, 0.0) for k in keys], np.float64)
    if nk <= exact_limit:
        pick = _milp_cover(a, cost, score)
    else:
        unc = np.ones(a.shape[0], np.float64)
        chosen = []
        while unc.any():
            k = int(np.lexsort((-score, -(a.T @ unc) / cost))[0])
            chosen.append(k)
            unc[a.indices[a.indptr[k] : a.indptr[k + 1]]] = 0
        pick = np.array(chosen)
    cov = np.diff(a.indptr)[pick]
    order = pick[np.lexsort((pick, -score[pick], -cov))]
    return [keys[i] for i in order]


@dataclass
class Fix:
    """One cover key with the tunes it is credited with explaining."""

    key: Hashable
    tunes: List[str]
    n_tunes: int
    score: float
    representative: str
    labels: Tuple[str, ...]


def rank_fixes(
    spec: Spectrum,
    cands: Sequence[Set[Hashable]],
    scores: np.ndarray,
    cover: Optional[Sequence[Hashable]] = None,
    labels: Optional[Mapping[str, Mapping[str, str]]] = None,
    weights: Optional[Mapping[Hashable, float]] = None,
    exact_limit: int = 2000,
) -> List[Fix]:
    """The minimum cover of ``cands`` as :class:`Fix` records, in cover order.

    A failing tune is credited to the first cover key hitting it; the
    representative is the earliest-onset tune, whose ``labels`` name the key.
    """
    kscore = {k: key_score(spec, scores, k) for s in cands for k in s}
    if cover is None:
        cover = min_cover(cands, weights, exact_limit, kscore)
    rank = {k: i for i, k in enumerate(cover)}
    owned: List[List[int]] = [[] for _ in cover]
    tune_of = spec.tune[spec.fail_rows]
    for i, s in enumerate(cands):
        hits = [rank[k] for k in s if k in rank]
        if hits:
            owned[min(hits)].append(int(tune_of[i]))
    fixes = []
    for k, ts in zip(cover, owned):
        rep = spec.tunes[min(ts, key=lambda t: (spec.onset[t], t))] if ts else ""
        lab = (labels or {}).get(rep, {})
        fixes.append(
            Fix(
                key=k,
                tunes=[spec.tunes[t] for t in ts],
                n_tunes=len(ts),
                score=kscore.get(k, key_score(spec, scores, k)),
                representative=rep,
                labels=tuple(lab.get(n, n) for n in key_names(k)),
            )
        )
    return fixes


def fixes_table(fixes: Sequence[Fix]) -> List[dict]:
    """CSV-ready rows of :class:`Fix` records."""
    return [
        {
            "rank": i,
            "kind": f.key[0],
            "key": " | ".join(key_names(f.key)),
            "n_tunes": f.n_tunes,
            "score": f.score,
            "representative": f.representative,
            "labels": " | ".join(f.labels),
            "tunes": " ".join(f.tunes),
        }
        for i, f in enumerate(fixes, 1)
    ]
