"""Checks for knn.py. Runs identically under CPython and Pyodide."""

import numpy as np

from learnml import Grid, Trace


def moons(n, seed, noise=0.3):
    """Two interleaving half-moons with labels −1 (outer) / +1 (inner)."""
    rng = np.random.default_rng(seed)
    t = np.pi * rng.uniform(size=n)
    outer = np.arange(n) < n // 2
    X = np.where(outer[:, None], np.column_stack([np.cos(t), np.sin(t)]),
                 np.column_stack([1 - np.cos(t), 0.5 - np.sin(t)]))
    return X + noise * rng.normal(size=(n, 2)), np.where(outer, -1, 1)


# The hand-worked example in the notes: eight past orders and a new address q = (4, 4).
EX_X = np.array([[6.2, 4], [5.5, 5.5], [4, 1.5], [2, 6], [1, 3], [7, 1], [8, 6], [2.5, 1]])
EX_Y = np.array([1, -1, 1, -1, -1, 1, 1, -1])
EX_Q = np.array([[4.0, 4.0]])


def check(s, t):
    X, y = moons(120, 0)
    Xt, yt = moons(400, 1)

    def t_dist():
        A = np.array([[0.0, 0.0], [1.0, 1.0]])
        B = np.array([[3.0, 4.0]])
        for p, want in [(1, [7, 5]), (2, [5, np.sqrt(13)]), (np.inf, [4, 3])]:
            got = s.pairwise_distances(A, B, p)
            assert got.shape == (2, 1), f"p = {p}: shape {got.shape}, expected (2, 1)"
            assert np.allclose(got[:, 0], want), f"p = {p}: got {np.round(got[:, 0], 3)}, expected {np.round(want, 3)}"
        D = s.pairwise_distances(X[:5], X[:7], 2)
        ref = np.array([[np.linalg.norm(a - b) for b in X[:7]] for a in X[:5]])
        assert np.allclose(D, ref), "Euclidean distances do not match a direct loop"

    def t_vote():
        cases = [(1, 1, 1), (1, 2, -1), (3, 1, 1), (3, 2, 1), (3, np.inf, -1)]
        for k, p, want in cases:
            got = int(s.knn_predict(EX_X, EX_Y, EX_Q, k, p)[0])
            assert got == want, f"worked example, k = {k}, p = {p}: predicted {got}, expected {want}"
        sc = float(s.knn_score(EX_X, EX_Y, EX_Q, 3, 2)[0])
        assert np.isclose(sc, 1 / 3), f"k = 3, L2: mean label {sc:.3f}, expected 1/3 (two late, one on time)"
        assert np.mean(s.knn_predict(X, y, X, 1) != y) == 0, "1-NN must have zero training error"

    def t_loo():
        for k in (1, 7):
            got = s.loo_error(X, y, k)
            pred = []
            for i in range(len(y)):
                d = np.linalg.norm(X - X[i], axis=1)
                d[i] = np.inf
                pred.append(1 if y[np.argsort(d, kind="stable")[:k]].sum() > 0 else -1)
            ref = float(np.mean(np.array(pred) != y))
            assert np.isclose(got, ref), f"k = {k}: LOO error {got:.4f}, expected {ref:.4f}"
        assert s.loo_error(X, y, 1) > 0, "LOO error of 1-NN should be positive (unlike its training error)"

    def t_choose():
        ks = list(range(1, 62, 2))
        k = s.choose_k(X, y, ks)
        errs = [s.loo_error(X, y, kk) for kk in ks]
        assert k == ks[int(np.argmin(errs))], f"chose k = {k}, but the smallest LOO error is at k = {ks[int(np.argmin(errs))]}"
        acc = float(np.mean(s.knn_predict(X, y, Xt, k) == yt))
        assert acc >= 0.85, f"test accuracy {acc:.3f} with k = {k} is below 0.85"

    t.test("三种距离", "Three distances", t_dist)
    t.test("多数表决（含讲义例题）", "Majority vote (incl. the worked example)", t_vote)
    t.test("留一法误差", "Leave-one-out error", t_loo)
    t.test("用留一法选 k", "Choosing k by leave-one-out", t_choose)

    def trace():
        grid = Grid(X, n=40, pad=0.4)
        tr = Trace("classifier2d", X, y, grid=grid, title={"zh": "k 从 1 增大到 61", "en": "k from 1 up to 61"})
        s.choose_k(X, y, [1, 3, 5, 7, 11, 15, 21, 31, 41, 61], trace=tr, grid=grid)
        return tr

    t.trace(trace)
