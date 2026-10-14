"""Checks for knn_kd.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import SplineTransformer

from learnml import Grid, Trace

KD = [[1, 6], [3, 2], [4, 7], [6, 3], [7, 8], [9, 5]]


def check(s, t):
    X, y01 = make_moons(200, noise=0.3, random_state=0)
    y = 2 * y01 - 1
    T, ty01 = make_moons(500, noise=0.3, random_state=1)
    ty = 2 * ty01 - 1

    def t_dist():
        Q = np.array([[0.0, 0.0]])
        P = np.array([[2, 0], [1.5, 1.5], [0, 2.2]])
        assert np.allclose(s.pairwise(Q, P, 1), [[2, 3, 2.2]]), "L1 distances of the worked example"
        assert np.allclose(s.pairwise(Q, P, 2), [[2, np.sqrt(4.5), 2.2]]), "L2 distances"
        assert np.allclose(s.pairwise(Q, P, np.inf), [[2, 1.5, 2.2]]), "L∞ distances"

    def t_knn():
        for k in (1, 7, 25):
            mine = s.knn_predict(X, y, T, k)
            ref = KNeighborsClassifier(n_neighbors=k).fit(X, y).predict(T)
            agree = np.mean(mine == ref)
            assert agree > 0.99, f"k = {k}: agrees with scikit-learn on {agree:.1%} of test points"
        assert np.all(s.knn_predict(X, y, X, 1) == y), "1-NN must have zero training error"

    def t_kd():
        root = s.build_kd(KD)
        i, d, visited = s.kd_nearest(root, KD, np.array([2.0, 6.5]))
        assert i == 0 and abs(d - np.sqrt(1.25)) < 1e-9, f"nearest {KD[i]}, expected (1, 6)"
        assert visited == [3, 0, 2, 1], f"visited {visited}; the notes visit (6,3), (1,6), (4,7), (3,2) and prune the rest"
        P = X.tolist()
        r = s.build_kd(P)
        for q in T[:40]:
            j, _, _ = s.kd_nearest(r, P, q)
            assert j == int(np.argmin(np.linalg.norm(X - q, axis=1))), "kd search must match brute force"

    def t_spline():
        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(0, 10, 60))
        yy = np.sin(x) + 0.2 * rng.normal(size=60)
        knots = np.linspace(x.min(), x.max(), 7)[1:-1]
        B = s.spline_basis(x, knots)
        assert B.shape == (60, 9), f"basis shape {B.shape}; K = 5 knots should give K + 4 = 9 columns"
        mine = B @ s.fit_spline(x, yy, knots)
        st = SplineTransformer(n_knots=7, degree=3, knots="uniform")
        ref = LinearRegression(fit_intercept=False).fit(st.fit_transform(x[:, None]), yy).predict(st.transform(x[:, None]))
        assert np.allclose(mine, ref, atol=1e-6), "same knots ⇒ same fitted values as scikit-learn's B-spline basis"

    t.test("三种距离", "Three distances", t_dist)
    t.test("kNN 与 scikit-learn 一致", "kNN matches scikit-learn", t_knn)
    t.test("kd 树搜索与剪枝", "kd-tree search and pruning", t_kd)
    t.test("三次样条基", "Cubic spline basis", t_spline)

    def trace():
        grid = Grid(X, n=40)
        tr = Trace("classifier2d", X, y, grid=grid, show_margin=False,
                   title={"zh": "你的 kNN：k 从 1 到 101", "en": "Your kNN: k from 1 to 101"})
        D = s.pairwise(grid.points, X)
        order = np.argsort(D, axis=1, kind="stable")
        for k in (1, 3, 7, 15, 31, 61, 101):
            field = y[order[:, :k]].mean(axis=1)
            acc = float(np.mean(s.knn_predict(X, y, T, k) == ty))
            tr.record(field=field, metrics={"acc": acc}, note=f"k = {k}")
        return tr

    t.trace(trace)
