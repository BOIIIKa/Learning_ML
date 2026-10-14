"""Checks for kmeans.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.cluster import KMeans

from learnml import Grid, Trace


def blobs(n=150, seed=0):
    rng = np.random.default_rng(seed)
    C = np.array([[-1.4, -0.6], [1.3, -0.9], [0.1, 1.4]])
    g = np.arange(n) % 3
    return C[g] + rng.normal(0, 0.55, (n, 2)), g


def check(s, t):
    X, g = blobs()

    def t_example():
        x = np.array([[1.0], [2], [4], [7], [8], [12]])
        labels, C = s.lloyd(x, np.array([[1.0], [2.0]]))
        assert labels.tolist() == [0, 0, 0, 1, 1, 1], f"labels {labels.tolist()}"
        assert np.allclose(C.ravel(), [7 / 3, 9]) and abs(s.objective(x, labels, C) - 56 / 3) < 1e-9, "the notes end at centres 7/3 and 9, J = 56/3"

    def t_monotone():
        rng = np.random.default_rng(1)
        objs = []
        s.lloyd(X, X[rng.choice(len(X), 3, replace=False)], trace=lambda t_, lab, C: objs.append(s.objective(X, lab, C)))
        assert all(b <= a + 1e-9 for a, b in zip(objs, objs[1:])), f"objective must not increase: {np.round(objs, 2)}"

    def t_pp():
        rng = np.random.default_rng(0)
        C = s.kmeans_pp(X, 3, rng)
        assert C.shape == (3, 2) and len({tuple(c) for c in C}) == 3, "three distinct data points"
        far = np.mean([len(set(s.assign(s.kmeans_pp(X, 3, np.random.default_rng(r)), np.array([[-1.4, -0.6], [1.3, -0.9], [0.1, 1.4]])))) for r in range(20)])
        assert far > 2.6, f"k-means++ should usually put one centre near each group ({far:.2f} of 3 on average)"

    def t_sklearn():
        best = min((s.lloyd(X, s.kmeans_pp(X, 3, np.random.default_rng(r))) for r in range(5)), key=lambda lc: s.objective(X, *lc))
        ref = KMeans(3, n_init=10, random_state=0).fit(X)
        assert abs(s.objective(X, *best) - ref.inertia_) < 1e-6 * ref.inertia_ + 1e-9, f"objective {s.objective(X, *best):.3f} vs sklearn {ref.inertia_:.3f}"

    t.test("例题：收敛到 7/3 与 9", "Worked example: converges to 7/3 and 9", t_example)
    t.test("目标单调不增", "The objective never increases", t_monotone)
    t.test("k-means++ 初始化", "k-means++ seeding", t_pp)
    t.test("与 scikit-learn 的目标值一致", "Same objective as scikit-learn", t_sklearn)

    def trace():
        rng = np.random.default_rng(3)
        X2 = np.vstack([rng.normal([-1.2, 0], 0.6, (60, 2)), rng.normal([1.2, 0.3], 0.6, (60, 2))])
        y = np.array([-1] * 60 + [1] * 60)
        grid = Grid(X2, n=40)
        tr = Trace("classifier2d", X2, y, grid=grid, show_margin=False,
                   title={"zh": "你的 k 均值（k = 2）：边界是两中心的垂直平分线", "en": "Your k-means (k = 2): the boundary bisects the two centres"})

        def rec(t_, lab, C):
            field = ((grid.points - C[0]) ** 2).sum(axis=1) - ((grid.points - C[1]) ** 2).sum(axis=1)
            tr.record(field=field, highlight=[int(i) for i in np.flatnonzero(lab == 1)], metrics={"wcss": s.objective(X2, lab, C)}, note=f"iter {t_ + 1}")

        s.lloyd(X2, np.array([[-2.0, 2.0], [-1.5, 1.8]]), trace=rec)
        return tr

    t.trace(trace)
