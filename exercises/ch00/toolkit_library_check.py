"""Checks for toolkit_library.py. Runs identically under CPython and Pyodide."""

import numpy as np

from learnml import Trace


def check(s, t):
    rng = np.random.default_rng(12)
    z = rng.normal(size=(60, 2))
    X = np.column_stack([0.9 * z[:, 0] + 0.3, 0.65 * z[:, 0] + 0.45 * z[:, 1] - 0.2])

    def t_pca():
        comps, var = s.pca_directions(X)
        vals, vecs = np.linalg.eigh(np.cov(X.T))  # sklearn uses n − 1
        assert np.allclose(var, vals[::-1]), f"variances {np.round(var, 4)}"
        for v, r in zip(comps, vecs.T[::-1]):
            assert abs(abs(v @ r) - 1) < 1e-8, "directions should match the covariance eigenvectors up to sign"

    def t_project():
        a, c = np.array([1.0, 1.4]), 1.0
        for target in ([1.8, 1.6], [-1.0, 0.0]):
            target = np.array(target)
            x = s.project_halfplane(target, a, c)
            viol = a @ target - c
            expected = target - max(0.0, viol) / (a @ a) * a  # closed form from the KKT conditions
            assert np.allclose(x, expected, atol=1e-5), f"x* = {np.round(x, 4)}, expected {np.round(expected, 4)}"

    t.test("用 scikit-learn 做 PCA", "PCA with scikit-learn", t_pca)
    t.test("用 SciPy 解约束优化，与 KKT 闭式解一致", "Constrained optimization with SciPy matches the KKT closed form", t_project)

    def trace():
        comps, _ = s.pca_directions(X)
        mean = X.mean(axis=0)
        tr = Trace("classifier2d", X, np.ones(len(X)), title={"zh": "两个主方向", "en": "The two principal directions"}, show_margin=False)
        for k, v in enumerate(comps):
            normal = np.array([-v[1], v[0]])
            tr.record(w=normal, b=-normal @ mean, note=f"PC{k + 1}")
        return tr

    t.trace(trace)
