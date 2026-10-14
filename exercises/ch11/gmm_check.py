"""Checks for gmm.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.mixture import GaussianMixture

from learnml import Grid, Trace


def shaped(n=300, seed=0):
    rng = np.random.default_rng(seed)
    A = [np.array([[0.9, 0], [0.55, 0.2]]), np.eye(2) * 0.35, np.array([[0.25, 0], [-0.3, 0.55]])]
    C = np.array([[-1.4, -0.4], [1.4, -0.8], [0.3, 1.5]])
    g = np.arange(n) % 3
    Z = rng.normal(size=(n, 2))
    return np.array([C[k] + A[k] @ z for k, z in zip(g, Z)]), g


def check(s, t):
    X, g = shaped()
    w0, m0, S0 = np.ones(3) / 3, np.array([[-1.0, 0.0], [1.0, -0.5], [0.0, 1.0]]), np.array([np.eye(2)] * 3)

    def t_example():
        x = np.array([[-1.0], [0.0], [3.0], [4.0]])
        R, _ = s.e_step(x, np.array([0.5, 0.5]), np.array([[0.0], [2.0]]), np.array([[[1.0]], [[1.0]]]))
        assert np.allclose(R[:, 1], 1 / (1 + np.exp(-(2 * x[:, 0] - 2))), atol=1e-12), "with equal weights and unit variances r₂ = σ(2x − 2)"
        _, m, _ = s.m_step(x, R)
        assert np.allclose(m.ravel(), [-0.4875, 3.2683], atol=1e-3), f"means {np.round(m.ravel(), 4)}; the notes get −0.488 and 3.268"

    def t_monotone():
        _, lls = s.em(X, w0, m0, S0, iters=60)
        assert all(b >= a - 1e-8 for a, b in zip(lls, lls[1:])), "EM must never decrease the log-likelihood"

    def t_sklearn():
        (w, m, S), lls = s.em(X, w0, m0, S0, iters=300, tol=1e-10)
        ref = GaussianMixture(3, weights_init=w0, means_init=m0, precisions_init=np.array([np.eye(2)] * 3), tol=1e-10, max_iter=300).fit(X)
        assert abs(lls[-1] / len(X) - ref.score(X)) < 1e-4, f"average log-likelihood {lls[-1] / len(X):.5f} vs sklearn {ref.score(X):.5f}"

    def t_recover():
        (w, m, S), _ = s.em(X, w0, m0, S0, iters=300)
        R, _ = s.e_step(X, w, m, S)
        lab = R.argmax(axis=1)
        agree = max(np.mean(lab == g), 0)
        from itertools import permutations
        agree = max(np.mean(np.array(p)[lab] == g) for p in permutations(range(3)))
        assert agree > 0.9, f"recovers the subtypes for {agree:.1%} of patients"

    t.test("例题：一步 EM", "Worked example: one EM step", t_example)
    t.test("似然单调上升", "The log-likelihood never decreases", t_monotone)
    t.test("与 scikit-learn 一致", "Matches scikit-learn", t_sklearn)
    t.test("找回三个亚型", "Recovers the three subtypes", t_recover)

    def trace():
        rng = np.random.default_rng(1)
        X2 = np.vstack([rng.normal([-1, 0], [1.0, 0.35], (80, 2)), rng.normal([1.2, 0.4], [0.35, 0.8], (80, 2))])
        y = np.array([-1] * 80 + [1] * 80)
        grid = Grid(X2, n=40)
        tr = Trace("classifier2d", X2, y, grid=grid, show_margin=False,
                   title={"zh": "你的 EM（两分量）：色带是 log r₂/r₁", "en": "Your EM (two components): shading is log r₂/r₁"})

        def rec(t_, R, params, ll):
            w, m, S = params
            lg = np.column_stack([np.log(w[k]) + s.log_gauss(grid.points, m[k], S[k]) for k in range(2)])
            tr.record(field=lg[:, 1] - lg[:, 0], metrics={"loglik": ll}, note=f"EM {t_}")

        s.em(X2, np.array([0.5, 0.5]), np.array([[0.0, 1.0], [0.3, 1.2]]), np.array([np.eye(2)] * 2), iters=25, trace=rec)
        return tr

    t.trace(trace)
