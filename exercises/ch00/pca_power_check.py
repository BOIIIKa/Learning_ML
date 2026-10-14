"""Checks for pca_power.py. Runs identically under CPython and Pyodide."""

import numpy as np

from learnml import Trace


def cloud(seed=12, n=60):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    return np.column_stack([0.9 * z[:, 0] + 0.3, 0.65 * z[:, 0] + 0.45 * z[:, 1] - 0.2])


def check(s, t):
    X = cloud()

    def t_cov():
        S = s.mle_covariance(X)
        assert np.allclose(S, np.cov(X.T, bias=True)), "should equal np.cov(X.T, bias=True) — divide by n"

    def t_power():
        S = np.array([[3.0, 1.0], [1.0, 2.0]])
        v = s.power_step(S, np.array([1.0, 0.0]))
        assert np.isclose(np.linalg.norm(v), 1.0), "result must have unit length"
        assert np.allclose(v, np.array([3.0, 1.0]) / np.sqrt(10)), f"got {np.round(v, 4)}"

    def t_rayleigh():
        S = np.array([[3.0, 1.0], [1.0, 2.0]])
        v = np.array([1.0, 1.0]) / np.sqrt(2)
        assert np.isclose(s.rayleigh(S, v), 3.5), f"vᵀSv = {s.rayleigh(S, v)}, expected 3.5"

    def t_deflate():
        S = np.diag([4.0, 1.0])
        assert np.allclose(s.deflate(S, 4.0, np.array([1.0, 0.0])), np.diag([0.0, 1.0])), "λ v vᵀ removed"

    def t_end_to_end():
        values, vectors = s.principal_directions(X)
        ref_vals, ref_vecs = np.linalg.eigh(np.cov(X.T, bias=True))
        assert np.allclose(values, ref_vals[::-1], atol=1e-6), f"eigenvalues {np.round(values, 4)} vs {np.round(ref_vals[::-1], 4)}"
        for v, r in zip(vectors, ref_vecs.T[::-1]):
            assert abs(abs(v @ r) - 1) < 1e-6, "eigenvectors should match up to sign"

    t.test("极大似然协方差", "Maximum-likelihood covariance", t_cov)
    t.test("幂迭代一步", "One power-iteration step", t_power)
    t.test("瑞利商", "Rayleigh quotient", t_rayleigh)
    t.test("收缩（deflation）", "Deflation", t_deflate)
    t.test("与 numpy.linalg.eigh 一致", "Matches numpy.linalg.eigh", t_end_to_end)

    def trace():
        tr = Trace("classifier2d", X, np.ones(len(X)), title={"zh": "幂迭代：方向逐步转向主轴", "en": "Power iteration turning toward the principal axis"},
                   show_margin=False)
        s.principal_directions(X, trace=tr)
        return tr

    t.trace(trace)
