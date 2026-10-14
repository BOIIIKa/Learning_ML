"""Checks for pca_power.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

from learnml import Trace


def check(s, t):
    digits = load_digits().data

    def t_example():
        v, lam = s.power_iteration(np.array([[2.0, 1.0], [1.0, 2.0]]), v0=[1.0, 0.0])
        assert abs(lam - 3) < 1e-9 and abs(abs(v[0]) - np.sqrt(0.5)) < 1e-6, f"λ = {lam:.4f}, v = {np.round(v, 4)}; the notes get 3 and (1,1)/√2"

    def t_sklearn():
        comps, var, scores = s.pca(digits, 5)
        ref = PCA(5).fit(digits)
        assert np.allclose(var, ref.explained_variance_ * (len(digits) - 1) / len(digits), rtol=1e-8), "variances = s² / n"
        assert np.allclose(np.abs(np.sum(comps * ref.components_, axis=1)), 1, atol=1e-8), "same directions as scikit-learn (up to sign)"

    def t_eckart():
        Xc = digits - digits.mean(axis=0)
        sv = np.linalg.svd(Xc, compute_uv=False)
        for k in (2, 10, 30):
            err = np.linalg.norm(digits - s.reconstruct(digits, k))
            assert abs(err - np.sqrt(np.sum(sv[k:] ** 2))) < 1e-6 * err, f"k = {k}: error {err:.4f} should equal the tail of the singular values"

    t.test("例题：幂迭代得到 3 与 (1,1)/√2", "Worked example: power iteration finds 3 and (1,1)/√2", t_example)
    t.test("与 scikit-learn 的 PCA 一致", "Matches scikit-learn's PCA", t_sklearn)
    t.test("Eckart–Young：误差等于奇异值尾部", "Eckart–Young: error equals the tail of the singular values", t_eckart)

    def trace():
        rng = np.random.default_rng(0)
        A = rng.normal(size=(120, 2)) * [1.5, 0.45]
        c, sn = np.cos(0.55), np.sin(0.55)
        X = A @ np.array([[c, sn], [-sn, c]])
        X -= X.mean(axis=0)
        y = np.where(X @ np.array([c, sn]) > 0, 1, -1)
        S = X.T @ X / len(X)
        tr = Trace("classifier2d", X, y, show_margin=False,
                   title={"zh": "幂迭代：直线是当前方向 v", "en": "Power iteration: the line is the current direction v"})
        # A line through the origin along v has normal (−v₂, v₁).
        s.power_iteration(S, iters=8, v0=[0.1, -1.0], trace=lambda t_, v, q: tr.record(w=[-v[1], v[0]], b=0.0, metrics={"variance": q}, note=f"iter {t_ + 1}"))
        return tr

    t.trace(trace)
