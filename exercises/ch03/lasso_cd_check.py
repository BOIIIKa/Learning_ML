"""Checks for lasso_cd.py. Runs identically under CPython and Pyodide."""

import numpy as np

from learnml import Grid, Trace


def study_data(seed=2, n=30):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.2, 9.8, n))
    y = 30 + 60 / (1 + np.exp(-(x - 4.5) / 1.3)) + rng.normal(0, 7, n)
    return x, y


def poly(x, d=9):
    u = (np.asarray(x, dtype=float) - 5) / 5
    return np.column_stack([u ** k for k in range(1, d + 1)])


def check(s, t):
    x, y = study_data()
    F = poly(x)
    mu, sd = F.mean(axis=0), F.std(axis=0)
    Z = (F - mu) / sd
    ybar = y.mean()
    yc = y - ybar

    def t_ridge():
        lam = 0.3
        w = s.ridge(Z, yc, lam)
        n = len(yc)
        grad = -Z.T @ (yc - Z @ w) / n + lam * w
        assert np.allclose(grad, 0, atol=1e-8), "the ridge gradient should vanish at the solution"

    def t_soft():
        z = np.array([-3.0, -0.5, 0.0, 0.4, 2.5])
        got = s.soft_threshold(z, 1.0)
        assert np.allclose(got, [-2.0, 0.0, 0.0, 0.0, 1.5]), f"S(z, 1) = {got}"

    def t_kkt():
        lam = 0.5
        w = s.lasso_cd(Z, yc, lam, sweeps=5000)
        g = Z.T @ (yc - Z @ w) / len(yc)
        act = np.abs(w) > 0
        assert act.sum() >= 1, "some coefficients should be nonzero at λ = 0.5"
        assert np.all(np.abs(g[~act]) <= lam + 1e-6), "inactive: |xⱼᵀr/n| ≤ λ"
        assert np.allclose(g[act], lam * np.sign(w[act]), atol=1e-4), "active: xⱼᵀr/n = λ·sign(wⱼ)"

    def t_limits():
        w0 = s.lasso_cd(Z[:, :3], yc, 0.0, sweeps=20000)
        ols = np.linalg.lstsq(Z[:, :3], yc, rcond=None)[0]
        assert np.allclose(w0, ols, atol=1e-3), "λ = 0 should give least squares"
        big = np.max(np.abs(Z.T @ yc)) / len(yc)
        assert np.all(s.lasso_cd(Z, yc, big * 1.01) == 0), "λ ≥ max|xⱼᵀy|/n should give w = 0"

    t.test("岭回归闭式解", "Ridge closed form", t_ridge)
    t.test("软阈值", "Soft thresholding", t_soft)
    t.test("Lasso 的 KKT 条件", "Lasso KKT conditions", t_kkt)
    t.test("两个极端：λ = 0 与 λ 很大", "The two extremes: λ = 0 and large λ", t_limits)

    def trace():
        lams = np.logspace(np.log10(25), -3, 30)
        ws = s.lasso_path(Z, yc, lams)
        P = np.column_stack([x, y])
        grid = Grid(P, n=50, pad=1.0)
        gx, gy = grid.points[:, 0], grid.points[:, 1]
        Zg = (poly(gx) - mu) / sd
        tr = Trace("classifier2d", P, np.ones(len(y)), grid=grid, show_margin=False,
                   title={"zh": "Lasso 路径：λ 从大到小", "en": "Lasso path: λ from large to small"})
        for lam, w in zip(lams, ws):
            tr.record(field=ybar + Zg @ w - gy, metrics={"nnz": int(np.sum(np.abs(w) > 0))}, note=f"λ = {lam:.3g}")
        return tr

    t.trace(trace)
