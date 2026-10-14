"""Checks for ridge_lasso.py. Runs identically under CPython and Pyodide."""

import numpy as np

from learnml import Grid, Trace


def fuel_data(n=12, seed=1):
    """Speed (km/h) → fuel use (L/100 km): a U-shaped truth plus noise."""
    rng = np.random.default_rng(seed)
    v = np.sort(rng.uniform(20, 120, n))
    y = 4 + 0.0009 * (v - 60) ** 2 + rng.normal(0, 0.4, n)
    return v, y


def poly_design(v, mu=None, sd=None, degree=9):
    u = (v - 70) / 50
    F = np.column_stack([u ** k for k in range(1, degree + 1)])
    if mu is None:
        mu, sd = F.mean(axis=0), F.std(axis=0)
    return (F - mu) / sd, mu, sd


def check(s, t):
    v, y = fuel_data()
    X, mu, sd = poly_design(v)
    ybar = y.mean()
    yc = y - ybar
    lam_l = 0.12  # ½‖·‖² scaling: λ / n = 0.01 per sample

    def t_ridge():
        w = s.ridge(X, yc, 0.5)
        ref = np.linalg.solve(X.T @ X + 0.5 * np.eye(X.shape[1]), X.T @ yc)
        assert np.allclose(w, ref, atol=1e-8), "ridge should solve (XᵀX + λI) w = Xᵀy"
        # The SVD view: w = V diag(d / (d² + λ)) Uᵀ y, i.e. shrink each direction by d² / (d² + λ).
        U, d, Vt = np.linalg.svd(X, full_matrices=False)
        assert np.allclose(w, Vt.T @ (d / (d ** 2 + 0.5) * (U.T @ yc)), atol=1e-8), "should match the SVD shrinkage formula"
        assert np.allclose(s.ridge(X, yc, 0.0), np.linalg.lstsq(X, yc, rcond=None)[0], atol=1e-5), "λ = 0 gives least squares"

    def t_soft():
        z = np.array([-3.0, -0.4, 0.0, 0.7, 2.5])
        got = np.asarray(s.soft_threshold(z, 1.0), dtype=float)
        assert np.allclose(got, [-2.0, 0.0, 0.0, 0.0, 1.5]), f"S_1({z.tolist()}) = {got.tolist()}"
        assert float(s.soft_threshold(-0.5, 0.2)) == -0.3 or np.isclose(float(s.soft_threshold(-0.5, 0.2)), -0.3), "works on scalars"

    def t_lasso_kkt():
        w = s.lasso_cd(X, yc, lam_l, sweeps=3000)
        g = X.T @ (yc - X @ w)  # must equal λ·sign(w_j) on the support and lie in [−λ, λ] off it
        on = np.abs(w) > 1e-10
        assert on.any(), "some coefficients should survive"
        assert np.allclose(g[on], lam_l * np.sign(w[on]), atol=2e-3), "on the support: x_jᵀ(y − Xw) = λ·sign(w_j)"
        assert np.all(np.abs(g[~on]) <= lam_l + 2e-3), "off the support: |x_jᵀ(y − Xw)| ≤ λ"

    def t_lasso_sparse():
        lam_max = float(np.max(np.abs(X.T @ yc)))
        w = s.lasso_cd(X, yc, lam_max * 1.01, sweeps=5)
        assert np.all(w == 0), "for λ ≥ max|Xᵀy| every coefficient is exactly 0"
        w = s.lasso_cd(X, yc, lam_l, sweeps=3000)
        assert 0 < np.sum(w != 0) < X.shape[1], f"lasso should zero out some coefficients, got {np.sum(w != 0)} non-zero"
        objs = []
        s.lasso_cd(X, yc, lam_l, sweeps=30, on_sweep=lambda k, ww: objs.append(s.lasso_objective(X, yc, ww, lam_l)))
        assert all(b <= a + 1e-9 for a, b in zip(objs, objs[1:])), "coordinate descent never increases the objective"

    t.test("岭回归闭式解（并与 SVD 收缩公式对拍）", "Ridge closed form (checked against the SVD shrinkage formula)", t_ridge)
    t.test("软阈值算子", "The soft-thresholding operator", t_soft)
    t.test("Lasso 坐标下降满足 KKT 条件", "Lasso coordinate descent satisfies the KKT conditions", t_lasso_kkt)
    t.test("稀疏性与目标单调下降", "Sparsity and a monotone objective", t_lasso_sparse)

    def trace():
        pts = np.column_stack([v, y])
        grid = Grid(np.array([[20.0, 2.5], [120.0, 8.5]]), n=48, pad=0.0)
        Zg, _, _ = poly_design(grid.points[:, 0], mu, sd)
        tr = Trace("classifier2d", pts, np.ones(len(y)), grid=grid, show_margin=False,
                   title={"zh": "Lasso 坐标下降：逐轮的拟合曲线", "en": "Lasso coordinate descent: the fit sweep by sweep"})
        keep = set(range(10)) | set(range(10, 60, 5)) | set(range(60, 400, 40))

        def on_sweep(k, w):
            if k in keep:
                field = Zg @ w + ybar - grid.points[:, 1]  # zero contour = the fitted curve
                tr.record(field=field, metrics={"objective": s.lasso_objective(X, yc, w, lam_l), "active": float(np.mean(w != 0))},
                          note=f"sweep {k + 1} · non-zero {int(np.sum(w != 0))}/9")

        s.lasso_cd(X, yc, lam_l, sweeps=400, on_sweep=on_sweep)
        return tr

    t.trace(trace)
