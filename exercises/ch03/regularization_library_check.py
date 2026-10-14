"""Checks for regularization_library.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.linear_model import Lasso, LassoCV, Ridge

from learnml import Grid, Trace


def fuel_data(n=30, seed=7):
    rng = np.random.default_rng(seed)
    v = rng.uniform(20, 120, n)
    y = 4 + 0.0009 * (v - 60) ** 2 + rng.normal(0, 0.4, n)
    return v, y


def check(s, t):
    v, y = fuel_data()
    X = ((v - 70) / 50).reshape(-1, 1)  # scaled speed u ∈ [−1, 1]
    alphas = np.logspace(-4, 0, 25)
    state = {}

    def t_pipeline():
        m = s.make_model("ridge", 0.1)
        names = [type(step).__name__ for step in m]
        assert names == ["PolynomialFeatures", "StandardScaler", "Ridge"], f"steps = {names}"
        assert m[0].degree == 9 and not m[0].include_bias, "PolynomialFeatures(9, include_bias=False)"
        m = s.make_model("lasso", 0.1)
        assert isinstance(m[-1], Lasso) and m[-1].alpha == 0.1, "the lasso model should be Lasso(alpha=alpha)"
        assert isinstance(s.make_model("ridge", 2.0)[-1], Ridge), "kind='ridge' gives Ridge"

    def t_path():
        P = np.asarray(s.coef_path("lasso", X, y, alphas))
        assert P.shape == (len(alphas), 9), f"shape {P.shape}, expected ({len(alphas)}, 9)"
        nnz = (np.abs(P) > 0).sum(axis=1)
        assert nnz[0] > nnz[-1], "more alpha → fewer non-zero coefficients"
        assert nnz[-1] <= 2, f"alpha = 1 should leave at most 2 non-zero coefficients, got {nnz[-1]}"
        R = np.asarray(s.coef_path("ridge", X, y, [1e-3, 1.0, 100.0]))
        norms = np.linalg.norm(R, axis=1)
        assert norms[0] > norms[1] > norms[2], "ridge: the coefficient norm shrinks as alpha grows"
        assert np.all(R != 0), "ridge shrinks but does not zero out coefficients"
        state["path"] = P

    def t_cv():
        m = s.pick_alpha(X, y, alphas)
        assert isinstance(m[-1], LassoCV), "the last step should be LassoCV"
        a = float(m[-1].alpha_)
        assert alphas[0] < a < alphas[-1], f"CV should pick an interior alpha, got {a:.2e}"
        mse = m[-1].mse_path_.mean(axis=1)
        assert np.isclose(mse.min(), mse[list(m[-1].alphas_).index(a)]), "alpha_ minimizes the mean CV error"
        state["cv"] = m

    t.test("组装管道：多项式特征 → 标准化 → 正则回归", "Pipeline: polynomial features → scaling → regularized regression", t_pipeline)
    t.test("系数路径：Lasso 变稀疏，岭回归只收缩", "Coefficient paths: lasso goes sparse, ridge only shrinks", t_path)
    t.test("LassoCV 选出内部的 alpha", "LassoCV picks an interior alpha", t_cv)

    def trace():
        grid = Grid(np.array([[20.0, 2.5], [120.0, 8.5]]), n=48, pad=0.0)
        Ug = ((grid.points[:, 0] - 70) / 50).reshape(-1, 1)
        cvm = s.pick_alpha(X, y, alphas)[-1]
        cv_mse = dict(zip(np.round(cvm.alphas_, 12), cvm.mse_path_.mean(axis=1)))
        tr = Trace("classifier2d", np.column_stack([v, y]), np.ones(len(y)), grid=grid, show_margin=False,
                   title={"zh": "Lasso 正则化路径（alpha 从大到小）", "en": "Lasso regularization path (alpha from large to small)"})
        for a in alphas[::-1]:
            m = s.make_model("lasso", a).fit(X, y)
            field = m.predict(Ug) - grid.points[:, 1]
            tr.record(field=field, metrics={"cv": float(cv_mse.get(np.round(a, 12), np.nan)),
                                            "train": float(np.mean((m.predict(X) - y) ** 2))},
                      note=f"alpha = {a:.1e} · non-zero {int(np.sum(m[-1].coef_ != 0))}/9")
        return tr

    t.trace(trace)
