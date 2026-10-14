"""Checks for sparse_library.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from learnml import Grid, Trace


def check(s, t):
    data = load_diabetes()
    X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.3, random_state=0)
    state = {}

    def t_build():
        ridge, lasso = s.build_models()
        assert isinstance(ridge, Pipeline) and isinstance(ridge[-1], RidgeCV), "ridge should be a pipeline ending in RidgeCV"
        assert isinstance(lasso, Pipeline) and isinstance(lasso[-1], LassoCV), "lasso should be a pipeline ending in LassoCV"
        assert len(ridge[-1].alphas) == 30, "RidgeCV should search 30 alphas"

    def t_fit():
        ridge, lasso = s.build_models()
        ridge.fit(X_tr, y_tr)
        lasso.fit(X_tr, y_tr)
        for name, m in (("ridge", ridge), ("lasso", lasso)):
            mse = np.mean((m.predict(X_te) - y_te) ** 2)
            assert mse < 3400, f"{name} test MSE {mse:.0f} should be below 3400"
        state["lasso"] = lasso

    def t_selected():
        lasso = state.get("lasso")
        assert lasso is not None, "fix the previous test first"
        names = s.selected(lasso, list(data.feature_names))
        expect = [n for n, c in zip(data.feature_names, lasso[-1].coef_) if c != 0]
        assert names == expect, f"{names} vs {expect}"
        assert "bmi" in names and len(names) < 10, "bmi should survive and at least one feature should be dropped"

    t.test("组装两条管道", "Assembling the two pipelines", t_build)
    t.test("训练并在测试集上评估", "Fitting and evaluating on the test set", t_fit)
    t.test("Lasso 选中的特征", "Features kept by the lasso", t_selected)

    def trace():
        rng = np.random.default_rng(2)
        x = np.sort(rng.uniform(0.2, 9.8, 30))
        y = 30 + 60 / (1 + np.exp(-(x - 4.5) / 1.3)) + rng.normal(0, 7, 30)
        F = np.column_stack([((x - 5) / 5) ** k for k in range(1, 10)])
        mu, sd = F.mean(axis=0), F.std(axis=0)
        Z = (F - mu) / sd
        alphas = np.logspace(np.log10(25), -3, 30)
        coefs = s.path(Z, y - y.mean(), alphas)
        P = np.column_stack([x, y])
        grid = Grid(P, n=50, pad=1.0)
        gx, gy = grid.points[:, 0], grid.points[:, 1]
        Zg = (np.column_stack([((gx - 5) / 5) ** k for k in range(1, 10)]) - mu) / sd
        tr = Trace("classifier2d", P, np.ones(len(y)), grid=grid, show_margin=False,
                   title={"zh": "sklearn.lasso_path 回放", "en": "Replaying sklearn.lasso_path"})
        for a, w in zip(alphas, coefs):
            tr.record(field=y.mean() + Zg @ w - gy, metrics={"nnz": int(np.sum(w != 0))}, note=f"α = {a:.3g}")
        return tr

    t.trace(trace)
