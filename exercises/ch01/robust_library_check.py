"""Checks for robust_library.py. Runs identically under CPython and Pyodide."""

import numpy as np

from learnml import Trace


def check(s, t):
    rng = np.random.default_rng(4)
    x = rng.uniform(0.3, 9.7, 40)
    y = np.clip(38 + 5.2 * x + rng.normal(0, 6, 40), 0, 100)
    y[:4] = rng.uniform(0, 3, 4)
    X = x.reshape(-1, 1)
    state = {}

    def t_fit():
        models = s.fit_three(X, y)
        names = [type(m).__name__ for m in models]
        assert names == ["LinearRegression", "HuberRegressor", "QuantileRegressor"], f"got {names}"
        assert models[2].quantile == 0.5 and models[2].alpha == 0, "median regression: quantile=0.5, alpha=0"
        state["models"] = models

    def t_robust():
        assert "models" in state, "fix the previous test first"
        clean = np.polyfit(x[4:], y[4:], 1)[0]  # slope without the absent students
        slopes = [float(m.coef_[0]) for m in state["models"]]
        err = [abs(b - clean) for b in slopes]
        assert err[1] < err[0] and err[2] < err[0], (
            f"slopes OLS/Huber/LAD = {np.round(slopes, 2)}; robust ones should be closer to the clean-data slope {clean:.2f}")

    t.test("拟合三种回归", "Fitting the three regressions", t_fit)
    t.test("稳健损失受缺考学生影响更小", "Robust losses are less affected by the absentees", t_robust)

    def trace():
        tr = Trace("classifier2d", np.column_stack([x, y]), np.ones(len(y)), show_margin=False,
                   title={"zh": "同一模型，三种策略", "en": "One model, three strategies"})
        for name, m in zip(["OLS (squared)", "Huber", "LAD (absolute)"], s.fit_three(X, y)):
            tr.record(w=[float(m.coef_[0]), -1.0], b=float(m.intercept_), note=name)
        return tr

    t.trace(trace)
