"""Checks for impl.py. Runs identically under CPython and Pyodide (keep to numpy / sklearn)."""

import numpy as np

from learnml import Trace


def data(seed=0, n=40):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    return np.column_stack([np.ones(n), x]), 1 + 2 * x + rng.normal(0, 0.2, n), x


def check(s, t):
    X, y, x = data()

    def t_loss():
        w = np.array([0.5, 1.0])
        assert np.isclose(s.loss(w, X, y), np.mean((X @ w - y) ** 2)), "mean of squared residuals"

    def t_fit():
        w = s.fit(X, y)
        ref = np.linalg.lstsq(X, y, rcond=None)[0]
        assert np.allclose(w, ref, atol=1e-2), f"w = {np.round(w, 3)}, least squares = {np.round(ref, 3)}"

    # Test names are bilingual; messages should say what was expected.
    t.test("损失函数", "Loss", t_loss)
    t.test("梯度下降收敛", "Gradient descent converges", t_fit)

    def trace():
        tr = Trace("classifier2d", np.column_stack([x, y]), np.ones(len(y)), show_margin=False,
                   title={"zh": "TODO(scaffold) 回放标题", "en": "TODO(scaffold) Replay title"})
        s.fit(X, y, trace=tr)
        return tr

    t.trace(trace)
