"""Checks for erm.py. Runs identically under CPython and Pyodide."""

import numpy as np

from learnml import Trace


def study_data(seed=4, n=40):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.3, 9.7, n)
    y = np.clip(38 + 5.2 * x + rng.normal(0, 6, n), 0, 100)
    y[:4] = rng.uniform(0, 3, 4)  # absent students
    return x, y


def check(s, t):
    x, y = study_data()
    X = s.design(x)

    def t_sq():
        w = np.array([40.0, 5.0])
        assert np.isclose(s.squared_risk(w, X, y), np.mean((X @ w - y) ** 2)), "mean of squared residuals"

    def t_abs():
        w = np.array([40.0, 5.0])
        assert np.isclose(s.absolute_risk(w, X, y), np.mean(np.abs(X @ w - y))), "mean of absolute residuals"

    def t_grad():
        w = np.array([30.0, 2.0])
        g = s.squared_grad(w, X, y)
        eps = 1e-5
        num = [(s.squared_risk(w + eps * e, X, y) - s.squared_risk(w - eps * e, X, y)) / (2 * eps) for e in np.eye(2)]
        assert np.allclose(g, num, rtol=1e-4), f"gradient {np.round(g, 3)} vs finite differences {np.round(num, 3)}"

    def t_fit():
        w = s.fit_gd(X, y)
        ref = np.linalg.lstsq(X, y, rcond=None)[0]
        assert np.allclose(w, ref, atol=0.05), f"w = {np.round(w, 3)}, least squares = {np.round(ref, 3)}"

    t.test("平方经验风险", "Squared empirical risk", t_sq)
    t.test("绝对值经验风险", "Absolute empirical risk", t_abs)
    t.test("梯度（与数值差分对比）", "Gradient (vs finite differences)", t_grad)
    t.test("梯度下降收敛到最小二乘解", "Gradient descent reaches the least-squares solution", t_fit)

    def trace():
        tr = Trace("classifier2d", np.column_stack([x, y]), np.ones(len(y)), show_margin=False,
                   title={"zh": "梯度下降中的直线", "en": "The line during gradient descent"})
        s.fit_gd(X, y, trace=tr)
        return tr

    t.trace(trace)
