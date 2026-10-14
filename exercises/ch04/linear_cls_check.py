"""Checks for linear_cls.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from learnml import Grid, Trace


def separable(seed=6, n=24):
    """Two separable blobs of patients in shuffled order, labels in {−1, +1} (17 perceptron updates)."""
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal([0.8, 0.78], 0.45, (n // 2, 2)), rng.normal([-0.8, -0.18], 0.45, (n // 2, 2))])
    y = np.array([1.0] * (n // 2) + [-1.0] * (n // 2))
    p = rng.permutation(n)
    return X[p], y[p]


def overlapping(seed=2, n=60):
    """Two overlapping blobs, labels in {0, 1}."""
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal([0.8, 0.6], 0.9, (n // 2, 2)), rng.normal([-0.8, -0.4], 0.9, (n // 2, 2))])
    return X, np.array([1.0] * (n // 2) + [0.0] * (n // 2))


def check(s, t):
    def t_example():
        X = np.array([[1.0, 3.0], [4.0, 1.0], [1.0, 0.0]])
        w, b, k = s.perceptron(X, np.array([1.0, 1.0, -1.0]))
        assert k == 6, f"{k} updates; the worked example of the notes makes 6"
        assert np.allclose(w, [1, 4]) and np.isclose(b, -2), f"w = {w}, b = {b}; expected w = (1, 4), b = −2"

    def t_separates():
        X, y = separable()
        w, b, k = s.perceptron(X, y)
        m = y * (X @ w + b)
        assert np.all(m > 0), f"{np.sum(m <= 0)} points still misclassified after {k} updates"

    def t_sigmoid():
        z = np.array([-1000.0, -2.0, 0.0, 2.0, 1000.0])
        with np.errstate(over="raise"):
            p = s.sigmoid(z)
        assert np.allclose(p, [0.0, 1 / (1 + np.e**2), 0.5, 1 / (1 + np.e**-2), 1.0]), f"σ(z) = {p}"

    def t_grad():
        X, y = overlapping()
        Xa = s.augment(X)
        w = np.array([0.4, -0.3, 0.2])
        g = s.gradient(w, Xa, y)
        fd = np.array([(s.nll(w + e, Xa, y) - s.nll(w - e, Xa, y)) / 2e-6 for e in 1e-6 * np.eye(3)])
        assert np.allclose(g, fd, atol=1e-4), f"gradient {np.round(g, 4)} vs finite differences {np.round(fd, 4)}"

    def t_hess():
        X, y = overlapping()
        Xa = s.augment(X)
        w = np.array([0.4, -0.3, 0.2])
        H = s.hessian(w, Xa)
        fd = np.column_stack([(s.gradient(w + e, Xa, y) - s.gradient(w - e, Xa, y)) / 2e-6 for e in 1e-6 * np.eye(3)])
        assert np.allclose(H, fd, atol=1e-4), "Hessian does not match finite differences of the gradient"
        assert np.linalg.eigvalsh(H).min() >= -1e-10, "the Hessian of the NLL should be positive semidefinite"

    def t_newton():
        X, y = overlapping()
        w, b = s.newton(X, y, steps=8)
        ref = LogisticRegression(C=1e10, tol=1e-10, max_iter=1000).fit(X, y)
        assert np.allclose(w, ref.coef_[0], atol=1e-4) and abs(b - ref.intercept_[0]) < 1e-4, (
            f"w = {np.round(w, 4)}, b = {b:.4f}; scikit-learn: {np.round(ref.coef_[0], 4)}, {ref.intercept_[0]:.4f}"
        )

    t.test("例题：6 次更新得到 w = (1, 4), b = −2", "Worked example: 6 updates to w = (1, 4), b = −2", t_example)
    t.test("可分数据上分对所有点", "Separates separable data", t_separates)
    t.test("数值稳定的 sigmoid", "Numerically stable sigmoid", t_sigmoid)
    t.test("梯度与有限差分一致", "Gradient matches finite differences", t_grad)
    t.test("Hessian 正确且半正定", "Hessian is correct and positive semidefinite", t_hess)
    t.test("牛顿法与 scikit-learn 一致", "Newton matches scikit-learn", t_newton)

    def perceptron_trace():
        X, y = separable()
        tr = Trace("classifier2d", X, y, show_margin=False, title={"zh": "你的感知机：每次更新", "en": "Your perceptron: every update"})
        s.perceptron(X, y, trace=tr)
        return tr

    def newton_trace():
        X, y = overlapping()
        grid = Grid(X, n=40)
        tr = Trace("classifier2d", X, 2 * y - 1, grid=grid, show_margin=False,
                   title={"zh": "你的牛顿法：概率场逐步变陡", "en": "Your Newton's method: the probability field sharpens"})
        Xa = s.augment(X)
        w = np.zeros(3)
        for k in range(7):
            tr.record(w=w[:2], b=w[2], field=s.sigmoid(s.augment(grid.points) @ w) - 0.5,
                      metrics={"nll": s.nll(w, Xa, y)}, note=f"Newton {k}")
            w = w - np.linalg.solve(s.hessian(w, Xa), s.gradient(w, Xa, y))
        return tr

    t.trace(perceptron_trace)
    t.trace(newton_trace)
