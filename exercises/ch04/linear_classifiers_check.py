"""Checks for linear_classifiers.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from learnml import Grid, Trace


def separable(seed=10, n=16, gap=0.2):
    """Patients labelled by a known line, keeping only points at least `gap` away from it."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    while len(X) < n:
        p = rng.uniform(-2.5, 2.5, 2)
        d = 0.8 * p[0] + 0.6 * p[1] - 0.2
        if abs(d) >= gap:
            X.append(p)
            y.append(1.0 if d > 0 else -1.0)
    return np.array(X), np.array(y)


def patients(seed=11, n=60):
    """Two overlapping groups: standardized glucose and BMI; t = 1 for diabetes."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) % 2
    centers = np.where(t[:, None] == 1, [0.8, 0.6], [-0.75, -0.55])
    return centers + 0.78 * rng.normal(size=(n, 2)), t.astype(float)


def check(s, t):
    def t_mistake():
        w, b = np.array([1.0, -1.0]), 0.5
        assert not s.is_mistake(w, b, np.array([2.0, 0.0]), 1), "y(w·x + b) = 2.5 > 0 is correct"
        assert s.is_mistake(w, b, np.array([2.0, 0.0]), -1), "y(w·x + b) = −2.5 is a mistake"
        assert s.is_mistake(w, 0.0, np.array([1.0, 1.0]), 1), "a point on the boundary counts as a mistake"

    def t_update():
        w, b = s.perceptron_update(np.array([1.0, 2.0]), 0.5, np.array([3.0, -1.0]), -1, 0.5)
        assert np.allclose(w, [-0.5, 2.5]) and np.isclose(b, 0.0), f"got w = {w}, b = {b}"

    def t_perceptron():
        # The worked example from the notes: three updates, on x1, x3, x3.
        X = np.array([[2.0, 2.0], [0.0, 2.0], [1.0, 0.0]])
        y = np.array([1.0, 1.0, -1.0])
        w, b = s.perceptron(X, y)
        assert np.allclose(w, [0, 2]) and np.isclose(b, -1), f"w = {w}, b = {b}; expected w = (0, 2), b = −1"
        Xs, ys = separable()
        w, b = s.perceptron(Xs, ys, eta=0.5)
        assert np.all(ys * (Xs @ w + b) > 0), "on separable data the final line must separate every point"

    def t_sigmoid():
        z = np.array([-800.0, -1.0, 0.0, 2.0, 800.0])
        with np.errstate(over="raise"):
            p = s.sigmoid(z)
        assert np.allclose(p, [0, 1 / (1 + np.e), 0.5, 1 / (1 + np.exp(-2)), 1]), f"σ = {p}"

    def t_grad_hess():
        X, tt = patients()
        Xh = np.column_stack([X, np.ones(len(tt))])
        th = np.array([0.4, -0.3, 0.2])
        _, g, H = s.nll_grad_hess(th, Xh, tt)
        eps = 1e-5
        f = lambda v: s.nll_grad_hess(v, Xh, tt)[0]
        num_g = np.array([(f(th + eps * e) - f(th - eps * e)) / (2 * eps) for e in np.eye(3)])
        assert np.allclose(g, num_g, rtol=1e-4, atol=1e-6), f"gradient {np.round(g, 4)} vs finite differences {np.round(num_g, 4)}"
        gr = lambda v: s.nll_grad_hess(v, Xh, tt)[1]
        num_H = np.column_stack([(gr(th + eps * e) - gr(th - eps * e)) / (2 * eps) for e in np.eye(3)])
        assert np.allclose(H, num_H, rtol=1e-4, atol=1e-6), "Hessian vs finite differences of the gradient"
        assert np.all(np.linalg.eigvalsh(H) > 0), "the Hessian should be positive definite (the NLL is convex)"

    def t_newton():
        # Five 1-D points from the notes: one step from θ = 0 lands on (w, b) = (0.8, 0.4).
        Xh = np.column_stack([[-2.0, -1.0, 0.0, 1.0, 2.0], np.ones(5)])
        tt = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
        th = s.newton_step(np.zeros(2), Xh, tt)
        assert np.allclose(th, [0.8, 0.4]), f"θ after one step = {th}, expected (0.8, 0.4)"

    def t_irls():
        X, tt = patients()
        w, b = s.logistic_irls(X, tt)
        ref = LogisticRegression(penalty=None, tol=1e-10, max_iter=1000).fit(X, tt)
        assert np.allclose(w, ref.coef_[0], atol=1e-3), f"w = {np.round(w, 4)}, sklearn = {np.round(ref.coef_[0], 4)}"
        assert abs(b - ref.intercept_[0]) < 1e-3, f"b = {b:.4f}, sklearn = {ref.intercept_[0]:.4f}"

    t.test("判断误分类", "Detecting a mistake", t_mistake)
    t.test("感知机更新", "The perceptron update", t_update)
    t.test("感知机：手算例题与线性可分数据", "Perceptron: the worked example and separable data", t_perceptron)
    t.test("数值稳定的 sigmoid", "A numerically stable sigmoid", t_sigmoid)
    t.test("梯度与 Hessian（与数值差分对比）", "Gradient and Hessian (vs finite differences)", t_grad_hess)
    t.test("一步牛顿法", "One Newton step", t_newton)
    t.test("IRLS 与 scikit-learn 的极大似然解一致", "IRLS matches scikit-learn's maximum-likelihood fit", t_irls)

    def irls_trace():
        X, tt = patients()
        tr = Trace("classifier2d", X, np.where(tt > 0, 1, -1), grid=Grid(X, n=40), show_margin=False,
                   title={"zh": "你的牛顿法（IRLS）", "en": "Your Newton's method (IRLS)"})
        s.logistic_irls(X, tt, trace=tr, grid=tr.grid)
        return tr

    def perceptron_trace():
        Xs, ys = separable()
        tr = Trace("classifier2d", Xs, ys, show_margin=False, title={"zh": "你的感知机", "en": "Your perceptron"})
        s.perceptron(Xs, ys, eta=0.5, trace=tr)
        return tr

    t.trace(irls_trace)
    t.trace(perceptron_trace)
