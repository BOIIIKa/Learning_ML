"""Checks for optimizers.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from learnml import Grid, Trace


def blobs(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal([-1, -0.5], 1.0, (n // 2, 2)), rng.normal([1.2, 0.8], 1.0, (n // 2, 2))])
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    return X, y


def mean_bce(X, y, theta):
    f = np.column_stack([X, np.ones(len(X))]) @ theta
    return float(np.mean(np.logaddexp(0, f) - y * f))


def check(s, t):
    def t_momentum():
        k = 10.0
        grad = lambda x: np.array([x[0] / k, x[1]])
        r = np.sqrt(k)
        eta, beta = 4 / (1 + 1 / r) ** 2, ((r - 1) / (r + 1)) ** 2
        path = s.momentum_path(grad, [-3.0, 1.0], eta, beta, 60)
        assert path.shape == (61, 2), f"the path should hold steps + 1 = 61 points (got {path.shape})"
        norms = np.linalg.norm(path, axis=1)
        steps = int(np.argmax(norms <= 1e-3 * norms[0]))
        assert steps == 15, f"with the best (η, β) at κ = 10 the error should shrink 1000× in 15 steps (got {steps})"
        const = s.momentum_path(lambda x: np.array([1.0]), [0.0], 0.1, 0.9, 300)
        assert abs((const[-1] - const[-2])[0] + 1.0) < 1e-9, "with a constant gradient the step should tend to −η/(1 − β) = −1"

    def t_adam_first():
        for g in [0.01, 100.0]:
            path = s.adam_path(lambda x: np.array([g]), [0.0], 0.1, 1, eps=0.0)
            assert abs(path[1, 0] + 0.1) < 1e-12, f"the first Adam step should be exactly −lr = −0.1 for g = {g} (got {path[1, 0]:.4f})"
        path = s.adam_path(lambda x: np.array([3.0, -0.2]), [0.0, 0.0], 0.05, 20, eps=0.0)
        assert np.allclose(np.diff(path, axis=0), [-0.05, 0.05]), "a constant gradient should give steps of exactly lr against its sign"

    def t_adam_reference():
        grad = lambda x: np.array([2 * x[0] - 1, 6 * x[1] ** 2 + 0.5])
        path = s.adam_path(grad, [1.0, -0.5], 0.03, 5)
        x, m, v = np.array([1.0, -0.5]), np.zeros(2), np.zeros(2)
        for k in range(1, 6):
            g = grad(x)
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g ** 2
            x = x - 0.03 * (m / (1 - 0.9 ** k)) / (np.sqrt(v / (1 - 0.999 ** k)) + 1e-8)
        assert np.allclose(path[-1], x, atol=1e-12), f"after 5 steps x = {path[-1]}, expected {x}"

    def t_adamw():
        path = s.adam_path(lambda x: np.zeros(1), [1.0], 0.01, 10, wd=0.1)
        assert abs(path[-1, 0] - (1 - 0.001) ** 10) < 1e-12, "with zero gradient AdamW should shrink x by (1 − lr·wd) per step"

    def t_logreg():
        X, y = blobs()
        theta = s.adam_path(s.logistic_grad(X, y), [0.0, 0.0, 0.0], 0.05, 600)[-1]
        ref = LogisticRegression(penalty=None).fit(X, y)
        best = mean_bce(X, y, np.r_[ref.coef_[0], ref.intercept_])
        got = mean_bce(X, y, theta)
        assert got < best + 1e-3, f"Adam's loss {got:.4f} should come within 1e-3 of the optimum {best:.4f}"

    t.test("重球动量", "Heavy-ball momentum", t_momentum)
    t.test("Adam 的第一步", "Adam's first step", t_adam_first)
    t.test("Adam 与逐步手算一致", "Adam matches a step-by-step reference", t_adam_reference)
    t.test("AdamW 的解耦衰减", "AdamW's decoupled decay", t_adamw)
    t.test("用 Adam 训练逻辑回归", "Training logistic regression with Adam", t_logreg)

    def trace():
        X, y = blobs()
        tr = Trace("classifier2d", X, np.where(y > 0, 1, -1), grid=Grid(X), show_margin=False,
                   title={"zh": "Adam 训练逻辑回归：分界线一步步转到位", "en": "Adam trains a logistic regression: the boundary turns into place"})
        keep = {1, 3, 8, 20, 50, 120, 300, 600}

        def rec(step, theta):
            if step in keep:
                tr.record(w=[float(theta[0]), float(theta[1])], b=float(theta[2]), metrics={"loss": mean_bce(X, y, theta)}, note=f"step {step}")

        s.adam_path(s.logistic_grad(X, y), [0.0, 0.0, 0.0], 0.05, 600, trace=rec)
        return tr

    t.trace(trace)
