"""Checks for adaboost.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.datasets import make_moons

from learnml import Grid, Trace


def moons():
    X, y = make_moons(n_samples=120, noise=0.25, random_state=0)
    return X, np.where(y == 1, 1.0, -1.0)


def check(s, t):
    def t_error():
        w = np.array([0.1, 0.2, 0.3, 0.4])
        y = np.array([1, -1, 1, -1])
        assert np.isclose(s.weighted_error(w, y, np.array([1, 1, 1, 1])), 0.6), "errors at indices 1 and 3"

    def t_alpha():
        assert np.isclose(s.stump_weight(0.3), 0.5 * np.log(7 / 3)), "e = 0.3"
        assert np.isclose(s.stump_weight(0.5), 0.0), "a coin flip gets no vote"

    def t_weights():
        w = np.full(4, 0.25)
        new = s.update_weights(w, np.log(3) / 2, np.array([1, 1, -1, -1]), np.array([1, 1, -1, 1]))
        assert np.isclose(new.sum(), 1.0), "weights should sum to 1"
        assert np.isclose(new[3], 0.5), f"the single mistake should carry half the weight, got {new[3]:.3f}"

    def t_predict():
        model = [(1.0, 0, 0.5, 1.0), (0.4, 0, 1.5, -1.0)]
        X = np.array([[0.0], [1.0], [2.0]])
        assert np.array_equal(s.predict(model, X), [1, -1, -1]), "weighted vote of two stumps"

    def t_classic():
        # The classic ten-point example: three rounds reach zero training error.
        X = np.arange(10, dtype=float).reshape(-1, 1)
        y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1], dtype=float)
        model = s.adaboost(X, y, rounds=3)
        alphas = [round(float(m[0]), 4) for m in model]
        assert np.allclose(alphas, [0.4236, 0.6496, 0.7520], atol=1e-3), f"α = {alphas}"
        assert np.array_equal(s.predict(model, X), y), "training error should be 0 after 3 rounds"

    t.test("加权误差", "Weighted error", t_error)
    t.test("弱分类器系数 α", "Stump weight α", t_alpha)
    t.test("样本权重更新", "Sample weight update", t_weights)
    t.test("加权投票预测", "Weighted-vote prediction", t_predict)
    t.test("经典十点例子", "The classic ten-point example", t_classic)

    def student_trace():
        X, y = moons()
        trace = Trace("classifier2d", X, y, grid=Grid(X, n=40), title={"zh": "你的 AdaBoost", "en": "Your AdaBoost"})
        s.adaboost(X, y, rounds=30, trace=trace)
        return trace

    t.trace(student_trace)
