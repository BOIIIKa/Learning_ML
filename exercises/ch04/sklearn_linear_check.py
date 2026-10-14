"""Checks for sklearn_linear.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from learnml import Grid, Trace


def patients(seed=11, n=60):
    rng = np.random.default_rng(seed)
    t = np.arange(n) % 2
    centers = np.where(t[:, None] == 1, [0.8, 0.6], [-0.75, -0.55])
    return centers + 0.78 * rng.normal(size=(n, 2)), t


def check(s, t):
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    state = {}

    def t_fit():
        logistic, perceptron = s.fit_both(X_tr, y_tr)
        for m in (logistic, perceptron):
            assert isinstance(m, Pipeline), "fit_both should return two Pipelines"
        names = [type(m.steps[-1][1]).__name__ for m in (logistic, perceptron)]
        assert names == ["LogisticRegression", "Perceptron"], f"final steps are {names}"
        assert type(logistic.steps[0][1]).__name__ == "StandardScaler", "standardize first"
        acc = [m.score(X_te, y_te) for m in (logistic, perceptron)]
        assert min(acc) >= 0.9, f"test accuracies {np.round(acc, 3)}; both should reach 0.9"
        state["models"] = (logistic, perceptron)

    def t_proba():
        logistic, perceptron = state.get("models") or s.fit_both(X_tr, y_tr)
        p = s.risk_scores(logistic, X_te)
        assert p.shape == (len(y_te),) and np.all((p >= 0) & (p <= 1)), "one probability per test patient"
        z = logistic.decision_function(X_te)
        assert np.allclose(p, 1 / (1 + np.exp(-z))), "P(y = 1 | x) should equal σ(w·x + b)"
        assert log_loss(y_te, p) < 0.2, f"test cross-entropy {log_loss(y_te, p):.3f} should be < 0.2"
        assert not hasattr(perceptron, "predict_proba"), "the perceptron gives no probabilities"

    def t_softmax():
        Xi, yi = load_iris(return_X_y=True)
        model = s.fit_both(Xi, yi)[0]
        P = s.softmax(model.decision_function(Xi))
        assert np.allclose(P, model.predict_proba(Xi)), "softmax of the three scores = predict_proba"
        big = s.softmax(np.array([[1000.0, 1001.0, 999.0]]))
        assert np.all(np.isfinite(big)) and np.isclose(big.sum(), 1), "softmax must not overflow"

    t.test("拟合两条流水线并达到 90% 准确率", "Fit both pipelines to 90% test accuracy", t_fit)
    t.test("逻辑回归给出的风险概率", "Risk probabilities from logistic regression", t_proba)
    t.test("三分类的 softmax", "Softmax for three classes", t_softmax)

    def logistic_trace():
        Xp, yp = patients()
        grid = Grid(Xp, n=40)
        tr = Trace("classifier2d", Xp, np.where(yp == 1, 1, -1), grid=grid, show_margin=False,
                   title={"zh": "逻辑回归：C 从小到大", "en": "Logistic regression, C from small to large"})
        for C in (0.002, 0.01, 0.05, 0.3, 3.0):
            m = s.fit_both(Xp, yp, C=C)[0]
            p = s.risk_scores(m, grid.points)
            tr.record(field=p - 0.5, metrics={"logloss": log_loss(yp, s.risk_scores(m, Xp))}, note=f"C = {C}")
        return tr

    def perceptron_trace():
        Xp, yp = patients()
        grid = Grid(Xp, n=40)
        tr = Trace("classifier2d", Xp, np.where(yp == 1, 1, -1), grid=grid, show_margin=False,
                   title={"zh": "感知机：换随机种子", "en": "Perceptron, different seeds"})
        for seed in range(5):
            m = s.fit_both(Xp, yp, seed=seed)[1]
            tr.record(field=m.decision_function(grid.points), note=f"random_state = {seed}")
        return tr

    t.trace(logistic_trace)
    t.trace(perceptron_trace)
