"""Checks for logreg_library.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from learnml import Grid, Trace


def blobs(seed=3, n=80):
    """Two overlapping standardized blobs with labels in {−1, +1}."""
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal([0.9, 0.7], 0.8, (n // 2, 2)), rng.normal([-0.9, -0.5], 0.8, (n // 2, 2))])
    return X, np.array([1.0] * (n // 2) + [-1.0] * (n // 2))


def check(s, t):
    data = load_wine()
    X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.3, random_state=0, stratify=data.target)
    state = {}

    def t_build():
        m = s.build_cv()
        assert isinstance(m, Pipeline) and isinstance(m[-1], LogisticRegressionCV), "a pipeline ending in LogisticRegressionCV"
        assert len(np.atleast_1d(m[-1].Cs)) == 13, "search 13 values of C"

    def t_fit():
        m = s.build_cv().fit(X_tr, y_tr)
        acc = m.score(X_te, y_te)
        assert acc > 0.93, f"test accuracy {acc:.3f} should exceed 0.93"
        assert m[-1].coef_.shape == (3, 13), f"coef_ has shape {m[-1].coef_.shape}; a softmax model has one weight vector per class: (3, 13)"
        state["model"] = m

    def t_conf():
        m = state.get("model")
        assert m is not None, "fix the previous test first"
        c = np.asarray(s.confidence(m, X_te))
        assert c.shape == (len(X_te),), f"one number per row, got shape {c.shape}"
        assert np.all(c >= 1 / 3 - 1e-9) and np.all(c <= 1), "the top class probability lies in [1/3, 1]"
        assert np.allclose(c, m.predict_proba(X_te).max(axis=1)), "maximum of each predict_proba row"

    def t_path():
        X, y = blobs()
        Cs = np.logspace(-3, 2, 6)
        norms = [np.linalg.norm(w) for w in s.c_path(X, y, Cs)]
        assert len(norms) == 6, "one coefficient vector per C"
        assert all(a < b for a, b in zip(norms, norms[1:])), f"‖w‖ should grow with C: {np.round(norms, 3)}"

    def t_sgd():
        models = s.sgd_models()
        assert sorted(models) == sorted(s.LOSSES) and all(isinstance(m, SGDClassifier) for m in models.values()), "one SGDClassifier per loss"
        X, y = blobs(n=200)
        for loss, m in models.items():
            assert m.loss == loss, f"model for {loss} has loss={m.loss}"
            assert m.average, f"{loss}: set average=True"
            acc = m.fit(X, y).score(X, y)
            assert acc > 0.85, f"{loss}: training accuracy {acc:.2f} should exceed 0.85"
        assert hasattr(models["log_loss"], "predict_proba") and not hasattr(models["hinge"], "predict_proba"), (
            "only probabilistic losses offer predict_proba"
        )

    t.test("组装交叉验证管道", "Assembling the CV pipeline", t_build)
    t.test("softmax 模型的测试准确率", "Test accuracy of the softmax model", t_fit)
    t.test("预测置信度", "Prediction confidence", t_conf)
    t.test("正则化路径：C 越大 ‖w‖ 越大", "Regularization path: ‖w‖ grows with C", t_path)
    t.test("四种损失的 SGD", "SGD with four losses", t_sgd)

    def path_trace():
        X, y = blobs()
        grid = Grid(X, n=40)
        Cs = np.logspace(-3, 2, 11)
        tr = Trace("classifier2d", X, y, grid=grid, show_margin=False,
                   title={"zh": "C 从小到大：边界与概率场", "en": "C from small to large: boundary and probability field"})
        from sklearn.linear_model import LogisticRegression

        for C, w in zip(Cs, s.c_path(X, y, Cs)):
            b = LogisticRegression(C=C, max_iter=2000).fit(X, y).intercept_[0]
            p = 1 / (1 + np.exp(-(grid.points @ w + b)))
            tr.record(w=w, b=b, field=p - 0.5, metrics={"norm": float(np.linalg.norm(w))}, note=f"C = {C:.3g}")
        return tr

    def sgd_trace():
        X, y = blobs()
        X = np.vstack([X, [[-2.6, -2.2], [-2.3, -2.5]]])  # two mislabeled outliers deep in the blue side
        y = np.concatenate([y, [1.0, 1.0]])
        grid = Grid(X, n=40)
        tr = Trace("classifier2d", X, y, grid=grid, show_margin=False,
                   title={"zh": "同一个线性模型，四种损失", "en": "One linear model, four losses"})
        for loss, m in s.sgd_models().items():
            m.fit(X, y)
            tr.record(w=m.coef_[0], b=m.intercept_[0], field=m.decision_function(grid.points),
                      metrics={"acc": float(m.score(X, y))}, note=loss)
        return tr

    t.trace(path_trace)
    t.trace(sgd_trace)
