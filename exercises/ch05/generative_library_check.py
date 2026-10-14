"""Checks for generative_library.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from learnml import Grid, Trace


def two_gaussians(n, seed):
    """Two 2-D Gaussians with different shapes, classes alternating."""
    rng = np.random.default_rng(seed)
    L0 = np.linalg.cholesky(np.array([[1.6, 0.7], [0.7, 0.6]]))
    L1 = np.linalg.cholesky(np.array([[0.12, 0.0], [0.0, 1.2]]))
    y = np.arange(n) % 2
    Z = rng.standard_normal((n, 2))
    X = np.where(y[:, None] == 0, Z @ L0.T + [-0.8, -0.4], Z @ L1.T + [0.9, 0.4])
    return X, y


def redundant(n, rng, rho=0.6):
    """14 features: 10 independent ones plus 4 copies of feature 0, correlated by rho (the chapter's widget)."""
    d0, c = 10, 4
    mu = np.full(d0 + c, 0.45)
    mu[0] = mu[d0:] = 0.8
    S = np.eye(d0 + c)
    idx = [0] + list(range(d0, d0 + c))
    S[np.ix_(idx, idx)] = rho
    S[idx, idx] = 1.0
    y = np.arange(n) % 2
    X = rng.standard_normal((n, d0 + c)) @ np.linalg.cholesky(S).T + np.outer(y, mu)
    return X, y


URGENT = ["fever pain fever pain", "fever pain fever schedule"]
ROUTINE = ["refill schedule schedule refill", "schedule refill schedule fever", "pain schedule schedule refill"]


def check(s, t):
    X, y = two_gaussians(300, 0)
    Xt, yt = two_gaussians(4000, 1)
    state = {}

    def t_models():
        models = s.build_models()
        kinds = {k: type(v) for k, v in models.items()}
        want = {"LDA": LinearDiscriminantAnalysis, "QDA": QuadraticDiscriminantAnalysis, "NB": GaussianNB, "LR": LogisticRegression}
        assert kinds == want, f"got {({k: v.__name__ for k, v in kinds.items()})}"
        assert models["LR"].C == 100, "use C=100: almost no regularization, as in Ng & Jordan"
        state["models"] = models

    def t_shapes():
        assert "models" in state, "fix the previous test first"
        acc = {k: m.fit(X, y).score(Xt, yt) for k, m in state["models"].items()}
        assert acc["QDA"] > acc["LDA"] + 0.02, f"accuracies {({k: round(v, 3) for k, v in acc.items()})}: QDA should win here"

    def t_curve():
        assert "models" in state, "fix the previous test first"
        rng = np.random.default_rng(0)
        Xte, yte = redundant(4000, rng)
        sizes = [24, 48, 512]
        nb, lr = np.zeros(3), np.zeros(3)
        for _ in range(6):
            Xtr, ytr = redundant(512, rng)
            nb += np.array(s.learning_curve(state["models"]["NB"], Xtr, ytr, Xte, yte, sizes)) / 6
            lr += np.array(s.learning_curve(state["models"]["LR"], Xtr, ytr, Xte, yte, sizes)) / 6
        assert len(nb) == 3, "one error per training size"
        assert nb[0] < lr[0], f"m = 24: naive Bayes {nb[0]:.3f} should beat logistic regression {lr[0]:.3f}"
        assert lr[2] < nb[2], f"m = 512: logistic regression {lr[2]:.3f} should beat naive Bayes {nb[2]:.3f}"

    def t_text():
        docs, labels = URGENT + ROUTINE, ["urgent"] * 2 + ["routine"] * 3
        clf = s.message_classifier(alpha=1.0).fit(docs, labels)
        k = list(clf.classes_).index("urgent")
        p = clf.predict_proba(["fever pain refill"])[0, k]
        assert abs(p - 1024 / 1672) < 1e-6, f"P(urgent | fever pain refill) = {p:.4f}; with α = 1 it should be 0.6124"
        tiny = s.message_classifier(alpha=0.001).fit(docs, labels).predict_proba(["fever pain refill"])[0, k]
        assert tiny < 0.02, f"with α = 0.001 an unseen word should nearly veto 'urgent' (got {tiny:.3f})"

    t.test("构建四个分类器", "Building the four classifiers", t_models)
    t.test("两类形状不同时 QDA 胜出", "QDA wins when the class shapes differ", t_shapes)
    t.test("学习曲线交叉：朴素贝叶斯先快、逻辑回归后好", "Crossing learning curves: naive Bayes early, logistic regression late", t_curve)
    t.test("多项式朴素贝叶斯与拉普拉斯平滑", "Multinomial naive Bayes with Laplace smoothing", t_text)

    def trace():
        grid = Grid(X[:200], n=40)
        tr = Trace("classifier2d", X[:200], np.where(y[:200] == 1, 1, -1), grid=grid, show_margin=False,
                   title={"zh": "四个分类器的边界", "en": "Four boundaries"})
        for name, m in s.build_models().items():
            m.fit(X, y)
            lp = m.predict_log_proba(grid.points)
            tr.record(field=np.clip(lp[:, 1] - lp[:, 0], -20, 20), metrics={"acc": m.score(Xt, yt)}, note=name)
        return tr

    t.trace(trace)
