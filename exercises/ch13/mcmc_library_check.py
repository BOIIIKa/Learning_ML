"""Checks for mcmc_library.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from learnml import Trace

MED = ["patient", "doctor", "hospital", "treatment", "nurse", "surgery"]
FIN = ["stock", "market", "invest", "price", "bank", "bond"]


def texts(seed=0, D=30, L=25):
    rng = np.random.default_rng(seed)
    out = []
    for d in range(D):
        m = (0.9, 0.1, 0.6)[d % 3]
        out.append(" ".join(rng.choice(MED) if rng.uniform() < m else rng.choice(FIN) for _ in range(L)))
    return out


def sampler(logp, x0, step, n, seed=0):
    """A ready-made random-walk Metropolis sampler."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float)
    lp, out = logp(x), []
    for _ in range(n):
        y = x + step * rng.normal(size=x.size)
        ly = logp(y)
        if np.log(rng.uniform()) < ly - lp:
            x, lp = y, ly
        out.append(x)
    return np.array(out)


def data(seed=0, n=40):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = np.where(X @ [1.5, -1.0] + 0.3 + 0.8 * rng.normal(size=n) > 0, 1, -1)
    return X, y


def check(s, t):
    state = {}

    def t_lda():
        pipe = s.lda_pipeline(2).fit(texts())
        tops = s.top_words(pipe, 4)
        assert len(tops) == 2 and all(len(w) == 4 for w in tops), "two topics, four words each"
        kinds = sorted({"med" if all(w in MED for w in ws) else "fin" if all(w in FIN for w in ws) else "mixed" for ws in tops})
        assert kinds == ["fin", "med"], f"top words {tops}: one topic should be all medical, the other all financial"

    def t_pagerank():
        assert np.allclose(s.pagerank_eig([[1, 2], [2], [0]]), np.array([686, 380, 703]) / 1769, atol=1e-9), "the worked example: (686, 380, 703)/1769"
        r = s.pagerank_eig([[1, 2], [], [0, 1], [2]], d=0.8)
        assert np.all(r > 0) and abs(r.sum() - 1) < 1e-12, "a probability vector, positive everywhere"

    def t_posterior():
        X, y = data()
        th = np.array([0.4, -0.2, 0.1])
        m = y * (X @ th[:2] + th[2])
        ref = -np.log1p(np.exp(-m)).sum() - th @ th / 18
        assert abs(s.log_posterior(th, X, y) - ref) < 1e-9, "log posterior = Σ log σ(margin) − ‖θ‖²/(2τ²)"
        draws = sampler(lambda th: s.log_posterior(th, X, y), np.zeros(3), 0.5, 6000, seed=1)[1000:]
        mean = draws.mean(axis=0)
        mode = LogisticRegression(C=9.0).fit(X, y)  # penalty ‖w‖²/(2C) ⇔ prior N(0, C); scikit-learn does not penalize b
        w = np.r_[mode.coef_[0], mode.intercept_]
        cos = mean[:2] @ w[:2] / np.linalg.norm(mean[:2]) / np.linalg.norm(w[:2])
        assert cos > 0.98, f"the posterior mean should point the same way as the MAP fit (cosine {cos:.3f})"
        state["draws"], state["X"], state["y"] = draws, X, y

    t.test("LDA 找到两个话题", "LDA finds the two topics", t_lda)
    t.test("PageRank 作为特征向量", "PageRank as an eigenvector", t_pagerank)
    t.test("贝叶斯逻辑回归的后验", "The Bayesian logistic-regression posterior", t_posterior)

    def trace():
        draws, X, y = state.get("draws"), state.get("X"), state.get("y")
        if draws is None:
            X, y = data()
            draws = sampler(lambda th: s.log_posterior(th, X, y), np.zeros(3), 0.5, 6000, seed=1)[1000:]
        tr = Trace("classifier2d", X, y, show_margin=False,
                   title={"zh": "后验中抽出的分类边界", "en": "Decision boundaries drawn from the posterior"})
        for th in draws[::500]:
            acc = float(np.mean(np.sign(X @ th[:2] + th[2]) == y))
            tr.record(w=th[:2], b=float(th[2]), metrics={"acc": acc}, note=f"w = ({th[0]:.2f}, {th[1]:.2f}), b = {th[2]:.2f}")
        return tr

    t.trace(trace)
