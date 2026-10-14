"""Checks for gda_nb.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB

from learnml import Grid, Trace


def gaussians(n=200, seed=0):
    """Class 1 is an elongated tilted Gaussian, class 0 a round one."""
    rng = np.random.default_rng(seed)
    A = np.array([[1.0, 0.0], [0.55, 0.45]])
    X1 = rng.normal(size=(n // 2, 2)) @ A.T + [0.9, 0.6]
    X0 = rng.normal(size=(n // 2, 2)) * 0.55 + [-0.9, -0.5]
    return np.vstack([X0, X1]), np.array([0] * (n // 2) + [1] * (n // 2))


# The worked example of the notes: features (fever, cough).
EX_X = np.array([[1, 1], [1, 1], [1, 1], [0, 1], [1, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0]])
EX_Y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])


def check(s, t):
    X, y = gaussians()

    def t_stats():
        pr, mu, cov = s.fit_gaussians(X, y)
        assert np.allclose(pr, [0.5, 0.5]), f"priors {pr}"
        assert np.allclose(mu[1], X[y == 1].mean(axis=0)), "class-1 mean"
        assert np.allclose(cov[0], np.cov(X[y == 0].T, bias=True)), "class-0 covariance should divide by n_0 (MLE)"

    def t_lda():
        pr, mu, cov = s.fit_gaussians(X, y)
        w, b = s.lda_line(pr, mu, s.pooled(pr, cov))
        ref = LinearDiscriminantAnalysis(solver="lsqr").fit(X, y)
        mine = 1 / (1 + np.exp(-(X @ w + b)))
        assert np.allclose(mine, ref.predict_proba(X)[:, 1], atol=1e-6), "P(1|x) should match scikit-learn's LDA"

    def t_qda():
        pr, mu, cov = s.fit_gaussians(X, y)
        lo = s.qda_log_odds(X, pr, mu, cov)
        ref = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X, y)
        # scikit-learn uses unbiased (n_k − 1) covariances; compare against the same Gaussians with MLE scaling.
        p_ref = 1 / (1 + np.exp(-lo))
        assert np.mean((p_ref > 0.5) == ref.predict(X)) > 0.97, "QDA decisions should agree with scikit-learn"
        assert np.all(np.isfinite(lo)), "log-odds must be finite"

    def t_quadratic():
        pr, mu, cov = s.fit_gaussians(X, y)
        pts = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        lo = s.qda_log_odds(pts, pr, mu, cov)
        assert abs(lo[0] - 2 * lo[1] + lo[2]) > 0.1, "with different covariances the log-odds should be quadratic"
        same = np.array([s.pooled(pr, cov)] * 2)
        lo2 = s.qda_log_odds(pts, pr, mu, same)
        assert abs(lo2[0] - 2 * lo2[1] + lo2[2]) < 1e-9, "with a shared covariance they must be linear"

    def t_nb():
        pr, th = s.bernoulli_nb(EX_X, EX_Y, alpha=1.0)
        ref = BernoulliNB(alpha=1.0).fit(EX_X, EX_Y)
        assert np.allclose(np.log(th), ref.feature_log_prob_), f"θ = {np.round(th, 3)}"
        p = s.nb_posterior(np.array([0, 0]), pr, th)
        assert abs(p - 32 / 437) < 1e-9, f"P(+ | no fever, no cough) = {p:.4f}; the notes get 32/437 ≈ 0.0732"
        _, th0 = s.bernoulli_nb(EX_X, EX_Y, alpha=0.0)
        assert s.nb_posterior(np.array([0, 0]), pr, th0) == 0.0, "without smoothing the posterior collapses to 0"

    t.test("类别统计量（MLE）", "Class statistics (MLE)", t_stats)
    t.test("LDA 与 scikit-learn 一致", "LDA matches scikit-learn", t_lda)
    t.test("QDA 的决策与 scikit-learn 一致", "QDA decisions match scikit-learn", t_qda)
    t.test("共享协方差 ⇒ 线性；否则二次", "Shared covariance ⇒ linear, otherwise quadratic", t_quadratic)
    t.test("例题：拉普拉斯平滑", "Worked example: Laplace smoothing", t_nb)

    def growing(name, use_qda):
        def make():
            grid = Grid(X, n=40)
            tr = Trace("classifier2d", X, 2 * y - 1, grid=grid, show_margin=False, title=name)
            order = np.random.default_rng(1).permutation(len(y))
            for n in (8, 12, 20, 32, 50, 80, 120, 200):
                idx = order[:n]
                if len(set(y[idx])) < 2:
                    continue
                pr, mu, cov = s.fit_gaussians(X[idx], y[idx])
                if use_qda:
                    cov = cov + 1e-3 * np.eye(2)
                    field = s.qda_log_odds(grid.points, pr, mu, cov)
                    tr.record(field=field, highlight=list(map(int, idx)), metrics={"acc": float(np.mean((s.qda_log_odds(X, pr, mu, cov) > 0) == y))}, note=f"n = {n}")
                else:
                    w, b = s.lda_line(pr, mu, s.pooled(pr, cov) + 1e-3 * np.eye(2))
                    tr.record(w=w, b=b, field=grid.points @ w + b, highlight=list(map(int, idx)), metrics={"acc": float(np.mean(((X @ w + b) > 0) == y))}, note=f"n = {n}")
            return tr
        return make

    t.trace(growing({"zh": "你的 LDA：样本越来越多", "en": "Your LDA as data accumulates"}, False))
    t.trace(growing({"zh": "你的 QDA：样本越来越多", "en": "Your QDA as data accumulates"}, True))
