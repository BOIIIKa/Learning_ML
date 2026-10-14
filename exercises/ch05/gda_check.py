"""Checks for gda.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from learnml import Grid, Trace


def two_gaussians(n, seed):
    """Routine (0) vs follow-up (1) patients: two 2-D Gaussians with different shapes, classes alternating."""
    rng = np.random.default_rng(seed)
    L0 = np.linalg.cholesky(np.array([[1.6, 0.7], [0.7, 0.6]]))
    L1 = np.linalg.cholesky(np.array([[0.12, 0.0], [0.0, 1.2]]))
    y = np.arange(n) % 2
    Z = rng.standard_normal((n, 2))
    X = np.where(y[:, None] == 0, Z @ L0.T + [-0.8, -0.4], Z @ L1.T + [0.9, 0.4])
    return X, y


def reference(X, y, shared):
    pri = np.array([np.mean(y == k) for k in (0, 1)])
    mu = np.array([X[y == k].mean(0) for k in (0, 1)])
    C = np.array([(X[y == k] - mu[k]).T @ (X[y == k] - mu[k]) / np.sum(y == k) for k in (0, 1)])
    if shared:
        C = np.array([pri[0] * C[0] + pri[1] * C[1]] * 2)
    return pri, mu, C


def ref_logpdf(X, m, S):
    D = X - m
    return -0.5 * (np.einsum("ni,ij,nj->n", D, np.linalg.inv(S), D) + np.log(np.linalg.det(S)) + 2 * np.log(2 * np.pi))


def check(s, t):
    X, y = two_gaussians(300, 0)
    Xt, yt = two_gaussians(4000, 1)

    def t_params():
        classes, pri, mu, C = s.fit_gaussians(X, y)
        rp, rm, rC = reference(X, y, False)
        assert list(classes) == [0, 1], f"classes = {classes}"
        assert np.allclose(pri, rp), f"priors {np.round(pri, 3)} vs {np.round(rp, 3)}"
        assert np.allclose(mu, rm), "class means"
        assert np.allclose(C, rC), "covariances should use 1/n_k (bias=True), one per class"

    def t_pooled():
        _, pri, mu, C = s.fit_gaussians(X, y, shared=True)
        _, _, rC = reference(X, y, True)
        assert np.allclose(C, rC), "both classes should get the prior-weighted pooled covariance"

    def t_logpdf():
        m, S = np.array([0.3, -0.2]), np.array([[1.5, 0.4], [0.4, 0.7]])
        got = s.log_gaussian(Xt[:50], m, S)
        assert np.allclose(got, ref_logpdf(Xt[:50], m, S)), "log density differs from the formula"

    def t_lda():
        model = s.fit_gaussians(X, y, shared=True)
        d = s.discriminants(Xt[:200], *model[1:])
        sk = LinearDiscriminantAnalysis().fit(X, y).decision_function(Xt[:200])
        assert np.allclose(d[:, 1] - d[:, 0], sk, atol=1e-6), "LDA log-odds should match scikit-learn's decision_function"

    def t_qda():
        qda = s.fit_gaussians(X, y)
        rp, rm, rC = reference(X, y, False)
        ref = np.argmax(np.column_stack([np.log(rp[k]) + ref_logpdf(Xt, rm[k], rC[k]) for k in (0, 1)]), axis=1)
        pred = s.predict(Xt, qda)
        assert np.array_equal(pred, ref), f"{np.sum(pred != ref)} predictions differ from the reference QDA"
        acc_q = np.mean(pred == yt)
        acc_l = np.mean(s.predict(Xt, s.fit_gaussians(X, y, shared=True)) == yt)
        assert acc_q > acc_l + 0.02, f"QDA accuracy {acc_q:.3f} should beat LDA {acc_l:.3f} when the shapes differ"

    def t_linear():
        _, pri, mu, C = s.fit_gaussians(X, y, shared=True)
        w, b = s.lda_linear(pri, mu, C[0])
        d = s.discriminants(Xt[:100], pri, mu, C)
        assert np.allclose(Xt[:100] @ w + b, d[:, 1] - d[:, 0], atol=1e-8), "w·x + b should equal δ₁(x) − δ₀(x)"

    t.test("极大似然参数（先验、均值、协方差）", "Maximum-likelihood parameters (priors, means, covariances)", t_params)
    t.test("合并协方差（LDA）", "Pooled covariance (LDA)", t_pooled)
    t.test("高斯对数密度", "Gaussian log-density", t_logpdf)
    t.test("LDA 的对数几率与 scikit-learn 一致", "LDA log-odds match scikit-learn", t_lda)
    t.test("QDA 的预测，以及形状不同时 QDA 胜过 LDA", "QDA predictions, and QDA beats LDA when the shapes differ", t_qda)
    t.test("共享协方差 ⇒ 线性判别函数", "Shared covariance ⇒ a linear discriminant", t_linear)

    def traces():
        grid = Grid(X[:160], n=40)
        title_l = {"zh": "你的 LDA：样本越来越多", "en": "Your LDA as the sample grows"}
        title_q = {"zh": "你的 QDA：样本越来越多", "en": "Your QDA as the sample grows"}
        tl = Trace("classifier2d", X[:160], np.where(y[:160] == 1, 1, -1), grid=grid, title=title_l, show_margin=False)
        tq = Trace("classifier2d", X[:160], np.where(y[:160] == 1, 1, -1), grid=grid, title=title_q, show_margin=False)
        for m in (6, 10, 16, 24, 40, 64, 100, 160):
            lda = s.fit_gaussians(X[:m], y[:m], shared=True)
            qda = s.fit_gaussians(X[:m], y[:m])
            fl = s.discriminants(grid.points, *lda[1:])
            fq = s.discriminants(grid.points, *qda[1:])
            acc = {"lda": float(np.mean(s.predict(Xt, lda) == yt)), "qda": float(np.mean(s.predict(Xt, qda) == yt))}
            w, b = s.lda_linear(lda[1], lda[2], lda[3][0])
            tl.record(w=w, b=b, field=np.clip(fl[:, 1] - fl[:, 0], -20, 20), metrics=acc, note=f"m = {m}")
            tq.record(field=np.clip(fq[:, 1] - fq[:, 0], -20, 20), metrics=acc, note=f"m = {m}")
        return tl, tq

    state = {}

    def first():
        state["pair"] = traces()
        return state["pair"][0]

    t.trace(first)
    if "pair" in state:
        t.trace(lambda: state["pair"][1])
