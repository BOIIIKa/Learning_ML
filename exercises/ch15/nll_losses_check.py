"""Checks for nll_losses.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.metrics import log_loss

from learnml import Grid, Trace


def hetero_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    y = 40 + 4 * x + (1.5 + 0.9 * x) * rng.normal(size=n)
    return x, y


def check(s, t):
    def t_bce():
        rng = np.random.default_rng(1)
        f = rng.normal(0, 3, 200)
        y = (rng.uniform(size=200) < 1 / (1 + np.exp(-f))).astype(float)
        ref = log_loss(y, 1 / (1 + np.exp(-f)))
        assert abs(s.bce_with_logits(y, f) - ref) < 1e-9, "should equal scikit-learn's log_loss on σ(f)"
        assert abs(s.bce_with_logits([1, 0], [2, 2]) - (0.126928 + 2.126928) / 2) < 1e-5, "the worked example: 0.1269 and 2.1269 at f = 2"
        big = s.bce_with_logits([0, 1], [800, -800])
        assert np.isfinite(big) and abs(big - 800) < 1e-9, f"f = ±800 must not overflow (got {big})"

    def t_softmax():
        rng = np.random.default_rng(2)
        F = rng.normal(0, 2, (150, 4))
        y = rng.integers(0, 4, 150)
        P = np.exp(F) / np.exp(F).sum(axis=1, keepdims=True)
        assert abs(s.softmax_ce(y, F) - log_loss(y, P, labels=range(4))) < 1e-9, "should equal scikit-learn's log_loss"
        assert abs(s.softmax_ce([0], [[2.0, 1.0, 0.0]]) - 0.407606) < 1e-6, "the worked example: logits (2, 1, 0) → 0.4076"
        assert np.isfinite(s.softmax_ce([0], [[1000.0, 0.0, -1000.0]])), "huge logits must not overflow"

    def t_gauss():
        assert abs(s.gaussian_nll([78.0], 70.0, np.log(5.0)) - 3.808376) < 1e-6, "the worked example: 3.808 for σ = 5"
        assert abs(s.gaussian_nll([78.0], 70.0, np.log(8.0)) - 3.498380) < 1e-6, "σ = 8 = |residual| gives 3.498"

    def t_hetero():
        x, y = hetero_data()
        w, v = s.fit_hetero(x, y)
        u = x / 10
        het = s.gaussian_nll(y, w[0] + w[1] * u, v[0] + v[1] * u)
        b = np.polyfit(x, y, 1)
        homo = s.gaussian_nll(y, np.polyval(b, x), np.log(np.std(y - np.polyval(b, x))))
        assert het < homo - 0.05, f"heteroscedastic NLL {het:.3f} should clearly beat the constant-σ {homo:.3f}"
        assert v[1] > 0.8, f"log σ should grow with x (slope {v[1]:.2f} per unit of x/10)"
        sd1, sd9 = np.exp(v[0] + 0.1 * v[1]), np.exp(v[0] + 0.9 * v[1])
        assert sd9 > 2.5 * sd1, f"σ(9) = {sd9:.1f} should be well above σ(1) = {sd1:.1f}"

    t.test("稳定的二元交叉熵", "Stable binary cross-entropy", t_bce)
    t.test("稳定的 softmax 交叉熵", "Stable softmax cross-entropy", t_softmax)
    t.test("高斯负对数似然", "Gaussian negative log-likelihood", t_gauss)
    t.test("异方差回归", "Heteroscedastic regression", t_hetero)

    def trace():
        x, y = hetero_data()
        X = np.column_stack([x, y])
        inside = np.abs(y - (40 + 4 * x)) <= 2 * (1.5 + 0.9 * x)
        grid = Grid(X, n=48)
        gu = grid.points[:, 0] / 10
        tr = Trace("classifier2d", X, np.where(inside, 1, -1), grid=grid, show_margin=False,
                   title={"zh": "±2σ 带在梯度下降中学会随 x 变宽", "en": "The ±2σ band learns to widen with x during gradient descent"})
        keep = {0, 30, 150, 600, 3999}
        def rec(step, w, v):
            if step in keep:
                mu, sd = w[0] + w[1] * gu, np.exp(v[0] + v[1] * gu)
                nll = s.gaussian_nll(y, w[0] + w[1] * x / 10, v[0] + v[1] * x / 10)
                tr.record(field=1 - np.abs(grid.points[:, 1] - mu) / (2 * sd), metrics={"nll": nll}, note=f"step {step + 1}")
        s.fit_hetero(x, y, trace=rec)
        return tr

    t.trace(trace)
