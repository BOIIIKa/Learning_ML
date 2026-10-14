"""Checks for nll_library.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from learnml import Grid, Trace


def hetero_data(n=300, seed=4):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    return x, 40 + 4 * x + (1.5 + 0.9 * x) * rng.normal(size=n)


def check(s, t):
    def t_nll():
        X, y = load_digits(return_X_y=True)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
        weak = make_pipeline(StandardScaler(), LogisticRegression(C=1e-4, max_iter=2000)).fit(Xtr, ytr)
        good = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(Xtr, ytr)
        a, b = s.nll_of(weak, Xte, yte), s.nll_of(good, Xte, yte)
        assert b < a and b < 0.3, f"a well-regularized model should have a much lower test NLL ({b:.3f} vs {a:.3f})"
        assert abs(a - np.log(10)) < 0.25, f"an almost-untrained 10-class model sits near log 10 = 2.30 (got {a:.3f})"

    def t_counts():
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 3, (400, 2))
        y = rng.poisson(np.exp(-1 + 0.9 * X[:, 0] + 0.4 * X[:, 1]))
        pois, lin = s.fit_counts(X, y)
        dp = mean_poisson_deviance(y, pois.predict(X))
        dl = mean_poisson_deviance(y, np.clip(lin.predict(X), 1e-3, None))
        assert dp < dl, f"the Poisson model should have the lower Poisson deviance ({dp:.3f} vs {dl:.3f})"
        assert np.all(pois.predict(X) > 0), "a log link never predicts a negative count"
        assert np.allclose(pois.coef_, [0.9, 0.4], atol=0.15), f"coefficients {np.round(pois.coef_, 2)} should be near (0.9, 0.4)"

    def t_interval():
        x, y = hetero_data()
        X = x[:, None]
        lo, hi = s.interval(X, y)
        inside = (y >= lo.predict(X)) & (y <= hi.predict(X))
        assert 0.85 <= inside.mean() <= 0.95, f"a 90% interval should cover about 90% of the points (got {inside.mean():.2f})"
        w1 = hi.predict([[1.0]])[0] - lo.predict([[1.0]])[0]
        w9 = hi.predict([[9.0]])[0] - lo.predict([[9.0]])[0]
        assert w9 > 2.5 * w1, f"the interval should widen with the noise: width {w1:.1f} at x = 1, {w9:.1f} at x = 9"

    t.test("分类器的测试负对数似然", "Test NLL of a classifier", t_nll)
    t.test("计数数据：泊松回归", "Count data: Poisson regression", t_counts)
    t.test("分位数回归给出预测区间", "Quantile regression gives prediction intervals", t_interval)

    def trace():
        x, y = hetero_data()
        X = np.column_stack([x, y])
        grid = Grid(X, n=48)
        tr = Trace("classifier2d", X, np.where(y > 40 + 4 * x, 1, -1), grid=grid, show_margin=False,
                   title={"zh": "越来越宽的预测区间（分位数回归）", "en": "Wider and wider prediction intervals (quantile regression)"})
        for level in (0.2, 0.5, 0.8, 0.9, 0.98):
            lo, hi = s.interval(x[:, None], y, level)
            gx = grid.points[:, :1]
            l, h = lo.predict(gx), hi.predict(gx)
            mid, half = (l + h) / 2, np.maximum((h - l) / 2, 1e-6)
            cover = float(np.mean((y >= lo.predict(x[:, None])) & (y <= hi.predict(x[:, None]))))
            tr.record(field=1 - np.abs(grid.points[:, 1] - mid) / half, metrics={"coverage": cover}, note=f"{int(level * 100)}%")
        return tr

    t.trace(trace)
