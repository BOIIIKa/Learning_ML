"""Checks for model_selection_library.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from learnml import Grid, Trace


def truth(h):
    return 30 + 50 / (1 + np.exp(-(h - 4)))


def check(s, t):
    rng = np.random.default_rng(21)
    n = 40
    x = 0.5 + 9 * (np.arange(n) + rng.uniform(size=n)) / n
    y = truth(x) + 8 * rng.normal(size=n)
    X = x.reshape(-1, 1)
    degrees = list(range(1, 11))
    state = {}

    def t_pipeline():
        model = s.make_model(4)
        kinds = [type(step).__name__ for _, step in model.steps]
        assert kinds == ["StandardScaler", "PolynomialFeatures", "LinearRegression"], f"pipeline steps {kinds}"
        assert model.named_steps["polynomialfeatures"].degree == 4, "PolynomialFeatures(degree)"
        cubic = 2 - x + 0.3 * x ** 2 - 0.02 * x ** 3
        model.fit(X, cubic)
        assert np.allclose(model.predict(X), cubic, atol=1e-6), "a degree-4 model must fit a noise-free cubic exactly"

    def t_cv():
        got = s.cv_score(X, y, 3, k=5)
        errs = []
        for tr, va in KFold(n_splits=5, shuffle=True, random_state=0).split(X):
            m = s.make_model(3).fit(X[tr], y[tr])
            errs.append(np.mean((m.predict(X[va]) - y[va]) ** 2))
        assert got > 0, "return the MSE itself (cross_val_score gives its negative)"
        assert np.isclose(got, np.mean(errs)), f"cv_score = {got:.3f}, manual 5-fold loop = {np.mean(errs):.3f}"

    def t_curve():
        train, val = s.degree_curve(X, y, degrees)
        assert len(train) == len(val) == len(degrees), "one value per degree"
        assert np.all(train > 0) and np.all(val > 0), "return MSEs (positive), not negative scores"
        assert np.all(np.diff(train) <= 1e-6), f"training MSE {np.round(train, 1)} should never increase with the degree"
        best = int(np.argmin(val))
        assert 0 < best < len(degrees) - 1, f"validation MSE {np.round(val, 1)} should be U-shaped"
        state["curve"] = (train, val)

    def t_pick():
        assert s.pick_degree([1, 2, 3, 4], [9.0, 4.0, 5.0, 7.0]) == 2, "pick the degree with the smallest error"
        assert "curve" in state, "fix the previous test first"
        d = s.pick_degree(degrees, state["curve"][1])
        assert 2 <= d <= 6, f"picked degree {d}"

    t.test("多项式回归流水线", "The polynomial pipeline", t_pipeline)
    t.test("cross_val_score 与手写 k 折一致", "cross_val_score matches a manual k-fold loop", t_cv)
    t.test("validation_curve：训练误差单调降、验证误差 U 形", "validation_curve: training error falls, validation error is U-shaped", t_curve)
    t.test("按验证误差选阶数", "Picking the degree by validation error", t_pick)

    def trace():
        Xp = np.column_stack([x, y])
        grid = Grid(Xp, n=48, pad=0.6)
        gx, gy = grid.points[:, 0], grid.points[:, 1]
        train, val = s.degree_curve(X, y, degrees)
        tr = Trace("classifier2d", Xp, -np.ones(n), grid=grid,
                   title={"zh": "阶数从 1 到 10：拟合曲线与验证误差", "en": "Degree 1 to 10: the fit and the validation error"})
        for d, e_tr, e_va in zip(degrees, train, val):
            m = s.make_model(d).fit(X, y)
            tr.record(field=m.predict(gx.reshape(-1, 1)) - gy, metrics={"train": float(e_tr), "val": float(e_va)},
                      note=f"degree {d}")
        return tr

    t.trace(trace)
