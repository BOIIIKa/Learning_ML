"""Checks for spline_library.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.linear_model import LinearRegression

from learnml import Grid, Trace


def trend(x):
    return 14 + 3.2 * x + 6 * np.sin(1.1 * x)


def rush(h):
    """Extra minutes by hour of day: a lunch and a dinner peak."""
    return 8 * np.exp(-((h - 12) ** 2) / 2) + 10 * np.exp(-((h - 19) ** 2) / 3)


def delivery(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.3, 9.7, n)
    return x, trend(x) + rng.normal(0, 3, n)


def check(s, t):
    x, y = delivery(80, 0)
    xt, yt = delivery(400, 1)
    X, Xt = x[:, None], xt[:, None]

    def t_model():
        m = s.spline_model(7, alpha=0.5)
        steps = [type(st).__name__ for _, st in m.steps]
        assert steps == ["SplineTransformer", "Ridge"], f"pipeline steps {steps}"
        sp, rg = m.steps[0][1], m.steps[1][1]
        assert (sp.n_knots, sp.degree, rg.alpha) == (7, 3, 0.5), f"(n_knots, degree, alpha) = {(sp.n_knots, sp.degree, rg.alpha)}"
        m.fit(X, y)
        assert m[:-1].transform(X).shape[1] == 9, "cubic B-splines with 7 knots give 7 + 3 − 1 = 9 basis functions"

    def t_cv():
        m = s.spline_model(6)
        got = s.cv_rmse(m, X, y)
        assert got > 0, f"cv_rmse should be positive, got {got:.3f}"
        lin = s.cv_rmse(LinearRegression(), X, y)
        assert got < lin, f"spline CV RMSE {got:.2f} should beat a straight line's {lin:.2f}"

    def t_choose():
        cands = [2, 3, 4, 6, 8, 12, 20, 30]
        k = s.choose_knots(X, y, cands)
        errs = [s.cv_rmse(s.spline_model(n), X, y) for n in cands]
        assert k == cands[int(np.argmin(errs))], f"chose {k}, but the smallest CV RMSE is at {cands[int(np.argmin(errs))]}"
        assert k not in (2, 30), f"chose {k} knots: the extremes should under- or over-fit"
        rm = float(np.sqrt(np.mean((s.spline_model(k).fit(X, y).predict(Xt) - yt) ** 2)))
        assert rm < 3.6, f"test RMSE {rm:.2f} min should be close to the noise level 3"

    def t_gam():
        rng = np.random.default_rng(2)
        n = 400
        Z = np.column_stack([rng.uniform(0.3, 9.7, n), rng.uniform(10, 22, n)])
        yz = trend(Z[:, 0]) + rush(Z[:, 1]) + rng.normal(0, 2, n)
        gam = s.spline_model(8, alpha=1e-2).fit(Z, yz)
        # Vary one input with the other held fixed: the curve should follow that input's true shape.
        hs = np.linspace(11, 21, 50)
        part = gam.predict(np.column_stack([np.full(50, 5.0), hs]))
        corr = np.corrcoef(part, rush(hs))[0, 1]
        assert corr > 0.95, f"the hour-of-day component correlates only {corr:.2f} with the true rush-hour curve"
        lin = LinearRegression().fit(Z, yz)
        assert np.mean((gam.predict(Z) - yz) ** 2) < 0.5 * np.mean((lin.predict(Z) - yz) ** 2), "the GAM should fit far better than a linear model"

    t.test("搭建样条 + 岭回归流水线", "Building the spline + ridge pipeline", t_model)
    t.test("交叉验证 RMSE", "Cross-validated RMSE", t_cv)
    t.test("用交叉验证选节点数", "Choosing the number of knots by CV", t_choose)
    t.test("同一流水线在两列上就是 GAM", "The same pipeline on two columns is a GAM", t_gam)

    def trace():
        P = np.column_stack([x, y])
        grid = Grid(P, n=50, pad=0.5)
        label = np.where(y > trend(x), 1, -1)
        tr = Trace("classifier2d", P, label, grid=grid, title={
            "zh": "节点数增加时的样条曲线（点色：慢于 / 快于真实趋势）",
            "en": "The spline as knots are added (colour: slower / faster than the true trend)"})
        for k in [2, 3, 4, 5, 6, 8, 10, 14, 20, 30]:
            m = s.spline_model(k).fit(X, y)
            train = float(np.sqrt(np.mean((m.predict(X) - y) ** 2)))
            # The field is (time − fitted curve), so its zero contour draws the curve itself.
            field = grid.points[:, 1] - m.predict(grid.points[:, :1])
            tr.record(field=field, metrics={"train": train, "cv": s.cv_rmse(s.spline_model(k), X, y)}, note=f"{k} knots")
        return tr

    t.trace(trace)
