"""Checks for cv_library.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.model_selection import GridSearchCV

from learnml import Grid, Trace


def study_data(seed=2, n=30):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.2, 9.8, n))
    y = 30 + 60 / (1 + np.exp(-(x - 4.5) / 1.3)) + rng.normal(0, 7, n)
    return x, y


def check(s, t):
    x, y = study_data()
    X = x[:, None]
    degrees = list(range(0, 13))
    state = {}

    def t_model():
        m = s.build_model(degree=4, alpha=0.5)
        p = m.get_params()
        assert p["polynomialfeatures__degree"] == 4 and p["ridge__alpha"] == 0.5, "degree and alpha should reach the pipeline"
        m.fit(X, y)
        assert m.predict(X).shape == y.shape, "predict should return one value per sample"

    def t_curve():
        train, valid = s.degree_curve(X, y, degrees)
        assert len(train) == len(degrees) == len(valid), "one value per degree"
        assert np.all(train >= 0) and np.all(valid >= 0), "return MSE (positive), not negative scores"
        assert train[-1] < train[0], "training error should fall with degree"
        best = degrees[int(np.argmin(valid))]
        assert 2 <= best <= 7 and valid[-1] > valid.min() * 1.2, f"validation error should be U-shaped (best degree {best})"
        state["curve"] = (train, valid)

    def t_grid():
        search = s.grid_search(X, y, degrees, [1e-6, 1e-2, 1.0])
        assert isinstance(search, GridSearchCV), "return the fitted GridSearchCV"
        assert len(search.cv_results_["params"]) == len(degrees) * 3, "search every degree × alpha pair"
        mse = -search.best_score_
        assert mse < 70, f"best cross-validated MSE {mse:.1f} should be under 70 (noise variance is 49)"
        state["search"] = search

    t.test("组装多项式管道", "Assembling the polynomial pipeline", t_model)
    t.test("验证曲线呈 U 形", "The validation curve is U-shaped", t_curve)
    t.test("次数 × 正则强度的网格搜索", "Grid search over degree × regularization", t_grid)

    def trace():
        train, valid = s.degree_curve(X, y, degrees)
        P = np.column_stack([x, y])
        grid = Grid(P, n=50, pad=1.0)
        gx, gy = grid.points[:, 0], grid.points[:, 1]
        tr = Trace("classifier2d", P, np.ones(len(y)), grid=grid, show_margin=False,
                   title={"zh": "validation_curve 逐个次数", "en": "validation_curve, degree by degree"})
        for d, a, b in list(zip(degrees, train, valid))[1:]:
            m = s.build_model(degree=d).fit(X, y)
            tr.record(field=m.predict(gx[:, None]) - gy, metrics={"train": float(a), "cv": float(min(b, 150.0))},
                      note=f"d = {d}")
        search = s.grid_search(X, y, degrees, [1e-6, 1e-2, 1.0])
        best = search.best_params_
        tr.record(field=search.best_estimator_.predict(gx[:, None]) - gy,
                  metrics={"train": float(np.mean((search.predict(X) - y) ** 2)), "cv": float(-search.best_score_)},
                  note=f"GridSearchCV: d = {best['polynomialfeatures__degree']}, α = {best['ridge__alpha']:g}")
        return tr

    t.trace(trace)
