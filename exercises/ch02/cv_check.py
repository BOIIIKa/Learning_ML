"""Checks for cv.py. Runs identically under CPython and Pyodide."""

import numpy as np

from learnml import Grid, Trace


def study_data(seed=2, n=30):
    """Scores that saturate with study hours, plus noise (the chapter's running example)."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.2, 9.8, n))
    y = 30 + 60 / (1 + np.exp(-(x - 4.5) / 1.3)) + rng.normal(0, 7, n)
    return x, y


def check(s, t):
    x, y = study_data()

    def t_fit():
        coef = s.fit_poly(x, y, 3)
        ref = np.linalg.lstsq(s.design(x, 3), y, rcond=None)[0]
        assert np.allclose(coef, ref, atol=1e-6), "least-squares coefficients"

    def t_folds():
        f = s.kfold_indices(23, 5, seed=1)
        assert len(f) == 5, f"{len(f)} folds, expected 5"
        allidx = np.sort(np.concatenate(f))
        assert np.array_equal(allidx, np.arange(23)), "every index must appear in exactly one fold"
        sizes = sorted(len(v) for v in f)
        assert sizes[-1] - sizes[0] <= 1, f"fold sizes {sizes} should differ by at most one"
        assert not np.array_equal(np.concatenate(f), np.arange(23)), "indices should be shuffled"

    def t_cv():
        got = s.cv_error(x, y, 3, k=5, seed=0)
        folds = s.kfold_indices(len(x), 5, 0)
        errs = []
        for held in folds:
            tr = np.setdiff1d(np.arange(len(x)), held)
            c = np.linalg.lstsq(s.design(x[tr], 3), y[tr], rcond=None)[0]
            errs.append(np.mean((s.design(x[held], 3) @ c - y[held]) ** 2))
        assert np.isclose(got, np.mean(errs)), f"cv error {got:.3f} vs expected {np.mean(errs):.3f}"
        train = np.mean((s.design(x, 3) @ s.fit_poly(x, y, 3) - y) ** 2)
        assert got > train, "held-out error should exceed training error"

    def t_select():
        degrees = list(range(0, 13))
        d = s.select_degree(x, y, degrees)
        assert 2 <= d <= 7, f"selected degree {d}; expected a moderate degree (2–7)"
        assert s.cv_error(x, y, 12) > s.cv_error(x, y, d), "degree 12 should cross-validate worse"

    t.test("最小二乘多项式拟合", "Least-squares polynomial fit", t_fit)
    t.test("k 折划分", "k-fold split", t_folds)
    t.test("交叉验证误差", "Cross-validation error", t_cv)
    t.test("用交叉验证选次数", "Choosing the degree by cross-validation", t_select)

    def trace():
        X = np.column_stack([x, y])
        grid = Grid(X, n=50, pad=1.0)
        gx, gy = grid.points[:, 0], grid.points[:, 1]
        tr = Trace("classifier2d", X, np.ones(len(y)), grid=grid, show_margin=False,
                   title={"zh": "次数从 1 到 12：训练误差 vs 交叉验证误差", "en": "Degree 1 to 12: training vs cross-validated error"})
        for d in range(1, 13):
            coef = s.fit_poly(x, y, d)
            train = float(np.mean((s.predict(coef, x) - y) ** 2))
            tr.record(field=s.predict(coef, gx) - gy,
                      metrics={"train": train, "cv": min(s.cv_error(x, y, d), 150.0)},
                      note=f"d = {d}")
        return tr

    t.trace(trace)
