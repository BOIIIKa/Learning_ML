"""Checks for cv_bias_variance.py. Runs identically under CPython and Pyodide."""

import numpy as np

from learnml import Grid, Trace


def truth(h):
    return 30 + 50 / (1 + np.exp(-(h - 4)))


def cohort(n=24, sigma=8.0, seed=21):
    """Study hours spread evenly (one student per bin) and noisy scores, as in the chapter's widgets."""
    rng = np.random.default_rng(seed)
    x = 0.5 + 9 * (np.arange(n) + rng.uniform(size=n)) / n
    return x, truth(x) + sigma * rng.normal(size=n)


def check(s, t):
    x, y = cohort()

    def t_folds():
        folds = s.kfold_indices(10, 3, np.random.default_rng(0))
        sizes = sorted(len(f) for f in folds)
        assert sizes == [3, 3, 4], f"fold sizes {sizes}, expected [3, 3, 4]"
        allidx = np.sort(np.concatenate(folds))
        assert np.array_equal(allidx, np.arange(10)), "the folds must partition 0..9 (each index exactly once)"
        assert not np.array_equal(np.concatenate(folds), np.arange(10)), "shuffle before splitting"

    def t_loo():
        d = 3
        Phi = s.features(x, d)
        H = Phi @ np.linalg.solve(Phi.T @ Phi, Phi.T)
        e = y - H @ y
        shortcut = float(np.mean((e / (1 - np.diag(H))) ** 2))
        got = s.cv_mse(x, y, d, k=len(x))
        assert np.isclose(got, shortcut, rtol=1e-6), f"LOO = {got:.4f}, but the hat-matrix shortcut gives {shortcut:.4f}"

    def t_select():
        cv = [s.cv_mse(x, y, d, k=6) for d in range(0, 9)]
        best = int(np.argmin(cv))
        assert 2 <= best <= 5, f"CV errors by degree {np.round(cv, 1)}; the minimum should be at a middle degree, got {best}"
        assert cv[0] > 2 * cv[best] and cv[8] > cv[best], "CV error should be U-shaped in the degree"

    def t_bv():
        xs = np.linspace(0.5, 9.5, 12)
        sigma = 8.0
        for d in (1, 4):
            b2, v = s.bias_variance(truth, xs, sigma, d, xs, reps=3000, seed=1)
            expect_v = sigma ** 2 * (d + 1) / len(xs)
            assert abs(v / expect_v - 1) < 0.07, f"degree {d}: variance {v:.2f}, theory σ²(d+1)/n = {expect_v:.2f}"
            Phi = s.features(xs, d)
            resid = truth(xs) - Phi @ np.linalg.lstsq(Phi, truth(xs), rcond=None)[0]
            expect_b = float(np.mean(resid ** 2))
            assert abs(b2 - expect_b) < 0.05 * expect_b + 3 * expect_v / 3000 + 0.05, (
                f"degree {d}: bias² {b2:.3f}, theory {expect_b:.3f}")

    t.test("k 折划分", "Splitting into k folds", t_folds)
    t.test("留一法与帽子矩阵捷径一致", "Leave-one-out matches the hat-matrix shortcut", t_loo)
    t.test("用 CV 选阶数：U 形", "Choosing the degree by CV: a U shape", t_select)
    t.test("偏差² 与方差符合理论值", "Bias² and variance match theory", t_bv)

    def trace():
        X = np.column_stack([x, y])
        grid = Grid(X, n=48, pad=0.6)
        gx, gy = grid.points[:, 0], grid.points[:, 1]
        tr = Trace("classifier2d", X, -np.ones(len(y)), grid=grid,
                   title={"zh": "6 折交叉验证（3 阶多项式）", "en": "6-fold cross-validation (cubic)"})
        seen = []

        def record(val, w, err):
            seen.append(err)
            tr.record(field=s.predict(w, gx) - gy, highlight=[int(i) for i in val],
                      metrics={"fold": err, "cv": float(np.mean(seen))}, note=f"fold {len(seen)} / 6")

        s.cv_mse(x, y, 3, k=6, on_fold=record)
        return tr

    t.trace(trace)
