"""k-fold cross-validation and a bias–variance simulation, written from scratch with numpy.

Model: polynomials of degree d in u = (x − 5)/5 (Legendre basis, for numerical stability).
Strategy: estimate the test error by k-fold cross-validation.
Algorithm: least squares for each fit; resampling for the estimates.
"""

import numpy as np


def features(x, degree):
    """Columns P0(u), …, Pd(u) with u = (x − 5)/5."""
    return np.polynomial.legendre.legvander((np.asarray(x, dtype=float) - 5) / 5, degree)


def fit(x, y, degree):
    """Least-squares coefficients."""
    return np.linalg.lstsq(features(x, degree), y, rcond=None)[0]


def predict(w, x):
    return features(x, len(w) - 1) @ w


def kfold_indices(n, k, rng):
    """Shuffle 0..n−1 with rng and cut it into k nearly equal folds (a list of index arrays)."""
    # @blank folds
    # zh: 用 rng.permutation(n) 打乱，再用 np.array_split 切成 k 份
    # en: Shuffle with rng.permutation(n), then cut into k parts with np.array_split
    return np.array_split(rng.permutation(n), k)
    # @end


def cv_mse(x, y, degree, k, seed=0, on_fold=None):
    """k-fold CV estimate of the test MSE: the size-weighted average of the per-fold validation MSEs.

    on_fold(val_idx, w, fold_mse) is called after each fold (used to record the replay).
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    total = 0.0
    for val in kfold_indices(n, k, rng):
        # @blank fold
        # zh: 训练集是不在 val 中的下标；在训练集上拟合，在 val 上算 MSE
        # en: The training indices are those not in val; fit on them, then compute the MSE on val
        train = np.setdiff1d(np.arange(n), val)
        w = fit(x[train], y[train], degree)
        err = float(np.mean((predict(w, x[val]) - y[val]) ** 2))
        # @end
        total += err * len(val)
        if on_fold is not None:
            on_fold(val, w, err)
    return total / n


def bias_variance(f, x, sigma, degree, x_test, reps=500, seed=0):
    """Fixed design x: draw reps training sets y = f(x) + N(0, σ²), fit each, and return
    (bias², variance) of the predictions, both averaged over the points x_test."""
    rng = np.random.default_rng(seed)
    preds = np.empty((reps, len(x_test)))
    for r in range(reps):
        # @blank resample
        # zh: 生成一批新的 y（真实值加噪声），拟合，并在 x_test 上预测
        # en: Draw a fresh y (truth plus noise), fit it, and predict at x_test
        y = f(x) + sigma * rng.normal(size=len(x))
        preds[r] = predict(fit(x, y, degree), x_test)
        # @end
    # @blank decompose
    # zh: 平均预测 f̄；偏差² = mean((f̄ − f)²)，方差 = 各点方差的平均
    # en: Mean prediction f̄; bias² = mean((f̄ − f)²), variance = the average of the pointwise variances
    mean = preds.mean(axis=0)
    bias2 = float(np.mean((mean - f(x_test)) ** 2))
    variance = float(np.mean(preds.var(axis=0)))
    # @end
    return bias2, variance
