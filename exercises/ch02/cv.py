"""k-fold cross-validation from scratch, used to choose a polynomial degree.

The training error always falls as the degree grows; the cross-validated error
falls and then rises again. Its minimum is our estimate of the best complexity.
"""

import numpy as np


def design(x, d):
    """Legendre features P_0..P_d of x rescaled from [0, 10] to [-1, 1] (better conditioned than raw powers)."""
    return np.polynomial.legendre.legvander((np.asarray(x, dtype=float) - 5) / 5, d)


def fit_poly(x, y, d):
    """Least-squares coefficients of a degree-d polynomial."""
    # @blank fit
    # zh: 构造设计矩阵 design(x, d)，用 np.linalg.lstsq 求最小二乘系数
    # en: Build design(x, d) and solve least squares with np.linalg.lstsq
    return np.linalg.lstsq(design(x, d), y, rcond=None)[0]
    # @end


def predict(coef, x):
    return design(x, len(coef) - 1) @ coef


def kfold_indices(n, k, seed=0):
    """Shuffle 0..n-1 with the given seed and cut it into k folds of (almost) equal size."""
    # @blank folds
    # zh: 用 np.random.default_rng(seed).permutation(n) 打乱，再用 np.array_split 切成 k 份
    # en: Shuffle with np.random.default_rng(seed).permutation(n), then cut into k parts with np.array_split
    order = np.random.default_rng(seed).permutation(n)
    return np.array_split(order, k)
    # @end


def cv_error(x, y, d, k=5, seed=0):
    """Average held-out mean squared error of degree-d fits over the k folds."""
    errs = []
    for held in kfold_indices(len(x), k, seed):
        # @blank cv
        # zh: 其余样本作训练集（np.setdiff1d），在训练集上拟合，在留出的 held 上算均方误差
        # en: Train on the rest (np.setdiff1d), then compute the mean squared error on the held-out fold
        train = np.setdiff1d(np.arange(len(x)), held)
        coef = fit_poly(x[train], y[train], d)
        errs.append(np.mean((predict(coef, x[held]) - y[held]) ** 2))
        # @end
    return float(np.mean(errs))


def select_degree(x, y, degrees, k=5, seed=0):
    """The degree with the smallest cross-validated error."""
    # @blank select
    # zh: 对每个候选次数算 cv_error，返回误差最小的那个次数
    # en: Compute cv_error for each candidate degree and return the one with the smallest error
    errs = [cv_error(x, y, d, k, seed) for d in degrees]
    return degrees[int(np.argmin(errs))]
    # @end
