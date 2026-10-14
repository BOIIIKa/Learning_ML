"""Basis expansions with scikit-learn: cubic-spline features + ridge, chosen by cross-validation.

SplineTransformer expands every input column into its own B-spline basis, so the same pipeline
on several columns fits an additive model (GAM): y ≈ β0 + f_1(x_1) + … + f_d(x_d).
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer


def spline_model(n_knots, alpha=1e-3):
    """Cubic B-spline features with `n_knots` knots (boundary knots included), then ridge regression."""
    # @blank model
    # zh: make_pipeline(SplineTransformer(n_knots=n_knots, degree=3), Ridge(alpha=alpha))
    # en: make_pipeline(SplineTransformer(n_knots=n_knots, degree=3), Ridge(alpha=alpha))
    return make_pipeline(SplineTransformer(n_knots=n_knots, degree=3), Ridge(alpha=alpha))
    # @end


def cv_rmse(model, X, y):
    """5-fold cross-validated root-mean-squared error (a positive number)."""
    # @blank cv
    # zh: cross_val_score(..., cv=5, scoring="neg_root_mean_squared_error") 返回负的 RMSE，取反后求平均
    # en: cross_val_score(..., cv=5, scoring="neg_root_mean_squared_error") returns negative RMSEs; negate and average
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
    return float(-np.mean(scores))
    # @end


def choose_knots(X, y, candidates):
    """The number of knots in `candidates` with the smallest cross-validated RMSE."""
    # @blank choose
    # zh: 对每个候选值建模型、算 cv_rmse，返回误差最小的那个候选值
    # en: Build a model for each candidate, compute cv_rmse, and return the candidate with the smallest error
    errs = [cv_rmse(spline_model(n), X, y) for n in candidates]
    return candidates[int(np.argmin(errs))]
    # @end
