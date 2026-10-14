"""Likelihood-based models in scikit-learn: log-loss of a classifier, Poisson regression for counts,
and quantile regression for prediction intervals that widen where the data are noisier."""

import numpy as np
from sklearn.linear_model import LinearRegression, PoissonRegressor, QuantileRegressor
from sklearn.metrics import log_loss


def nll_of(model, X, y):
    """Mean negative log-likelihood (cross-entropy) of a fitted classifier on (X, y)."""
    # @blank nll
    # zh: log_loss 需要预测的概率 predict_proba，而不是类别
    # en: log_loss needs the predicted probabilities, predict_proba, not the classes
    return float(log_loss(y, model.predict_proba(X), labels=model.classes_))
    # @end


def fit_counts(X, y):
    """Fit (PoissonRegressor(alpha=0), LinearRegression()) to count data; return both."""
    # @blank counts
    # zh: 泊松回归 = 对数连接 + 泊松负对数似然；线性回归 = 恒等连接 + 平方损失
    # en: Poisson regression = log link + Poisson NLL; linear regression = identity link + squared loss
    return PoissonRegressor(alpha=0).fit(X, y), LinearRegression().fit(X, y)
    # @end


def interval(X, y, level=0.9):
    """Two QuantileRegressor(alpha=0, solver="highs") fits for the (1 − level)/2 and (1 + level)/2
    quantiles; return (lower model, upper model)."""
    # @blank quantile
    # zh: 例如 level = 0.9 时取 0.05 和 0.95 两个分位数
    # en: For level = 0.9, fit the 0.05 and 0.95 quantiles
    lo = QuantileRegressor(quantile=(1 - level) / 2, alpha=0, solver="highs").fit(X, y)
    hi = QuantileRegressor(quantile=(1 + level) / 2, alpha=0, solver="highs").fit(X, y)
    return lo, hi
    # @end
