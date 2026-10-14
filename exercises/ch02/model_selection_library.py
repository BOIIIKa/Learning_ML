"""Model selection with scikit-learn: a polynomial pipeline, cross_val_score and validation_curve."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def make_model(degree):
    """Standardize x, expand it into polynomial features of the given degree, then fit least squares."""
    # @blank pipeline
    # zh: make_pipeline(StandardScaler(), PolynomialFeatures(degree), LinearRegression())
    # en: make_pipeline(StandardScaler(), PolynomialFeatures(degree), LinearRegression())
    return make_pipeline(StandardScaler(), PolynomialFeatures(degree), LinearRegression())
    # @end


def cv_score(X, y, degree, k=5):
    """k-fold CV estimate of the test MSE (shuffled folds, random_state=0)."""
    # @blank cross-val
    # zh: cv = KFold(n_splits=k, shuffle=True, random_state=0)；cross_val_score 的 scoring 取 "neg_mean_squared_error"，返回负的平均值
    # en: cv = KFold(n_splits=k, shuffle=True, random_state=0); use scoring="neg_mean_squared_error" in cross_val_score and return minus the mean
    cv = KFold(n_splits=k, shuffle=True, random_state=0)
    scores = cross_val_score(make_model(degree), X, y, cv=cv, scoring="neg_mean_squared_error")
    return float(-scores.mean())
    # @end


def degree_curve(X, y, degrees, k=5):
    """Training and validation MSE for every degree, as two arrays (averaged over the k folds)."""
    # @blank curve
    # zh: validation_curve(make_model(1), X, y, param_name="polynomialfeatures__degree", param_range=degrees, cv=..., scoring=...)
    # en: validation_curve(make_model(1), X, y, param_name="polynomialfeatures__degree", param_range=degrees, cv=..., scoring=...)
    cv = KFold(n_splits=k, shuffle=True, random_state=0)
    train, val = validation_curve(make_model(1), X, y, param_name="polynomialfeatures__degree",
                                  param_range=degrees, cv=cv, scoring="neg_mean_squared_error")
    return -train.mean(axis=1), -val.mean(axis=1)
    # @end


def pick_degree(degrees, val_mse):
    """The degree with the smallest validation error."""
    # @blank pick
    # zh: 用 np.argmin 找验证误差最小的位置
    # en: Use np.argmin to find where the validation error is smallest
    return degrees[int(np.argmin(val_mse))]
    # @end
