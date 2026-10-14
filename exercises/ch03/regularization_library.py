"""Ridge, lasso and cross-validated lasso with scikit-learn, on degree-9 polynomial features."""

import numpy as np
from sklearn.linear_model import Lasso, LassoCV, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def make_model(kind, alpha, degree=9):
    """PolynomialFeatures(degree, no bias column) → StandardScaler → Ridge(alpha) or Lasso(alpha)."""
    # @blank pipeline
    # zh: kind 为 "ridge" 时用 Ridge(alpha=alpha)，否则用 Lasso(alpha=alpha, max_iter=100_000)；前面接 PolynomialFeatures(degree, include_bias=False) 与 StandardScaler()
    # en: Use Ridge(alpha=alpha) when kind == "ridge", else Lasso(alpha=alpha, max_iter=100_000); put PolynomialFeatures(degree, include_bias=False) and StandardScaler() in front
    reg = Ridge(alpha=alpha) if kind == "ridge" else Lasso(alpha=alpha, max_iter=100_000)
    return make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), reg)
    # @end


def coef_path(kind, X, y, alphas):
    """Fit one model per alpha; return an array of shape (len(alphas), degree) with the coefficients."""
    # @blank path
    # zh: 对每个 alpha 拟合 make_model(kind, alpha)，取出管道最后一步的 coef_
    # en: Fit make_model(kind, alpha) for each alpha and take coef_ from the pipeline's last step
    return np.array([make_model(kind, a).fit(X, y)[-1].coef_ for a in alphas])
    # @end


def pick_alpha(X, y, alphas, degree=9):
    """Choose alpha by 4-fold cross-validation with LassoCV; return the fitted pipeline."""
    # @blank cv
    # zh: 把 Lasso 换成 LassoCV(alphas=alphas, cv=KFold(4), max_iter=100_000)，其余同 make_model；拟合后返回管道
    # en: Same as make_model but with LassoCV(alphas=alphas, cv=KFold(4), max_iter=100_000); fit and return the pipeline
    cv = LassoCV(alphas=alphas, cv=KFold(4), max_iter=100_000)
    return make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), cv).fit(X, y)
    # @end
