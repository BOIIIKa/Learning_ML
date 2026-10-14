"""Ridge and lasso in scikit-learn, with the penalty chosen by cross-validation.

The diabetes data (442 patients, 10 standardized measurements) is small enough to fit
in the browser. RidgeCV and LassoCV each search a grid of penalties internally.
"""

import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV, lasso_path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_models():
    """Two pipelines: StandardScaler → RidgeCV(alphas=logspace(−3, 3, 30)), and StandardScaler → LassoCV(cv=5)."""
    # @blank models
    # zh: 用 make_pipeline 组装；LassoCV 设 cv=5、random_state=0
    # en: Assemble with make_pipeline; give LassoCV cv=5 and random_state=0
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30)))
    lasso = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=0))
    return ridge, lasso
    # @end


def selected(lasso_pipe, names):
    """Names of the features whose lasso coefficient is nonzero."""
    # @blank selected
    # zh: 系数在 lasso_pipe[-1].coef_ 中，挑出非零的那些对应的名字
    # en: The coefficients live in lasso_pipe[-1].coef_; keep the names whose coefficient is nonzero
    coef = lasso_pipe[-1].coef_
    return [n for n, c in zip(names, coef) if c != 0]
    # @end


def path(X, y, alphas):
    """Lasso coefficients for each alpha, as a list of arrays (one per alpha)."""
    # @blank path
    # zh: 调用 lasso_path(X, y, alphas=alphas)，返回值第二项形状为 (特征数, alpha 数)，按列拆开
    # en: Call lasso_path(X, y, alphas=alphas); its second output has shape (features, alphas), so split it by column
    _, coefs, _ = lasso_path(X, y, alphas=alphas)
    return [coefs[:, k] for k in range(coefs.shape[1])]
    # @end
