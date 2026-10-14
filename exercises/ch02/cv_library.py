"""Model selection with scikit-learn: a polynomial pipeline, a validation curve and a grid search.

Each library call mirrors a piece you wrote by hand: Pipeline = design + fit,
validation_curve = cv_error over a list of degrees, GridSearchCV = select_degree
over several hyperparameters at once.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures


def build_model(degree=3, alpha=1e-6):
    """Rescale x from [0, 10] to [-1, 1], expand to polynomial features, then ridge regression."""
    # @blank pipe
    # zh: make_pipeline(FunctionTransformer(lambda x: (x - 5) / 5), PolynomialFeatures(degree), Ridge(alpha=alpha))
    # en: make_pipeline(FunctionTransformer(lambda x: (x - 5) / 5), PolynomialFeatures(degree), Ridge(alpha=alpha))
    return make_pipeline(FunctionTransformer(lambda x: (x - 5) / 5), PolynomialFeatures(degree), Ridge(alpha=alpha))
    # @end


def degree_curve(X, y, degrees, k=5):
    """Mean training and validation MSE for each degree (5-fold, shuffled with random_state=0)."""
    cv = KFold(k, shuffle=True, random_state=0)
    # @blank curve
    # zh: 调用 validation_curve，param_name="polynomialfeatures__degree"、scoring="neg_mean_squared_error"；把分数取负、按折求均值
    # en: Call validation_curve with param_name="polynomialfeatures__degree" and scoring="neg_mean_squared_error"; negate and average over folds
    train, valid = validation_curve(build_model(), X, y, param_name="polynomialfeatures__degree",
                                    param_range=degrees, cv=cv, scoring="neg_mean_squared_error")
    return -train.mean(axis=1), -valid.mean(axis=1)
    # @end


def grid_search(X, y, degrees, alphas, k=5):
    """Search degree × ridge strength jointly; return the fitted GridSearchCV."""
    cv = KFold(k, shuffle=True, random_state=0)
    # @blank grid
    # zh: GridSearchCV(build_model(), {"polynomialfeatures__degree": degrees, "ridge__alpha": alphas}, cv=cv, scoring="neg_mean_squared_error")，然后 fit
    # en: GridSearchCV(build_model(), {"polynomialfeatures__degree": degrees, "ridge__alpha": alphas}, cv=cv, scoring="neg_mean_squared_error"), then fit
    search = GridSearchCV(build_model(), {"polynomialfeatures__degree": degrees, "ridge__alpha": alphas},
                          cv=cv, scoring="neg_mean_squared_error")
    return search.fit(X, y)
    # @end
