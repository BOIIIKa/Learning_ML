"""Non-parametric methods in scikit-learn: a tuned kNN classifier and an additive spline model (a GAM).

SplineTransformer expands every feature separately into B-splines; a linear model on top is then
a sum of one smooth curve per feature, f(x) = β₀ + Σ_j f_j(x_j): a generalized additive model.
"""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


def knn_search():
    """GridSearchCV(StandardScaler → KNeighborsClassifier) over n_neighbors 1, 3, …, 31 and weights uniform/distance, cv=5."""
    # @blank knn
    # zh: 管道里的参数名带前缀：kneighborsclassifier__n_neighbors、kneighborsclassifier__weights
    # en: Pipeline parameter names carry a prefix: kneighborsclassifier__n_neighbors, kneighborsclassifier__weights
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
    grid = {"kneighborsclassifier__n_neighbors": list(range(1, 32, 2)),
            "kneighborsclassifier__weights": ["uniform", "distance"]}
    return GridSearchCV(pipe, grid, cv=5)
    # @end


def gam():
    """Pipeline: SplineTransformer(n_knots=5, degree=3) → RidgeCV(alphas=np.logspace(−3, 3, 13))."""
    # @blank gam
    # zh: 样条展开 + 带交叉验证的岭回归，就是一个可加模型
    # en: Spline expansion + cross-validated ridge regression is an additive model
    return make_pipeline(SplineTransformer(n_knots=5, degree=3), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    # @end


def compare(X, y):
    """Mean 5-fold CV R² of (linear RidgeCV with the same alphas, the GAM), as a tuple."""
    # @blank compare
    # zh: cross_val_score 默认对回归器用 R²
    # en: cross_val_score uses R² for regressors by default
    lin = RidgeCV(alphas=np.logspace(-3, 3, 13))
    return float(np.mean(cross_val_score(lin, X, y, cv=5))), float(np.mean(cross_val_score(gam(), X, y, cv=5)))
    # @end


def partial_curve(model, X, j, grid):
    """The fitted GAM's prediction along feature j (values in `grid`), other features fixed at their means."""
    # @blank curve
    # zh: 复制 len(grid) 份均值行，把第 j 列换成 grid，再 predict
    # en: Repeat the mean row len(grid) times, replace column j by grid, then predict
    Z = np.tile(X.mean(axis=0), (len(grid), 1))
    Z[:, j] = grid
    return model.predict(Z)
    # @end
