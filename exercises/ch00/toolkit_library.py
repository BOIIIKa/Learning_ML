"""The same tools from libraries: PCA from scikit-learn, constrained optimization from SciPy."""

import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA


def pca_directions(X):
    """Principal directions (rows) and their variances, largest first."""
    # @blank pca
    # zh: 用 PCA(n_components=2) 拟合 X，返回 components_ 与 explained_variance_
    # en: Fit PCA(n_components=2) on X and return components_ and explained_variance_
    pca = PCA(n_components=2).fit(X)
    return pca.components_, pca.explained_variance_
    # @end


def project_halfplane(t, a, c):
    """argmin ‖x − t‖² subject to a·x ≤ c, solved numerically."""
    # @blank slsqp
    # zh: 用 scipy.optimize.minimize（method="SLSQP"），约束写成 {"type": "ineq", "fun": lambda x: c - a @ x}（要求 ≥ 0），初值取 t
    # en: Use scipy.optimize.minimize (method="SLSQP") with the constraint {"type": "ineq", "fun": lambda x: c - a @ x} (required ≥ 0), starting from t
    res = minimize(lambda x: np.sum((x - t) ** 2), x0=t, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": lambda x: c - a @ x}])
    return res.x
    # @end
