"""Empirical risk minimization for a line y ≈ w0 + w1·x, written from scratch.

Model: a line. Strategy: the average loss on the training set (empirical risk).
Algorithm: full-batch gradient descent. Each piece is a separate function.
"""

import numpy as np


def design(x):
    """The design matrix [1, x]: one row per sample."""
    return np.column_stack([np.ones_like(x), x])


def squared_risk(w, X, y):
    """Empirical risk under squared loss: (1/n) Σ (xᵢᵀw − yᵢ)²."""
    # @blank sq-risk
    # zh: 先算残差 X @ w − y，再求平方的平均
    # en: Compute the residuals X @ w − y, then average their squares
    r = X @ w - y
    return float(np.mean(r ** 2))
    # @end


def absolute_risk(w, X, y):
    """Empirical risk under absolute loss: (1/n) Σ |xᵢᵀw − yᵢ|."""
    # @blank abs-risk
    # zh: 残差绝对值的平均
    # en: The average absolute residual
    return float(np.mean(np.abs(X @ w - y)))
    # @end


def squared_grad(w, X, y):
    """Gradient of the squared empirical risk with respect to w."""
    # @blank grad
    # zh: ∇ = (2/n) Xᵀ (X w − y)
    # en: ∇ = (2/n) Xᵀ (X w − y)
    return 2.0 / len(y) * X.T @ (X @ w - y)
    # @end


def fit_gd(X, y, lr=0.02, steps=5000, trace=None):
    """Gradient descent from w = 0 (lr must stay below 2 / largest Hessian eigenvalue). Records the line every 250 steps."""
    w = np.zeros(X.shape[1])
    for t in range(steps):
        if trace is not None and t % 250 == 0:
            # The line w0 + w1·x − y = 0 in the (x, y) plane: normal (w1, −1), offset w0.
            trace.record(w=[w[1], -1.0], b=w[0], metrics={"risk": squared_risk(w, X, y)}, note=f"step {t}")
        # @blank step
        # zh: 沿负梯度走一步：w ← w − lr · ∇
        # en: Step against the gradient: w ← w − lr · ∇
        w = w - lr * squared_grad(w, X, y)
        # @end
    return w
