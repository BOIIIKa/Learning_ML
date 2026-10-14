"""The perceptron and logistic regression from scratch.

Both work with augmented inputs x̂ = (x, 1), so the bias is the last weight of ŵ.
Labels are y ∈ {−1, +1} for the perceptron and y ∈ {0, 1} for logistic regression.
"""

import numpy as np


def augment(X):
    """Append a column of ones: rows become x̂ = (x, 1)."""
    return np.column_stack([X, np.ones(len(X))])


def perceptron(X, y, eta=1.0, max_epochs=100, trace=None):
    """Primal perceptron from ŵ = 0, visiting the points in order. Returns (w, b, number of updates)."""
    Xa = augment(X)
    w = np.zeros(Xa.shape[1])
    updates = 0
    if trace is not None:
        trace.record(w=w[:-1], b=w[-1], metrics={"updates": 0}, note="w = 0")
    for _ in range(max_epochs):
        clean = True
        for i in range(len(y)):
            # @blank update
            # zh: 若 y_i ŵ·x̂_i ≤ 0（边界上也算错），则 ŵ ← ŵ + η y_i x̂_i；用 wrong 记录是否更新
            # en: If y_i ŵ·x̂_i ≤ 0 (on the boundary counts as wrong), set ŵ ← ŵ + η y_i x̂_i; record it in wrong
            wrong = y[i] * (Xa[i] @ w) <= 0
            if wrong:
                w = w + eta * y[i] * Xa[i]
            # @end
            if wrong:
                updates += 1
                clean = False
                if trace is not None:
                    trace.record(w=w[:-1], b=w[-1], highlight=[i], metrics={"updates": updates}, note=f"x{i + 1}")
        if clean:
            break
    return w[:-1], w[-1], updates


def sigmoid(z):
    """σ(z) = 1 / (1 + e^(−z)), without overflow for large |z|."""
    # @blank sigmoid
    # zh: 令 e = exp(−|z|)；z ≥ 0 时 σ = 1/(1+e)，否则 σ = e/(1+e)（用 np.where）
    # en: Let e = exp(−|z|); σ = 1/(1+e) when z ≥ 0 and e/(1+e) otherwise (use np.where)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    # @end


def nll(w, Xa, y):
    """Negative log-likelihood Σ log(1 + e^(ŵ·x̂)) − y ŵ·x̂."""
    z = Xa @ w
    return float(np.sum(np.logaddexp(0.0, z) - y * z))


def gradient(w, Xa, y):
    """Gradient of the negative log-likelihood, shape (p + 1,)."""
    # @blank grad
    # zh: X̂ᵀ(p − y)，其中 p = σ(X̂ŵ)
    # en: X̂ᵀ(p − y) with p = σ(X̂ŵ)
    return Xa.T @ (sigmoid(Xa @ w) - y)
    # @end


def hessian(w, Xa):
    """Hessian of the negative log-likelihood, X̂ᵀ S X̂ with S = diag(p(1 − p))."""
    # @blank hess
    # zh: 先算 s = p(1 − p)，再算 X̂ᵀ (s[:, None] * X̂)，不要构造 n×n 的对角矩阵
    # en: Compute s = p(1 − p), then X̂ᵀ (s[:, None] * X̂); don't build the n×n diagonal matrix
    p = sigmoid(Xa @ w)
    return Xa.T @ ((p * (1 - p))[:, None] * Xa)
    # @end


def newton(X, y, steps=10, trace=None):
    """Newton's method (IRLS) from ŵ = 0. Returns (w, b)."""
    Xa = augment(X)
    w = np.zeros(Xa.shape[1])
    for t in range(steps):
        if trace is not None:
            trace.record(w=w[:-1], b=w[-1], metrics={"nll": nll(w, Xa, y)}, note=f"Newton {t}")
        # @blank newton
        # zh: ŵ ← ŵ − H⁻¹ g；用 np.linalg.solve 解线性方程组，不要求逆
        # en: ŵ ← ŵ − H⁻¹ g; solve the linear system with np.linalg.solve instead of inverting
        w = w - np.linalg.solve(hessian(w, Xa), gradient(w, Xa, y))
        # @end
    if trace is not None:
        trace.record(w=w[:-1], b=w[-1], metrics={"nll": nll(w, Xa, y)}, note=f"Newton {steps}")
    return w[:-1], w[-1]
