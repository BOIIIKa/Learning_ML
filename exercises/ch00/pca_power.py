"""Principal directions of a 2-D data cloud: covariance by maximum likelihood, then power iteration.

The top eigenvector of the covariance matrix is the direction of largest variance.
Power iteration finds it by repeatedly applying the matrix and renormalizing; deflation
removes the found component so the same procedure finds the next one.
"""

import numpy as np


def mle_covariance(X):
    """Maximum-likelihood covariance (divide by n, not n − 1)."""
    # @blank cov
    # zh: 先减去均值，再计算平均外积 Σ = (1/n) Σᵢ (xᵢ − μ)(xᵢ − μ)ᵀ
    # en: Center the data, then average the outer products: Σ = (1/n) Σᵢ (xᵢ − μ)(xᵢ − μ)ᵀ
    Xc = X - X.mean(axis=0)
    return Xc.T @ Xc / len(X)
    # @end


def power_step(S, v):
    """One power-iteration step: v ← S v / ‖S v‖."""
    # @blank power
    # zh: 用矩阵乘一次向量，再归一化为单位长度
    # en: Multiply by the matrix once, then normalize to unit length
    u = S @ v
    return u / np.linalg.norm(u)
    # @end


def rayleigh(S, v):
    """Eigenvalue estimate for a unit vector v."""
    # @blank rayleigh
    # zh: 瑞利商 vᵀ S v（v 已是单位向量）
    # en: The Rayleigh quotient vᵀ S v (v is already a unit vector)
    return float(v @ S @ v)
    # @end


def deflate(S, lam, v):
    """Remove the found component so the next power iteration finds the next eigenvector."""
    # @blank deflate
    # zh: 减去秩一矩阵 λ v vᵀ
    # en: Subtract the rank-one matrix λ v vᵀ
    return S - lam * np.outer(v, v)
    # @end


def principal_directions(X, iters=50, trace=None):
    """Returns (eigenvalues, eigenvectors as rows), largest first. Records the first direction per step."""
    X = np.asarray(X, dtype=float)
    S = mle_covariance(X)
    mean = X.mean(axis=0)
    values, vectors = [], []
    for k in range(2):
        v = np.array([1.0, 0.0]) if k == 0 else np.array([0.0, 1.0])
        for step in range(iters):
            if trace is not None and k == 0 and step < 15:
                normal = np.array([-v[1], v[0]])  # the line through the mean along v
                trace.record(w=normal, b=-normal @ mean, note=f"iter {step}")
            v = power_step(S, v)
        lam = rayleigh(S, v)
        values.append(lam)
        vectors.append(v)
        S = deflate(S, lam, v)
    return np.array(values), np.array(vectors)
