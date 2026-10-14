"""PCA from scratch: centring, power iteration for the top direction, and rank-k reconstruction by SVD."""

import numpy as np


def center(X):
    """Return (X − column means, column means)."""
    m = X.mean(axis=0)
    return X - m, m


def power_iteration(S, iters=100, v0=None, trace=None):
    """Top eigenvector of a symmetric PSD matrix S by power iteration. Returns (v, Rayleigh quotient)."""
    v = np.ones(S.shape[0]) if v0 is None else np.asarray(v0, dtype=float)
    v = v / np.linalg.norm(v)
    for t in range(iters):
        # @blank power
        # zh: v ← S v，再归一化
        # en: v ← S v, then normalize
        v = S @ v
        v = v / np.linalg.norm(v)
        # @end
        if trace is not None:
            trace(t, v, float(v @ S @ v))
    return v, float(v @ S @ v)


def pca(X, k):
    """Return (components (k, p), explained variances (k,), scores (n, k)) using the SVD of the centred data."""
    Xc, _ = center(X)
    # @blank pca
    # zh: U, s, Vt = np.linalg.svd(Xc, full_matrices=False)；主方向是 Vt 的前 k 行，方差是 s² / n，得分是 Xc @ Vtᵀ
    # en: U, s, Vt = np.linalg.svd(Xc, full_matrices=False); directions are the first k rows of Vt, variances s² / n, scores Xc @ Vtᵀ
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:k]
    return comps, s[:k] ** 2 / len(X), Xc @ comps.T
    # @end


def reconstruct(X, k):
    """Rank-k reconstruction of X (in the original coordinates, means added back)."""
    Xc, m = center(X)
    # @blank recon
    # zh: 截断 SVD：U[:, :k] · diag(s[:k]) · Vt[:k]，再加回均值
    # en: Truncated SVD: U[:, :k] · diag(s[:k]) · Vt[:k], then add the means back
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :k] * s[:k] @ Vt[:k] + m
    # @end
