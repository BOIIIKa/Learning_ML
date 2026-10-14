"""EM for Gaussian mixture models from scratch.

X: (n, d) data; weights: (K,), means: (K, d), covs: (K, d, d).
"""

import numpy as np


def log_gauss(X, mu, S):
    """log N(x; μ, Σ) for every row of X."""
    d = X.shape[1]
    D = X - mu
    _, logdet = np.linalg.slogdet(S)
    return -0.5 * np.sum(D * np.linalg.solve(S, D.T).T, axis=1) - 0.5 * logdet - 0.5 * d * np.log(2 * np.pi)


def e_step(X, weights, means, covs):
    """Responsibilities R (n, K) and the log-likelihood, computed stably with log-sum-exp."""
    # @blank estep
    # zh: L[i, k] = log π_k + log N(x_i; μ_k, Σ_k)；按行做 log-sum-exp 得 z_i；R = exp(L − z)，似然 = Σ z_i
    # en: L[i, k] = log π_k + log N(x_i; μ_k, Σ_k); a row-wise log-sum-exp gives z_i; R = exp(L − z), log-likelihood = Σ z_i
    L = np.column_stack([np.log(w) + log_gauss(X, m, S) for w, m, S in zip(weights, means, covs)])
    mx = L.max(axis=1, keepdims=True)
    z = mx + np.log(np.exp(L - mx).sum(axis=1, keepdims=True))
    return np.exp(L - z), float(z.sum())
    # @end


def m_step(X, R, reg=1e-6):
    """New (weights, means, covs) from responsibilities R; add reg·I to every covariance."""
    n, d = X.shape
    # @blank mstep
    # zh: N_k = R 的列和；π_k = N_k / n；μ_k = Rᵀ X / N_k；Σ_k = Σ_i R_ik (x_i − μ_k)(x_i − μ_k)ᵀ / N_k + reg·I
    # en: N_k = column sums of R; π_k = N_k / n; μ_k = Rᵀ X / N_k; Σ_k = Σ_i R_ik (x_i − μ_k)(x_i − μ_k)ᵀ / N_k + reg·I
    Nk = R.sum(axis=0)
    weights = Nk / n
    means = (R.T @ X) / Nk[:, None]
    covs = np.array([((R[:, k, None] * (X - means[k])).T @ (X - means[k])) / Nk[k] + reg * np.eye(d) for k in range(R.shape[1])])
    # @end
    return weights, means, covs


def em(X, weights, means, covs, iters=100, tol=1e-8, trace=None):
    """Run EM until the log-likelihood improves by less than tol. Returns (params, log-likelihoods)."""
    lls = []
    for t in range(iters):
        R, ll = e_step(X, weights, means, covs)
        if trace is not None:
            trace(t, R, (weights, means, covs), ll)
        lls.append(ll)
        if t and lls[-1] - lls[-2] < tol:
            break
        weights, means, covs = m_step(X, R)
    return (weights, means, covs), lls
