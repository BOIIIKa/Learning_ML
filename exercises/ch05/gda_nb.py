"""Gaussian discriminant analysis (LDA / QDA) and Bernoulli naive Bayes from scratch.

Labels are y ∈ {0, 1}. Everything is a closed-form maximum-likelihood (or MAP) estimate:
no iterative optimization anywhere.
"""

import numpy as np


def fit_gaussians(X, y):
    """Return (priors, means, covs): arrays of shape (2,), (2, p), (2, p, p), MLE covariances (divide by n_k)."""
    # @blank stats
    # zh: 对 k = 0, 1：先验 n_k / n，均值，协方差 (X_k − μ_k)ᵀ(X_k − μ_k) / n_k
    # en: For k = 0, 1: prior n_k / n, mean, covariance (X_k − μ_k)ᵀ(X_k − μ_k) / n_k
    priors, means, covs = [], [], []
    for k in (0, 1):
        Xk = X[y == k]
        mu = Xk.mean(axis=0)
        priors.append(len(Xk) / len(X))
        means.append(mu)
        covs.append((Xk - mu).T @ (Xk - mu) / len(Xk))
    return np.array(priors), np.array(means), np.array(covs)
    # @end


def pooled(priors, covs):
    """Shared covariance for LDA: the prior-weighted average of the class covariances."""
    # @blank pooled
    # zh: Σ = π₀Σ₀ + π₁Σ₁（这里 π_k = n_k / n）
    # en: Σ = π₀Σ₀ + π₁Σ₁ (here π_k = n_k / n)
    return priors[0] * covs[0] + priors[1] * covs[1]
    # @end


def lda_line(priors, means, S):
    """LDA log-odds log P(1|x)/P(0|x) = w·x + b. Returns (w, b)."""
    # @blank lda
    # zh: w = Σ⁻¹(μ₁ − μ₀)，b = −½(μ₁ᵀΣ⁻¹μ₁ − μ₀ᵀΣ⁻¹μ₀) + log(π₁/π₀)；用 np.linalg.solve
    # en: w = Σ⁻¹(μ₁ − μ₀), b = −½(μ₁ᵀΣ⁻¹μ₁ − μ₀ᵀΣ⁻¹μ₀) + log(π₁/π₀); use np.linalg.solve
    Si_m = np.linalg.solve(S, means.T)  # columns Σ⁻¹μ₀, Σ⁻¹μ₁
    w = Si_m[:, 1] - Si_m[:, 0]
    b = -0.5 * (means[1] @ Si_m[:, 1] - means[0] @ Si_m[:, 0]) + np.log(priors[1] / priors[0])
    return w, b
    # @end


def log_gauss(X, mu, S):
    """log N(x; μ, Σ) for every row of X."""
    D = X - mu
    _, logdet = np.linalg.slogdet(S)
    return -0.5 * np.sum(D * np.linalg.solve(S, D.T).T, axis=1) - 0.5 * logdet - 0.5 * X.shape[1] * np.log(2 * np.pi)


def qda_log_odds(X, priors, means, covs):
    """QDA log-odds for every row of X, each class with its own covariance."""
    # @blank qda
    # zh: log(π₁/π₀) + log N(x; μ₁, Σ₁) − log N(x; μ₀, Σ₀)，用上面的 log_gauss
    # en: log(π₁/π₀) + log N(x; μ₁, Σ₁) − log N(x; μ₀, Σ₀), using log_gauss above
    return np.log(priors[1] / priors[0]) + log_gauss(X, means[1], covs[1]) - log_gauss(X, means[0], covs[0])
    # @end


def bernoulli_nb(X, y, alpha=1.0):
    """Bernoulli naive Bayes on binary features. Returns (priors, theta) with theta[k, j] = P(x_j = 1 | y = k)."""
    # @blank nb
    # zh: 先验是类别频率；θ_kj = (类 k 中 x_j = 1 的个数 + α) / (n_k + 2α)
    # en: Priors are class frequencies; θ_kj = (#{x_j = 1 in class k} + α) / (n_k + 2α)
    priors = np.array([np.mean(y == k) for k in (0, 1)])
    theta = np.array([(X[y == k].sum(axis=0) + alpha) / ((y == k).sum() + 2 * alpha) for k in (0, 1)])
    return priors, theta
    # @end


def nb_posterior(x, priors, theta):
    """P(y = 1 | x) for one binary vector x."""
    with np.errstate(divide="ignore"):
        logp = np.log(priors) + np.sum(np.log(np.where(x == 1, theta, 1 - theta)), axis=1)
    if np.all(np.isinf(logp)):
        return 0.5
    p = np.exp(logp - logp.max())
    return float(p[1] / p.sum())
