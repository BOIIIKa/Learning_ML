"""Gaussian discriminant analysis (LDA and QDA) from scratch, with numpy only.

Model: a prior π_k and a Gaussian N(μ_k, Σ_k) for each class. Strategy: maximize the joint
likelihood Π p(x_i, y_i). Algorithm: closed-form estimates (counting, averaging), then
classify by the largest posterior, i.e. the largest δ_k(x) = log π_k + log N(x; μ_k, Σ_k).
"""

import numpy as np


def fit_gaussians(X, y, shared=False):
    """Maximum-likelihood priors (K,), means (K, d) and covariances (K, d, d).

    With shared=True every class gets the pooled covariance Σ = Σ_k π_k Σ_k (LDA);
    otherwise each class keeps its own (QDA).
    """
    classes = np.unique(y)
    # @blank params
    # zh: 先验 = 各类样本占比；均值 = 类内平均；协方差用 1/n_k 归一化（极大似然，np.cov(..., bias=True)）
    # en: Prior = class share; mean = within-class average; covariance normalized by 1/n_k (MLE, np.cov(..., bias=True))
    priors = np.array([np.mean(y == k) for k in classes])
    means = np.array([X[y == k].mean(axis=0) for k in classes])
    covs = np.array([np.cov(X[y == k].T, bias=True) for k in classes])
    # @end
    if shared:
        # @blank pooled
        # zh: 合并协方差：按先验加权平均各类协方差，再复制给每一类
        # en: Pooled covariance: average the class covariances weighted by the priors, then give it to every class
        pooled = np.einsum("k,kij->ij", priors, covs)
        covs = np.array([pooled for _ in classes])
        # @end
    return classes, priors, means, covs


def log_gaussian(X, mean, cov):
    """log N(x; mean, cov) for every row x of X."""
    d = X.shape[1]
    diff = X - mean
    # @blank logpdf
    # zh: −½[(x−μ)ᵀΣ⁻¹(x−μ) + log det Σ + d log 2π]；用 np.linalg.solve 与 np.linalg.slogdet，不要显式求逆
    # en: −½[(x−μ)ᵀΣ⁻¹(x−μ) + log det Σ + d log 2π]; use np.linalg.solve and np.linalg.slogdet rather than an explicit inverse
    maha = np.sum(diff * np.linalg.solve(cov, diff.T).T, axis=1)
    _, logdet = np.linalg.slogdet(cov)
    return -0.5 * (maha + logdet + d * np.log(2 * np.pi))
    # @end


def discriminants(X, priors, means, covs):
    """δ_k(x) = log π_k + log N(x; μ_k, Σ_k): one column per class."""
    # @blank delta
    # zh: 对每一类算 log π_k + log_gaussian(X, μ_k, Σ_k)，按列拼起来
    # en: For each class compute log π_k + log_gaussian(X, μ_k, Σ_k) and stack the results as columns
    return np.column_stack([np.log(p) + log_gaussian(X, m, S) for p, m, S in zip(priors, means, covs)])
    # @end


def predict(X, model):
    classes, priors, means, covs = model
    return classes[np.argmax(discriminants(X, priors, means, covs), axis=1)]


def lda_linear(priors, means, cov):
    """With a shared covariance, δ_1(x) − δ_0(x) = w·x + b. Return (w, b)."""
    # @blank linear
    # zh: w = Σ⁻¹(μ₁ − μ₀)，b = −½(μ₁ᵀΣ⁻¹μ₁ − μ₀ᵀΣ⁻¹μ₀) + log(π₁/π₀)
    # en: w = Σ⁻¹(μ₁ − μ₀), b = −½(μ₁ᵀΣ⁻¹μ₁ − μ₀ᵀΣ⁻¹μ₀) + log(π₁/π₀)
    w = np.linalg.solve(cov, means[1] - means[0])
    b = -0.5 * (means[1] @ np.linalg.solve(cov, means[1]) - means[0] @ np.linalg.solve(cov, means[0])) + np.log(priors[1] / priors[0])
    return w, b
    # @end
