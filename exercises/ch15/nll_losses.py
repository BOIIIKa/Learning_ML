"""Losses as negative log-likelihoods, from scratch: numerically stable binary and softmax
cross-entropy, the Gaussian NLL, and a heteroscedastic regression fitted by gradient descent."""

import numpy as np


def bce_with_logits(y, f):
    """Mean Bernoulli NLL with logits f: log(1 + e^f) − y·f, stable for any f."""
    y = np.asarray(y, dtype=float)
    f = np.asarray(f, dtype=float)
    # @blank bce
    # zh: log(1 + e^f) 用 np.logaddexp(0, f) 计算，不会溢出
    # en: Compute log(1 + e^f) as np.logaddexp(0, f), which never overflows
    return float(np.mean(np.logaddexp(0, f) - y * f))
    # @end


def softmax_ce(y, F):
    """Mean categorical NLL: log Σ_k e^{F_ik} − F_{i, y_i}, stable for any logits F (n, K)."""
    F = np.asarray(F, dtype=float)
    # @blank softmax
    # zh: 先减去每行最大值 m，log Σ e^F = m + log Σ e^{F − m}；再减去正确类的 logit
    # en: Subtract each row's max m first: log Σ e^F = m + log Σ e^{F − m}; then subtract the correct class's logit
    m = F.max(axis=1)
    lse = m + np.log(np.exp(F - m[:, None]).sum(axis=1))
    return float(np.mean(lse - F[np.arange(len(F)), np.asarray(y)]))
    # @end


def gaussian_nll(y, mu, log_sigma):
    """Mean Gaussian NLL: (y − μ)² / (2σ²) + log σ + ½ log 2π, with σ = exp(log_sigma)."""
    # @blank gauss
    # zh: 按公式逐项写出，再取平均
    # en: Write out the three terms, then average
    r = np.asarray(y, dtype=float) - mu
    return float(np.mean(r ** 2 / (2 * np.exp(2 * log_sigma)) + log_sigma + 0.5 * np.log(2 * np.pi)))
    # @end


def fit_hetero(x, y, steps=4000, lr=0.05, trace=None):
    """Fit μ(x) = w₀ + w₁u and log σ(x) = v₀ + v₁u (u = x / 10) by gradient descent on the mean
    Gaussian NLL. Returns (w, v) as arrays, in terms of u."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    u = x / 10
    w = np.polyfit(u, y, 1)[::-1].copy()  # start from least squares …
    v = np.array([np.log(np.std(y - w[0] - w[1] * u)), 0.0])  # … with a constant σ
    for t in range(steps):
        mu = w[0] + w[1] * u
        s = v[0] + v[1] * u
        # @blank gd
        # zh: r = y − μ；∂ℓ/∂μ = −r/σ²，∂ℓ/∂s = 1 − r²/σ²；对 (1, u) 求平均得到梯度。v 走一步 −lr·梯度；w 的梯度很小（含 1/σ²），先乘以 e^{2v₀} 作预条件再走一步
        # en: r = y − μ; ∂ℓ/∂μ = −r/σ², ∂ℓ/∂s = 1 − r²/σ²; average against (1, u) for the gradients. Step v by −lr·gradient; w's gradient is tiny (it carries 1/σ²), so precondition it by e^{2v₀} before stepping
        r = y - mu
        g_mu = -r * np.exp(-2 * s)
        g_s = 1 - r ** 2 * np.exp(-2 * s)
        w = w - lr * np.array([g_mu.mean(), (g_mu * u).mean()]) * np.exp(2 * v[0])
        v = v - lr * np.array([g_s.mean(), (g_s * u).mean()])
        # @end
        if trace is not None:
            trace(t, w, v)
    return w, v
