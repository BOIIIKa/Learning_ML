"""Backpropagation for a ReLU network from scratch, with He and Xavier initialisation."""

import numpy as np


def init_params(sizes, scheme, rng):
    """[(W, b), …] for layer sizes [d, h1, …, 1]: W of shape (n_out, n_in) drawn from N(0, var), b = 0,
    with var = 2/n_in ("he") or 2/(n_in + n_out) ("xavier")."""
    params = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        # @blank init
        # zh: 按 scheme 选方差，再用 rng.normal(0, np.sqrt(var), (n_out, n_in)) 采样
        # en: Pick the variance by scheme, then sample rng.normal(0, np.sqrt(var), (n_out, n_in))
        var = 2 / n_in if scheme == "he" else 2 / (n_in + n_out)
        W = rng.normal(0, np.sqrt(var), (n_out, n_in))
        # @end
        params.append((W, np.zeros(n_out)))
    return params


def forward(params, X):
    """Forward pass for a batch X (N, d): ReLU hidden layers, linear output.
    Returns (zs, hs): pre-activations of every layer, and activations with hs[0] = X."""
    hs, zs = [X], []
    for l, (W, b) in enumerate(params):
        # @blank forward
        # zh: z = h W^T + b；最后一层不过 ReLU
        # en: z = h W^T + b; the last layer has no ReLU
        z = hs[-1] @ W.T + b
        zs.append(z)
        hs.append(z if l == len(params) - 1 else np.maximum(z, 0))
        # @end
    return zs, hs


def loss(params, X, y):
    """Mean binary cross-entropy with logits, labels y ∈ {0, 1}."""
    f = forward(params, X)[1][-1][:, 0]
    return float(np.mean(np.logaddexp(0, f) - y * f))


def backward(params, X, y):
    """Gradients [(dW, db), …] of the mean BCE, by backpropagation."""
    zs, hs = forward(params, X)
    f = hs[-1][:, 0]
    # @blank delta-out
    # zh: 输出层的误差信号 δ = (σ(f) − y)/N，形状 (N, 1)
    # en: The output error signal δ = (σ(f) − y)/N, shape (N, 1)
    delta = ((1 / (1 + np.exp(-f)) - y) / len(y))[:, None]
    # @end
    grads = [None] * len(params)
    for l in range(len(params) - 1, -1, -1):
        W, _ = params[l]
        # @blank backward
        # zh: dW = δᵀ h_{l−1}，db = δ 按行求和；再把 δ 传回上一层：(δ W) ⊙ [z_{l−1} > 0]
        # en: dW = δᵀ h_{l−1}, db = δ summed over rows; then send δ back: (δ W) ⊙ [z_{l−1} > 0]
        grads[l] = (delta.T @ hs[l], delta.sum(axis=0))
        if l > 0:
            delta = (delta @ W) * (zs[l - 1] > 0)
        # @end
    return grads


def train(X, y, sizes, epochs=300, lr=0.5, seed=0, trace=None):
    """Full-batch gradient descent from He initialisation."""
    params = init_params(sizes, "he", np.random.default_rng(seed))
    for e in range(epochs):
        grads = backward(params, X, y)
        params = [(W - lr * dW, b - lr * db) for (W, b), (dW, db) in zip(params, grads)]
        if trace is not None:
            trace(e, params)
    return params
