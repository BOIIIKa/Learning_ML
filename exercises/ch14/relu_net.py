"""ReLU networks from scratch: the forward pass, activation patterns, a deep "tent" network that
folds [0, 1] L times, and a shallow network built to match any broken line.

Weights follow scikit-learn's convention: Ws[l] has shape (inputs, outputs), bs[l] shape (outputs,).
Hidden layers use ReLU; the last layer is linear.
"""

import numpy as np


def forward(Ws, bs, X):
    """Network output for the rows of X, shape (n, outputs)."""
    H = np.asarray(X, dtype=float)
    # @blank forward
    # zh: 每层 H ← H @ W + b；除最后一层外再取 max(0, ·)
    # en: Each layer H ← H @ W + b; apply max(0, ·) on every layer except the last
    for l, (W, b) in enumerate(zip(Ws, bs)):
        H = H @ W + b
        if l < len(Ws) - 1:
            H = np.maximum(0, H)
    # @end
    return H


def patterns(Ws, bs, X):
    """On/off pattern of every hidden unit for each row of X: a boolean array (n, total hidden units)."""
    H = np.asarray(X, dtype=float)
    on = []
    # @blank patterns
    # zh: 逐个隐层计算 Z = H @ W + b，记录 Z > 0，再令 H = max(0, Z)；最后按列拼接
    # en: For each hidden layer compute Z = H @ W + b, record Z > 0, then set H = max(0, Z); stack the columns
    for W, b in zip(Ws[:-1], bs[:-1]):
        Z = H @ W + b
        on.append(Z > 0)
        H = np.maximum(0, Z)
    # @end
    return np.hstack(on)


def tent_net(L):
    """Weights of an L-hidden-layer net (2 ReLUs per layer) computing t∘…∘t, t(x) = 2 relu(x) − 4 relu(x − ½)."""
    # @blank tent
    # zh: 第一层把 x 送进两个单元 relu(x)、relu(x − ½)；之后每层先算 t = 2h₁ − 4h₂ 再送进同样两个单元；输出层权重 (2, −4)
    # en: Layer 1 feeds x to relu(x) and relu(x − ½); every later layer forms t = 2h₁ − 4h₂ and feeds it to the same two units; the output weights are (2, −4)
    Ws = [np.array([[1.0, 1.0]])] + [np.array([[2.0, 2.0], [-4.0, -4.0]]) for _ in range(L - 1)] + [np.array([[2.0], [-4.0]])]
    bs = [np.array([0.0, -0.5]) for _ in range(L)] + [np.array([0.0])]
    return Ws, bs
    # @end


def hinge_net(a, fa, knots, slopes):
    """A one-hidden-layer net equal on [a, ∞) to the broken line with value fa at a, kinks at `knots`
    and slopes `slopes` (one more slope than knots): units relu(x − a), relu(x − k₁), …"""
    knots = np.asarray(knots, dtype=float)
    slopes = np.asarray(slopes, dtype=float)
    # @blank hinge
    # zh: 隐层：权重全为 1，偏置 −a, −k₁, …；输出权重：s₀ 和各拐点处的斜率改变量 s_d − s_{d−1}；输出偏置 fa
    # en: Hidden layer: all weights 1, biases −a, −k₁, …; output weights: s₀ and the slope changes s_d − s_{d−1}; output bias fa
    centres = np.concatenate([[a], knots])
    W1 = np.ones((1, len(centres)))
    b1 = -centres
    W2 = np.concatenate([[slopes[0]], np.diff(slopes)])[:, None]
    b2 = np.array([fa])
    return [W1, W2], [b1, b2]
    # @end
