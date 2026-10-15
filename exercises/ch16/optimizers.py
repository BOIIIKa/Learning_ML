"""First-order optimizers from scratch: heavy-ball momentum, and Adam with bias correction and
AdamW's decoupled weight decay. Each function returns the whole path of iterates."""

import numpy as np


def momentum_path(grad, x0, lr, beta, steps):
    """Heavy ball: v ← βv − η∇f(x), x ← x + v, starting from v = 0. Returns an array (steps + 1, d)."""
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    path = [x.copy()]
    for _ in range(steps):
        # @blank momentum
        # zh: 先更新速度 v，再用新的 v 移动 x
        # en: Update the velocity v first, then move x by the new v
        v = beta * v - lr * grad(x)
        x = x + v
        # @end
        path.append(x.copy())
    return np.array(path)


def adam_path(grad, x0, lr, steps, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, trace=None):
    """Adam (wd = 0) or AdamW (wd > 0, decay decoupled from the adaptive ratio). Returns (steps + 1, d)."""
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    path = [x.copy()]
    for t in range(1, steps + 1):
        g = grad(x)
        # @blank moments
        # zh: m、v 分别是 g 和 g² 的指数滑动平均（权重 b1、b2）
        # en: m and v are exponential moving averages of g and g² (weights b1, b2)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        # @end
        # @blank correct
        # zh: 偏差修正：除以 1 − b1^t 和 1 − b2^t（t 从 1 开始）
        # en: Bias correction: divide by 1 − b1^t and 1 − b2^t (t starts at 1)
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        # @end
        # @blank update
        # zh: x ← x − lr·(m̂/(√v̂ + eps) + wd·x)：衰减项在比值之外
        # en: x ← x − lr·(m̂/(√v̂ + eps) + wd·x): the decay sits outside the ratio
        x = x - lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * x)
        # @end
        path.append(x.copy())
        if trace is not None:
            trace(t, x)
    return np.array(path)


def logistic_grad(X, y):
    """Gradient of the mean logistic loss in θ = (w₁, w₂, b) for labels y ∈ {0, 1}."""
    A = np.column_stack([X, np.ones(len(X))])

    def grad(theta):
        p = 1 / (1 + np.exp(-A @ theta))
        return A.T @ (p - y) / len(y)

    return grad
