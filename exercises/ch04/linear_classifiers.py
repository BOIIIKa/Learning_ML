"""The perceptron and logistic regression by Newton's method (IRLS), written from scratch.

Perceptron: labels y ∈ {−1, +1}; on every mistake y(w·x + b) ≤ 0 do w ← w + η y x, b ← b + η y.
Logistic regression: labels t ∈ {0, 1}; θ = (w, b) maximizes the Bernoulli log-likelihood, and each
Newton step solves H Δθ = −∇ with ∇ = X̂ᵀ(p − t), H = X̂ᵀ diag(p(1 − p)) X̂, where X̂ = [X, 1].
"""

import numpy as np


def is_mistake(w, b, x, y):
    """True when (x, y) lies on the wrong side of the line or on it: y (w·x + b) ≤ 0."""
    # @blank mistake
    # zh: 函数间隔 y(w·x + b) 不大于 0 就算误分类（落在边界上也算）
    # en: A functional margin y(w·x + b) of at most 0 is a mistake (lying on the boundary counts)
    return y * (x @ w + b) <= 0
    # @end


def perceptron_update(w, b, x, y, eta):
    """One stochastic gradient step on the perceptron loss −y(w·x + b) at a misclassified point."""
    # @blank update
    # zh: 返回新的 (w, b)：w + η y x 与 b + η y
    # en: Return the new (w, b): w + η y x and b + η y
    return w + eta * y * x, b + eta * y
    # @end


def perceptron(X, y, eta=1.0, max_epochs=100, trace=None):
    """Visit the points in order, update on every mistake; stop after a pass with no mistakes."""
    w, b = np.zeros(X.shape[1]), 0.0
    if trace is not None:
        trace.record(w=w, b=b, metrics={"mistakes": len(y)}, note="w = 0")
    for _ in range(max_epochs):
        clean = True
        for i in range(len(y)):
            if is_mistake(w, b, X[i], y[i]):
                w, b = perceptron_update(w, b, X[i], y[i], eta)
                clean = False
                if trace is not None:
                    wrong = int(np.sum(y * (X @ w + b) <= 0))
                    trace.record(w=w, b=b, highlight=[i], metrics={"mistakes": wrong}, note=f"x{i + 1}, y = {int(y[i]):+d}")
        if clean:
            break
    return w, b


def sigmoid(z):
    """σ(z) = 1 / (1 + e^{−z}), without overflow for large |z|."""
    # @blank sigmoid
    # zh: 用 e^{−|z|} 避免溢出：z ≥ 0 时 1/(1+e^{−z})，z < 0 时 e^{z}/(1+e^{z})
    # en: Use e^{−|z|} to avoid overflow: 1/(1+e^{−z}) for z ≥ 0 and e^{z}/(1+e^{z}) for z < 0
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    # @end


def nll_grad_hess(theta, Xh, t):
    """Negative log-likelihood, its gradient and its Hessian at θ (Xh already has the column of ones)."""
    z = Xh @ theta
    nll = float(np.sum(np.logaddexp(0.0, z) - t * z))
    # @blank grad-hess
    # zh: p = σ(X̂θ)；梯度 X̂ᵀ(p − t)；Hessian X̂ᵀ diag(p(1 − p)) X̂
    # en: p = σ(X̂θ); gradient X̂ᵀ(p − t); Hessian X̂ᵀ diag(p(1 − p)) X̂
    p = sigmoid(z)
    grad = Xh.T @ (p - t)
    hess = Xh.T @ (Xh * (p * (1 - p))[:, None])
    # @end
    return nll, grad, hess


def newton_step(theta, Xh, t):
    """θ ← θ − H⁻¹ ∇ (solve the linear system instead of inverting H)."""
    _, grad, hess = nll_grad_hess(theta, Xh, t)
    # @blank newton
    # zh: 用 np.linalg.solve(hess, grad) 求出 Newton 方向，再从 θ 中减去
    # en: Get the Newton direction with np.linalg.solve(hess, grad) and subtract it from θ
    return theta - np.linalg.solve(hess, grad)
    # @end


def logistic_irls(X, t, iters=8, trace=None, grid=None):
    """Newton's method from θ = 0. Returns (w, b)."""
    Xh = np.column_stack([X, np.ones(len(t))])
    theta = np.zeros(Xh.shape[1])
    for k in range(iters + 1):
        if trace is not None:
            nll, grad, _ = nll_grad_hess(theta, Xh, t)
            field = None if grid is None else grid.points @ theta[:-1] + theta[-1]
            trace.record(w=theta[:-1], b=theta[-1], field=field,
                         metrics={"nll": nll / len(t)}, note=f"k = {k}, ‖∇‖ = {np.linalg.norm(grad):.1e}")
        if k < iters:
            theta = newton_step(theta, Xh, t)
    return theta[:-1], theta[-1]
