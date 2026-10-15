"""Automatic differentiation with the autograd library: the gradient of a network's loss in one call,
and a Jacobian built column by column (forward mode, JVPs) or row by row (reverse mode, VJPs)."""

import autograd.numpy as anp
import numpy as np
from autograd import grad, make_jvp, make_vjp


def mlp_loss(params, X, y):
    """Mean BCE of a tanh MLP; params is a list of (W, b) with W of shape (n_out, n_in)."""
    h = X
    for l, (W, b) in enumerate(params):
        z = anp.dot(h, W.T) + b
        h = z if l == len(params) - 1 else anp.tanh(z)
    f = h[:, 0]
    return anp.mean(anp.logaddexp(0, f) - y * f)


def loss_grad(params, X, y):
    """Gradient of mlp_loss with respect to params (same nested structure)."""
    # @blank grad
    # zh: grad(f) 默认对第一个参数求导，返回同样结构的梯度
    # en: grad(f) differentiates in the first argument and returns gradients of the same structure
    return grad(mlp_loss)(params, X, y)
    # @end


def jacobian_by_columns(f, x):
    """J (m × n) of f at x from n JVPs: make_jvp(f)(x) returns a function v ↦ (f(x), J v)."""
    # @blank columns
    # zh: 对每个单位向量 e_j 求 J e_j，按列拼起来
    # en: Compute J e_j for every unit vector e_j and stack them as columns
    jvp = make_jvp(f)(x)
    return np.column_stack([jvp(e)[1] for e in np.eye(len(x))])
    # @end


def jacobian_by_rows(f, x):
    """J (m × n) of f at x from m VJPs: make_vjp(f)(x) returns (u ↦ uᵀ J, f(x))."""
    # @blank rows
    # zh: 先拿到 vjp 函数和 f(x)，再对每个输出方向 e_i 求一行
    # en: Get the vjp function and f(x), then compute one row per output direction e_i
    vjp, fx = make_vjp(f)(x)
    return np.vstack([vjp(e) for e in np.eye(len(fx))])
    # @end


def train(X, y, sizes, epochs=200, lr=0.5, seed=0, trace=None):
    """Full-batch gradient descent with loss_grad, from Xavier initialisation."""
    rng = np.random.default_rng(seed)
    params = [(rng.normal(0, np.sqrt(2 / (a + b)), (b, a)), np.zeros(b)) for a, b in zip(sizes[:-1], sizes[1:])]
    for e in range(epochs):
        g = loss_grad(params, X, y)
        params = [(W - lr * dW, b - lr * db) for (W, b), (dW, db) in zip(params, g)]
        if trace is not None:
            trace(e, params)
    return params
