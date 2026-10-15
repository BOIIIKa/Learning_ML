"""Checks for autodiff_library.py. Runs identically under CPython and Pyodide."""

import autograd.numpy as anp
import numpy as np
from autograd import jacobian
from sklearn.datasets import make_moons

from learnml import Grid, Trace


def check(s, t):
    def t_grad():
        rng = np.random.default_rng(0)
        params = [(rng.normal(size=(4, 2)), rng.normal(size=4)), (rng.normal(size=(1, 4)), rng.normal(size=1))]
        X = rng.normal(size=(25, 2))
        y = (rng.uniform(size=25) < 0.5).astype(float)
        g = s.loss_grad(params, X, y)
        assert len(g) == 2 and g[0][0].shape == (4, 2), "the gradient should have the same structure as params"
        h = 1e-6
        W = params[0][0]
        old = W[2, 1]
        W[2, 1] = old + h; up = s.mlp_loss(params, X, y)
        W[2, 1] = old - h; dn = s.mlp_loss(params, X, y)
        W[2, 1] = old
        num = (up - dn) / (2 * h)
        assert abs(g[0][0][2, 1] - num) < 1e-8, f"∂L/∂W₁[2,1] = {g[0][0][2, 1]:.8f}, central difference {num:.8f}"

    def f(x):
        A = np.array([[0.5, -1.0, 0.3, 0.8], [1.2, 0.4, -0.7, 0.1], [-0.3, 0.9, 0.6, -1.1]])
        return anp.tanh(anp.dot(A, x)) * (1 + x[:3] ** 2)

    def t_modes():
        x = np.array([0.3, -0.7, 1.1, 0.2])
        Jc = s.jacobian_by_columns(f, x)
        Jr = s.jacobian_by_rows(f, x)
        ref = jacobian(f)(x)
        assert Jc.shape == (3, 4) and Jr.shape == (3, 4), f"J should be 3 × 4 (got {Jc.shape} and {Jr.shape})"
        assert np.allclose(Jc, ref, atol=1e-12), "forward mode (JVPs, one column each) should give autograd's Jacobian"
        assert np.allclose(Jr, ref, atol=1e-12), "reverse mode (VJPs, one row each) should give autograd's Jacobian"

    def t_train():
        X, y = make_moons(200, noise=0.15, random_state=0)
        params = s.train(X, y.astype(float), [2, 16, 16, 1])
        h = X
        for l, (W, b) in enumerate(params):
            h = h @ W.T + b if l == len(params) - 1 else np.tanh(h @ W.T + b)
        acc = np.mean((h[:, 0] > 0) == (y == 1))
        assert acc > 0.95, f"training with autograd's gradients should fit the moons (accuracy {acc:.3f})"

    t.test("一行求出网络的梯度", "A network's gradient in one call", t_grad)
    t.test("前向模式与反向模式", "Forward mode and reverse mode", t_modes)
    t.test("用 autograd 训练", "Training with autograd", t_train)

    def trace():
        X, y = make_moons(200, noise=0.15, random_state=0)
        grid = Grid(X, n=48)
        tr = Trace("classifier2d", X, np.where(y > 0, 1, -1), grid=grid, show_margin=False,
                   title={"zh": "autograd 求梯度训练的 tanh 网络", "en": "A tanh network trained with autograd's gradients"})
        keep = {0, 10, 30, 80, 199}

        def rec(e, params):
            if e in keep:
                h = grid.points
                for l, (W, b) in enumerate(params):
                    h = h @ W.T + b if l == len(params) - 1 else np.tanh(h @ W.T + b)
                tr.record(field=h[:, 0], metrics={"loss": float(s.mlp_loss(params, X, y.astype(float)))}, note=f"epoch {e + 1}")

        s.train(X, y.astype(float), [2, 16, 16, 1], trace=rec)
        return tr

    t.trace(trace)
