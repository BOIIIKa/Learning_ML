"""Checks for backprop.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.datasets import make_moons

from learnml import Grid, Trace


def check(s, t):
    def t_init():
        rng = np.random.default_rng(0)
        (W, _), = s.init_params([400, 300], "he", rng)
        assert W.shape == (300, 400), f"W should be (n_out, n_in) = (300, 400), got {W.shape}"
        assert abs(W.var() / (2 / 400) - 1) < 0.03, f"He: Var(W) = {W.var():.5f}, expected 2/400 = 0.005"
        (V, _), = s.init_params([400, 300], "xavier", rng)
        assert abs(V.var() / (2 / 700) - 1) < 0.03, f"Xavier: Var(W) = {V.var():.5f}, expected 2/700 = {2 / 700:.5f}"

    def t_forward():
        rng = np.random.default_rng(1)
        params = s.init_params([200] * 21, "he", rng)
        X = rng.normal(size=(64, 200))
        zs, hs = s.forward(params, X)
        assert len(zs) == 20 and len(hs) == 21, "one z per layer, and hs[0] = X"
        ratio = zs[-1].std() / zs[0].std()
        assert 0.3 < ratio < 3, f"with He init the pre-activation scale should stay level over 20 ReLU layers (ratio {ratio:.2f})"
        assert np.all(hs[5] >= 0), "hidden activations should be ReLU outputs"

    def t_grad():
        rng = np.random.default_rng(2)
        params = s.init_params([2, 5, 4, 1], "he", rng)
        X = rng.normal(size=(30, 2))
        y = (rng.uniform(size=30) < 0.5).astype(float)
        grads = s.backward(params, X, y)
        h = 1e-6
        worst = 0.0
        for l, (W, b) in enumerate(params):
            for arr, g in [(W, grads[l][0]), (b, grads[l][1])]:
                assert g.shape == arr.shape, f"layer {l}: gradient shape {g.shape} should match {arr.shape}"
                for idx in np.ndindex(arr.shape):
                    old = arr[idx]
                    arr[idx] = old + h; up = s.loss(params, X, y)
                    arr[idx] = old - h; dn = s.loss(params, X, y)
                    arr[idx] = old
                    worst = max(worst, abs((up - dn) / (2 * h) - g[idx]))
        assert worst < 1e-7, f"backprop should match central differences (worst gap {worst:.2e})"

    def t_train():
        X, y = make_moons(200, noise=0.15, random_state=0)
        params = s.train(X, y.astype(float), [2, 16, 16, 16, 1])
        acc = np.mean((s.forward(params, X)[1][-1][:, 0] > 0) == (y == 1))
        assert acc > 0.95, f"a 3-hidden-layer network with He init should fit the moons (accuracy {acc:.3f})"

    t.test("He 与 Xavier 初始化", "He and Xavier initialisation", t_init)
    t.test("前向传播与方差保持", "Forward pass and variance preservation", t_forward)
    t.test("反向传播 = 有限差分", "Backprop matches finite differences", t_grad)
    t.test("训练一个三层网络", "Training a three-layer network", t_train)

    def trace():
        X, y = make_moons(200, noise=0.15, random_state=0)
        grid = Grid(X, n=48)
        tr = Trace("classifier2d", X, np.where(y > 0, 1, -1), grid=grid, show_margin=False,
                   title={"zh": "你写的反向传播：分界线一轮轮弯过去", "en": "Your backprop at work: the boundary bends epoch by epoch"})
        keep = {0, 10, 30, 80, 150, 299}

        def rec(e, params):
            if e in keep:
                f = s.forward(params, grid.points)[1][-1][:, 0]
                tr.record(field=f, metrics={"loss": s.loss(params, X, y.astype(float))}, note=f"epoch {e + 1}")

        s.train(X, y.astype(float), [2, 16, 16, 16, 1], trace=rec)
        return tr

    t.trace(trace)
