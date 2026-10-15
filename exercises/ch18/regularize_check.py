"""Checks for regularize.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.datasets import make_moons

from learnml import Grid, Trace


def check(s, t):
    def t_dropout():
        rng = np.random.default_rng(0)
        h = rng.uniform(0.5, 1.5, size=(400, 300))
        out, mask = s.dropout(h, 0.3, rng)
        zero = np.mean(out == 0)
        assert abs(zero - 0.3) < 0.01, f"about 30% of the entries should be dropped (got {zero:.3f})"
        kept = out[out != 0] / h[out != 0]
        assert np.allclose(kept, 1 / 0.7), "survivors should be divided by 1 − p = 0.7"
        assert abs(out.mean() / h.mean() - 1) < 0.01, f"inverted dropout keeps the mean (ratio {out.mean() / h.mean():.3f})"
        assert mask.shape == h.shape, "the mask should have the shape of h"
        same, none = s.dropout(h, 0.3, rng, train=False)
        assert same is h and none is None, "at test time dropout should return h unchanged"

    def t_decay():
        W, b = np.array([[1.0, -2.0]]), np.array([0.5])
        dW, db = np.array([[0.1, 0.2]]), np.array([0.3])
        (W2, b2), = s.sgd_step([(W, b)], [(dW, db)], lr=0.1, wd=0.5)
        assert np.allclose(W2, [[0.94, -1.92]]), f"W ← (1 − 0.05)·W − 0.1·dW = [[0.94, −1.92]], got {W2}"
        assert np.allclose(b2, [0.47]), f"the bias is not decayed: 0.5 − 0.1·0.3 = 0.47, got {b2}"

    def t_grad():
        rng = np.random.default_rng(2)
        # Nonzero biases: a row whose inputs were all dropped would otherwise sit exactly on a ReLU kink.
        params = [(W, rng.normal(0, 0.3, b.shape)) for W, b in s.init([2, 6, 5, 1], rng)]
        X = rng.normal(size=(20, 2))
        y = (rng.uniform(size=20) < 0.5).astype(float)
        _, cache = s.forward(params, X, 0.4, rng, True)
        masks = cache[2]
        assert any(m is not None and np.any(m == 0) for m in masks), "with p = 0.4 some units should be dropped"
        grads = s.backward(params, X, y, cache)
        h, worst = 1e-6, 0.0
        for l, (W, b) in enumerate(params):
            for arr, g in [(W, grads[l][0]), (b, grads[l][1])]:
                for idx in np.ndindex(arr.shape):
                    old = arr[idx]
                    arr[idx] = old + h; up = s.loss(params, X, y, masks=masks)
                    arr[idx] = old - h; dn = s.loss(params, X, y, masks=masks)
                    arr[idx] = old
                    worst = max(worst, abs((up - dn) / (2 * h) - g[idx]))
        assert worst < 1e-7, f"with the dropout masks held fixed, backprop should match central differences (worst gap {worst:.2e})"

    def t_early():
        vals = iter([5.0, 4.0, 3.0, 3.5, 2.5, 2.6, 2.7, 2.8, 2.9, 1.0])
        state = {"k": 0}

        def step():
            state["k"] += 1
            return [(np.array([[float(state["k"])]]), np.zeros(1))]

        best, epoch, ran = s.early_stopping(step, lambda q: next(vals), max_epochs=10, patience=3)
        assert epoch == 5, f"the lowest validation loss before stopping is at epoch 5 (got {epoch})"
        assert ran == 8, f"three epochs without improvement after epoch 5: stop at epoch 8 (got {ran})"
        assert best[0][0][0, 0] == 5.0, "return a copy of the parameters from the best epoch"

    def data():
        X, y = make_moons(80, noise=0.3, random_state=0)
        Xv, yv = make_moons(200, noise=0.3, random_state=10)
        return X, y.astype(float), Xv, yv.astype(float)

    def t_train():
        X, y, Xv, yv = data()
        Xt, yt = make_moons(1000, noise=0.3, random_state=20)
        norm = lambda P: np.sqrt(sum((W ** 2).sum() for W, _ in P))
        plain, _, _ = s.train(X, y, Xv, yv, [2, 32, 32, 1], max_epochs=200)
        reg, best, ran = s.train(X, y, Xv, yv, [2, 32, 32, 1], p=0.3, wd=0.01, max_epochs=200, patience=30)
        assert norm(reg) < 0.9 * norm(plain), f"weight decay should shrink the weights (norm {norm(reg):.2f} vs {norm(plain):.2f} without)"
        assert best < ran, f"early stopping should return an earlier epoch than the last one run ({best} vs {ran})"
        acc = np.mean((s.forward(reg, Xt, 0, None, False)[0] > 0) == (yt == 1))
        assert acc > 0.8, f"the regularised network should still fit the moons (test accuracy {acc:.3f})"

    t.test("Inverted Dropout", "Inverted dropout", t_dropout)
    t.test("解耦的权重衰减", "Decoupled weight decay", t_decay)
    t.test("穿过 Dropout 的反向传播", "Backprop through dropout", t_grad)
    t.test("早停", "Early stopping", t_early)
    t.test("带正则训练", "Training with regularisation", t_train)

    def trace():
        X, y, Xv, yv = data()
        grid = Grid(X, n=48)
        tr = Trace("classifier2d", X, np.where(y > 0, 1, -1), grid=grid, show_margin=False,
                   title={"zh": "Dropout + 权重衰减 + 早停：分界线怎样长出来", "en": "Dropout + weight decay + early stopping: how the boundary grows"})
        keep = {1, 5, 15, 40, 80, 120, 200}

        def rec(e, params):
            if e in keep:
                f = s.forward(params, grid.points, 0, None, False)[0]
                tr.record(field=f, metrics={"val": s.loss(params, Xv, yv)}, note=f"epoch {e}")

        s.train(X, y, Xv, yv, [2, 32, 32, 1], p=0.3, wd=0.01, max_epochs=200, trace=rec)
        return tr

    t.trace(trace)
