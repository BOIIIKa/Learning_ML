"""Checks for relu_net.py. Runs identically under CPython and Pyodide."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

from learnml import Grid, Trace


def tent_pow(L, x):
    for _ in range(L):
        x = 2 * np.maximum(0, x) - 4 * np.maximum(0, x - 0.5)
    return x


def check(s, t):
    def t_forward():
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m = MLPRegressor(hidden_layer_sizes=(7, 5), activation="relu", max_iter=30, random_state=0).fit(X, X @ [1.0, -2.0, 0.5])
        out = s.forward(m.coefs_, m.intercepts_, X)
        assert out.shape == (60, 1), f"output shape {out.shape}, expected (60, 1)"
        assert np.allclose(out[:, 0], m.predict(X), atol=1e-10), "same weights must give scikit-learn's predictions"

    def t_tent():
        x = (np.arange(4096) + 0.5) / 4096
        for L in (1, 2, 3, 4, 5):
            Ws, bs = s.tent_net(L)
            y = s.forward(Ws, bs, x[:, None])[:, 0]
            assert np.allclose(y, tent_pow(L, x), atol=1e-12), f"L = {L}: the net should compute t applied {L} times"
            n = len({tuple(r) for r in s.patterns(Ws, bs, x[:, None])})
            assert n == 2 ** L, f"L = {L}: {n} linear regions along [0, 1]; folding should give 2^L = {2 ** L}"
        assert sum(W.size + b.size for W, b in zip(*s.tent_net(5))) == 31, "6L + 1 = 31 parameters for L = 5"

    def t_hinge():
        Ws, bs = s.hinge_net(0.0, 1.0, [1.0, 2.0], [2.0, -1.0, 1.0])
        x = np.array([[0.0], [1.0], [2.0], [3.0], [0.5], [2.5]])
        assert np.allclose(s.forward(Ws, bs, x)[:, 0], [1, 3, 2, 3, 2, 2.5]), "the worked example: values 1, 3, 2, 3 at x = 0, 1, 2, 3"
        rng = np.random.default_rng(1)
        knots = np.sort(rng.uniform(0, 10, 6))
        slopes = rng.normal(size=7)
        xs = np.linspace(0, 10, 501)
        # Reference broken line: integrate the slopes piece by piece.
        edges = np.concatenate([[0.0], knots, [10.0]])
        ref = np.interp(xs, edges, 0.3 + np.concatenate([[0.0], np.cumsum(slopes * np.diff(edges))]))
        out = s.forward(*s.hinge_net(0.0, 0.3, knots, slopes), xs[:, None])[:, 0]
        assert np.allclose(out, ref, atol=1e-9), "the hinge net should equal the broken line on [a, ∞)"

    t.test("前向传播与 scikit-learn 一致", "Forward pass matches scikit-learn", t_forward)
    t.test("帐篷网络：L 次折叠，2^L 个区域", "Tent net: L folds, 2^L regions", t_tent)
    t.test("用铰链拼出任意折线", "Hinges build any broken line", t_hinge)

    def trace():
        rng = np.random.default_rng(3)
        X = rng.uniform(-2, 2, size=(160, 2))
        u = (X[:, 0] + 2) / 4
        y = np.where(tent_pow(3, u) + 0.25 * X[:, 1] > 0.5, 1, -1)
        grid = Grid(X, n=48, pad=0.0)
        gu = (grid.points[:, 0] + 2) / 4
        tr = Trace("classifier2d", X, y, grid=grid, show_margin=False,
                   title={"zh": "每加一层，决策区域对折一次", "en": "Every extra layer folds the regions once more"})
        line = (np.arange(2048) + 0.5) / 2048
        for L in (1, 2, 3, 4, 5):
            Ws, bs = s.tent_net(L)
            field = s.forward(Ws, bs, gu[:, None])[:, 0] + 0.25 * grid.points[:, 1] - 0.5
            n = len({tuple(r) for r in s.patterns(Ws, bs, line[:, None])})
            tr.record(field=field, metrics={"regions": n}, note=f"L = {L}")
        return tr

    t.trace(trace)
