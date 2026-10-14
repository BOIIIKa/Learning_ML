"""Checks for nn_library.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from learnml import Grid, Trace


def check(s, t):
    X, y = make_moons(400, noise=0.25, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    state = {}

    def t_fit():
        pipe = s.make_mlp((16, 16)).fit(Xtr, ytr)
        acc = pipe.score(Xte, yte)
        assert acc > 0.9, f"test accuracy {acc:.3f} should exceed 0.9 on the moons"
        state["pipe"] = pipe

    def t_params():
        pipe = state.get("pipe")
        assert pipe is not None, "fix the previous test first"
        n = s.n_params(pipe[-1])
        assert n == 3 * 16 + 17 * 16 + 17 * 1, f"{n} parameters; (2+1)·16 + (16+1)·16 + (16+1)·1 = 337"

    def t_regions():
        grid = Grid(X, n=60).points
        shallow = s.make_mlp((8,)).fit(Xtr, ytr)
        r1 = s.regions_used(shallow, grid)
        assert 2 <= r1 <= 1 + 8 + 28, f"a single layer of 8 units on 2 inputs makes at most 1 + 8 + C(8,2) = 37 regions, got {r1}"
        r2 = s.regions_used(state["pipe"], grid)
        assert r2 > 1, "the two-layer net should use more than one linear region"
        state["regions"] = (r1, r2)

    t.test("在月牙数据上训练 MLP", "Train an MLP on the moons", t_fit)
    t.test("参数个数", "Parameter count", t_params)
    t.test("实际用到的线性区域", "Linear regions actually used", t_regions)

    def trace():
        sc = StandardScaler().fit(Xtr)
        Z = sc.transform(Xtr)
        grid = Grid(X, n=48)
        G = sc.transform(grid.points)
        y2 = 2 * ytr - 1
        tr = Trace("classifier2d", Xtr, y2, grid=grid, show_margin=False,
                   title={"zh": "训练中的 ReLU 网络：边界由一段段直线拼成", "en": "A ReLU net in training: the boundary is made of straight pieces"})
        mlp = MLPClassifier(hidden_layer_sizes=(16, 16), activation="relu", learning_rate_init=0.01, random_state=0)
        done = 0
        for epochs in (1, 5, 20, 60, 150, 300):
            while done < epochs:
                mlp.partial_fit(Z, ytr, classes=[0, 1])
                done += 1
            field = 2 * mlp.predict_proba(G)[:, 1] - 1
            tr.record(field=field, metrics={"acc": float(mlp.score(Z, ytr))}, note=f"epoch {epochs}")
        return tr

    t.trace(trace)
