"""Checks for optim_library.py. Runs identically under CPython and Pyodide."""

import warnings

import numpy as np
from sklearn.datasets import make_moons

from learnml import Grid, Trace


def check(s, t):
    state = {}

    def t_scipy():
        res = s.scipy_race(np.array([-1.2, 1.0]))
        assert set(res) == set(s.METHODS), f"expected one result per method, got {sorted(res)}"
        for m, r in res.items():
            assert r.fun < 1e-8, f"{m} should reach the minimum (f = {r.fun:.2e})"
            assert np.allclose(r.x, [1, 1], atol=1e-3), f"{m} should end at (1, 1), got {r.x}"
        assert res["BFGS"].njev < res["CG"].njev, f"BFGS ({res['BFGS'].njev} gradients) should need fewer gradient evaluations than CG ({res['CG'].njev})"
        assert getattr(res["Newton-CG"], "nhev", 0) > 0, "Newton-CG should have been given the Hessian (hess=rosen_hess)"

    def t_mlp():
        m = s.make_mlp("nesterov")
        assert m.solver == "sgd" and m.momentum == 0.9 and m.nesterovs_momentum, "the nesterov MLP: solver='sgd', momentum=0.9, nesterovs_momentum=True"
        assert s.make_mlp("sgd").momentum == 0, "plain SGD needs momentum=0"
        assert s.make_mlp("adam").solver == "adam", "the adam MLP needs solver='adam'"

    def t_curves():
        X, y = make_moons(200, noise=0.2, random_state=0)
        c = s.loss_curves(X, y)
        state["c"] = c
        assert all(len(v) == 120 for v in c.values()), "each run should train for all 120 epochs"
        assert c["nesterov"][-1] < 0.7 * c["sgd"][-1], f"Nesterov's final loss {c['nesterov'][-1]:.3f} should beat plain SGD's {c['sgd'][-1]:.3f}"
        assert c["adam"][-1] < 0.7 * c["sgd"][-1], f"Adam's final loss {c['adam'][-1]:.3f} should beat plain SGD's {c['sgd'][-1]:.3f}"

    t.test("SciPy：拟牛顿与牛顿", "SciPy: quasi-Newton and Newton", t_scipy)
    t.test("三种 MLP 优化器的设置", "Setting up the three MLP optimizers", t_mlp)
    t.test("训练曲线", "Training curves", t_curves)

    def trace():
        X, y = make_moons(200, noise=0.2, random_state=0)
        grid = Grid(X, n=48)
        tr = Trace("classifier2d", X, np.where(y > 0, 1, -1), grid=grid, show_margin=False,
                   title={"zh": "Adam 训练 MLP：决策边界一轮轮弯过去", "en": "Adam trains an MLP: the boundary bends epoch by epoch"})
        clf = s.make_mlp("adam")
        keep = {1, 3, 8, 20, 50, 120}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for epoch in range(1, 121):
                clf.partial_fit(X, y, classes=[0, 1])
                if epoch in keep:
                    p = np.clip(clf.predict_proba(grid.points)[:, 1], 1e-6, 1 - 1e-6)
                    tr.record(field=np.log(p / (1 - p)), metrics={"loss": float(clf.loss_)}, note=f"epoch {epoch}")
        return tr

    t.trace(trace)
