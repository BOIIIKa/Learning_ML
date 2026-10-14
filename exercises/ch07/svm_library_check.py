"""Checks for svm_library.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from learnml import Grid, Trace


def check(s, t):
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    state = {}

    def t_pipeline():
        model = s.build_model(C=3.0, gamma=0.05)
        assert isinstance(model, Pipeline), "build_model should return a Pipeline"
        steps = [type(step).__name__ for _, step in model.steps]
        assert steps == ["StandardScaler", "SVC"], f"pipeline steps are {steps}"
        svc = model.steps[-1][1]
        assert (svc.kernel, svc.C, svc.gamma) == ("rbf", 3.0, 0.05), "pass kernel, C and gamma through"

    def t_tune():
        search = s.tune(X_tr, y_tr)
        params = set(search.param_grid)
        assert params == {"svc__C", "svc__gamma"}, f"grid keys are {sorted(params)}"
        state["acc"] = search.score(X_te, y_te)
        assert state["acc"] >= 0.95, f"test accuracy {state['acc']:.3f} < 0.95"

    t.test("组装流水线", "Assembling the pipeline", t_pipeline)
    t.test("网格搜索并达到 95% 测试准确率", "Grid search reaching 95% test accuracy", t_tune)

    def hyperparameter_trace():
        Xm, ym = make_moons(n_samples=120, noise=0.25, random_state=0)
        grid = Grid(Xm, n=40)
        trace = Trace("classifier2d", Xm, np.where(ym == 1, 1, -1), grid=grid,
                      title={"zh": "C 与 γ 如何塑造边界", "en": "How C and γ shape the boundary"})
        for C in (0.1, 1, 100):
            for gamma in (0.1, 1, 10):
                model = s.build_model(C=C, gamma=gamma).fit(Xm, ym)
                trace.record(field=model.decision_function(grid.points),
                             support=model.steps[-1][1].support_, note=f"C={C}, γ={gamma}")
        return trace

    t.trace(hyperparameter_trace)
