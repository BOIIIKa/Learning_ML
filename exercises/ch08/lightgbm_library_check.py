"""Checks for lightgbm_library.py. Runs identically under CPython and Pyodide."""

import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split

from learnml import Grid, Trace


def check(s, t):
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.25, random_state=0, stratify=y_tr)
    state = {}

    def t_model():
        model = s.build_model(n_estimators=50, learning_rate=0.2, num_leaves=4)
        assert isinstance(model, lgb.LGBMClassifier), "build_model should return an LGBMClassifier"
        p = model.get_params()
        got = (p["n_estimators"], p["learning_rate"], p["num_leaves"], p["min_child_samples"])
        assert got == (50, 0.2, 4, 5), f"(n_estimators, learning_rate, num_leaves, min_child_samples) = {got}"

    def t_fit():
        model = s.build_model()
        curve = s.fit(model, X_tr, y_tr, X_va, y_va)
        assert len(curve) == model.n_estimators, "one validation loss per boosting round"
        assert curve[-1] < curve[0], "validation loss should go down"
        state["model"] = model
        acc = model.score(X_te, y_te)
        assert acc >= 0.93, f"test accuracy {acc:.3f} < 0.93"

    def t_staged():
        model = state.get("model")
        assert model is not None, "fix the previous test first"
        scores = s.staged_scores(model, X_te[:5], [1, 10, 100])
        assert len(scores) == 3 and all(len(v) == 5 for v in scores), "one score vector per k"
        assert np.allclose(scores[-1], model.predict(X_te[:5], raw_score=True)), "k = all trees should match predict"
        assert not np.allclose(scores[0], scores[-1]), "1 tree and 100 trees should differ"

    t.test("构建模型", "Building the model", t_model)
    t.test("带验证集训练并达到 93% 测试准确率", "Training with validation, reaching 93% test accuracy", t_fit)
    t.test("分阶段预测", "Staged predictions", t_staged)

    def boosting_trace():
        Xm, ym = make_moons(n_samples=120, noise=0.25, random_state=0)
        Xv, yv = make_moons(n_samples=120, noise=0.25, random_state=1)
        grid = Grid(Xm, n=40)
        model = s.build_model(n_estimators=60, learning_rate=0.2)
        curve = s.fit(model, Xm, ym, Xv, yv)
        trace = Trace("classifier2d", Xm, np.where(ym == 1, 1, -1), grid=grid,
                      title={"zh": "LightGBM 逐轮提升", "en": "LightGBM, round by round"})
        iterations = list(range(1, 61, 2))
        for k, field in zip(iterations, s.staged_scores(model, grid.points, iterations)):
            train_pred = model.predict(Xm, num_iteration=k)
            trace.record(field=field, metrics={"error": float(np.mean(train_pred != ym)), "logloss": curve[k - 1]},
                         note=f"{k} trees")
        return trace

    t.trace(boosting_trace)
