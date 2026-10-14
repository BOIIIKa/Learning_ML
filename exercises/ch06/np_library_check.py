"""Checks for np_library.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from learnml import Grid, Trace


def friedman(n=300, seed=0):
    """An additive target: y = 10 sin(π x₁) + 20 (x₂ − 0.5)² + 10 x₃ + noise (x₄ is irrelevant)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, 4))
    y = 10 * np.sin(np.pi * X[:, 0]) + 20 * (X[:, 1] - 0.5) ** 2 + 10 * X[:, 2] + rng.normal(0, 1, n)
    return X, y


def check(s, t):
    state = {}

    def t_knn():
        data = load_breast_cancer()
        g = s.knn_search()
        assert isinstance(g, GridSearchCV), "return a GridSearchCV"
        g.fit(data.data, data.target)
        assert g.best_score_ > 0.95, f"best CV accuracy {g.best_score_:.3f} should exceed 0.95"
        k = g.best_params_["kneighborsclassifier__n_neighbors"]
        assert k > 1, f"best k = {k}; 1-NN should not win on noisy data"

    def t_compare():
        X, y = friedman()
        lin, add = s.compare(X, y)
        assert add > lin + 0.05, f"GAM R² {add:.3f} should clearly beat linear {lin:.3f} on curved effects"
        state["fit"] = s.gam().fit(X, y)
        state["X"] = X

    def t_curve():
        m, X = state.get("fit"), state.get("X")
        assert m is not None, "fix the previous test first"
        grid = np.linspace(0.05, 0.95, 19)
        c = s.partial_curve(m, X, 1, grid)
        assert c.shape == (19,), "one prediction per grid value"
        assert c[9] < c[0] - 2 and c[9] < c[-1] - 2, "feature 2 enters as 20 (x − 0.5)²: the curve should dip in the middle"
        lin = s.partial_curve(m, X, 2, grid)
        assert lin[-1] > lin[0] + 5, "feature 3 enters linearly with slope 10"

    t.test("kNN 网格搜索", "kNN grid search", t_knn)
    t.test("可加模型胜过线性模型", "The additive model beats the linear one", t_compare)
    t.test("部分依赖曲线", "Partial curves", t_curve)

    def trace():
        X, y01 = make_moons(200, noise=0.3, random_state=0)
        y = 2 * y01 - 1
        grid = Grid(X, n=40)
        tr = Trace("classifier2d", X, y, grid=grid, show_margin=False,
                   title={"zh": "按距离加权的 kNN", "en": "Distance-weighted kNN"})
        for k in (1, 5, 15, 45):
            for w in ("uniform", "distance"):
                m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, weights=w)).fit(X, y)
                p = m.predict_proba(grid.points)[:, 1]
                tr.record(field=2 * p - 1, metrics={"acc": float(m.score(X, y))}, note=f"k = {k}, {w}")
        return tr

    t.trace(trace)
