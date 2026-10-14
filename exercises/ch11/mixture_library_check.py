"""Checks for mixture_library.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.mixture import GaussianMixture

from learnml import Grid, Trace


def stretched(n=450, seed=0):
    """Three long, parallel, tilted clusters: hard for k-means, easy for a GMM."""
    rng = np.random.default_rng(seed)
    g = np.arange(n) % 3
    Z = rng.normal(size=(n, 2)) * [2.0, 0.25]
    R = np.array([[np.cos(0.6), -np.sin(0.6)], [np.sin(0.6), np.cos(0.6)]])
    return Z @ R.T + np.outer(g - 1, [-0.9, 1.4]), g


def check(s, t):
    X, y = stretched()
    state = {}

    def t_bic():
        tab = s.bic_table(X, [1, 2, 3, 4, 5])
        assert len(tab) == 20, "5 values of k × 4 covariance types"
        state["tab"] = tab

    def t_best():
        tab = state.get("tab")
        assert tab is not None, "fix the previous test first"
        m = s.best_model(X, tab)
        assert isinstance(m, GaussianMixture) and m.n_components == 3, f"BIC picks {m.n_components} components; the data have 3"
        assert m.covariance_type in ("tied", "full"), f"BIC picks '{m.covariance_type}'; the clusters share one tilted shape"

    def t_compare():
        km, gm = s.compare(X, y, 3)
        assert gm > 0.9 and gm > km + 0.3, f"GMM ARI {gm:.2f} should beat k-means {km:.2f} by far on stretched clusters"

    t.test("BIC 表", "BIC table", t_bic)
    t.test("BIC 选出 3 个分量与合适的协方差", "BIC picks 3 components and a sensible covariance", t_best)
    t.test("拉长的簇：GMM 胜过 k 均值", "Stretched clusters: GMM beats k-means", t_compare)

    def trace():
        X2, y2 = stretched(240, 1)
        mask = y2 < 2
        X2, y2 = X2[mask], np.where(y2[mask] == 1, 1, -1)
        grid = Grid(X2, n=40)
        tr = Trace("classifier2d", X2, y2, grid=grid, show_margin=False,
                   title={"zh": "四种协方差结构的 GMM 边界", "en": "GMM boundaries for four covariance types"})
        for c in ("spherical", "diag", "tied", "full"):
            m = GaussianMixture(2, covariance_type=c, n_init=3, random_state=0).fit(X2)
            P = m.predict_proba(grid.points)
            k1 = int(np.argmax(m.predict_proba(X2)[y2 > 0].mean(axis=0)))
            tr.record(field=np.log(P[:, k1] + 1e-12) - np.log(P[:, 1 - k1] + 1e-12), metrics={"bic": float(m.bic(X2))}, note=c)
        return tr

    t.trace(trace)
