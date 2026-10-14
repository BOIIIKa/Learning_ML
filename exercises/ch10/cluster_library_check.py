"""Checks for cluster_library.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score

from learnml import Trace


def check(s, t):
    data = load_wine()
    state = {}

    def t_scale():
        Z = s.standardize(data.data)
        assert np.allclose(Z.mean(axis=0), 0, atol=1e-9) and np.allclose(Z.std(axis=0), 1), "mean 0, variance 1"
        raw = adjusted_rand_score(data.target, KMeans(3, n_init=10, random_state=0).fit_predict(data.data))
        std = adjusted_rand_score(data.target, KMeans(3, n_init=10, random_state=0).fit_predict(Z))
        assert std > raw + 0.3, f"standardizing should matter a lot: ARI {raw:.2f} raw vs {std:.2f} standardized"
        state["Z"] = Z

    def t_k():
        Z = state.get("Z")
        assert Z is not None, "fix the previous test first"
        k = s.best_k(Z, [2, 3, 4, 5, 6])
        assert k == 3, f"silhouette picks k = {k}; the wine data have 3 cultivars"

    def t_links():
        Z = state.get("Z")
        assert Z is not None, "fix the previous test first"
        sc = s.linkage_scores(Z, data.target, 3)
        assert set(sc) == {"ward", "complete", "average", "single"}, "four linkages"
        assert sc["ward"] > 0.75, f"Ward ARI {sc['ward']:.2f} should exceed 0.75"
        assert sc["single"] < 0.2, f"single linkage chains everything into one cluster (ARI {sc['single']:.2f})"

    t.test("标准化很重要", "Standardization matters", t_scale)
    t.test("轮廓系数选 k", "Silhouette picks k", t_k)
    t.test("四种链接的 ARI", "ARI of four linkages", t_links)

    def trace():
        rng = np.random.default_rng(2)
        X = np.vstack([rng.normal([-1.2, 0], 0.5, (40, 2)), rng.normal([1.2, 0.2], 0.5, (40, 2)), [[0.0, 0.1], [-0.4, 0.05], [0.4, 0.15]]])
        y = np.array([-1] * 40 + [1] * 40 + [1, -1, 1])
        tr = Trace("classifier2d", X, y, show_margin=False,
                   title={"zh": "两簇之间一座“桥”：黄圈 = 被分到第 2 簇", "en": "A bridge between two groups: rings = assigned to cluster 2"})
        for L in ("single", "average", "complete", "ward"):
            lab = AgglomerativeClustering(n_clusters=2, linkage=L).fit_predict(X)
            if np.mean(lab[y > 0]) < 0.5:
                lab = 1 - lab
            tr.record(highlight=[int(i) for i in np.flatnonzero(lab == 1)], metrics={"ari": float(adjusted_rand_score(y, lab))}, note=L)
        return tr

    t.trace(trace)
