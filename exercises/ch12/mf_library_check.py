"""Checks for mf_library.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

from learnml import Trace

DOCS = [
    "patient doctor hospital treatment",
    "doctor nurse patient ward hospital",
    "hospital patient surgery doctor",
    "stock market shares investor",
    "market prices stock trading investor",
    "investor shares trading market",
]


def check(s, t):
    digits = load_digits()

    def t_ncomp():
        k = s.components_for(digits.data, 0.9)
        cum = np.cumsum(PCA().fit(digits.data).explained_variance_ratio_)
        assert cum[k - 1] >= 0.9 and cum[k - 2] < 0.9, f"k = {k} is not the smallest with ≥ 90%"
        assert k < 64 // 2, f"{k} of 64 components keep 90% of the variance"

    def t_lsa():
        vec, svd, Z = s.lsa(DOCS, 2)
        assert Z.shape == (6, 2), "two coordinates per document"
        cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
        within = cos(Z[0], Z[1])
        across = cos(Z[0], Z[3])
        assert within > 0.9 and abs(across) < 0.3, f"same topic {within:.2f}, different topics {across:.2f}"

    def t_nmf():
        topics = s.nmf_topics(DOCS, 2, 3)
        flat = [set(tp) for tp in topics]
        med = {"patient", "doctor", "hospital"}
        fin = {"market", "stock", "investor", "shares", "trading"}
        assert any(f <= med | {"nurse", "ward", "surgery", "treatment"} for f in flat) and any(f <= fin | {"prices"} for f in flat), f"topics {topics}"

    t.test("保留 90% 方差需要几个主成分", "How many components keep 90% of the variance", t_ncomp)
    t.test("LSA 把同一话题的文档聚在一起", "LSA groups documents of the same topic", t_lsa)
    t.test("NMF 找出两个话题", "NMF finds the two topics", t_nmf)

    def trace():
        mask = np.isin(digits.target, [3, 8])
        X = digits.data[mask][:150]
        y = np.where(digits.target[mask][:150] == 3, 1, -1)
        Z = PCA(2).fit_transform(X)
        tr = Trace("classifier2d", Z, y, show_margin=False,
                   title={"zh": "手写 3 与 8 在前两个主成分上的投影（未用标签）", "en": "Handwritten 3s and 8s on the first two components (no labels used)"})
        tr.record(metrics={"var": float(np.var(Z[:, 0]))}, note="PCA")
        return tr

    t.trace(trace)
