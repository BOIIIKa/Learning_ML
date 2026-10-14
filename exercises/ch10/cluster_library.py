"""Clustering with scikit-learn: choose k by silhouette, compare linkages, score against known labels.

The wine data (178 wines, 13 chemical measurements) come from 3 cultivars; we pretend not to know
them while clustering, then use them only to evaluate.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def standardize(X):
    """Return the standardized features (mean 0, variance 1 per column)."""
    # @blank scale
    # zh: StandardScaler().fit_transform
    # en: StandardScaler().fit_transform
    return StandardScaler().fit_transform(X)
    # @end


def best_k(Z, ks):
    """The k in ks with the highest silhouette for KMeans(k, n_init=10, random_state=0)."""
    # @blank bestk
    # zh: 对每个 k 拟合 KMeans，用 silhouette_score(Z, labels) 打分，返回得分最高的 k
    # en: Fit KMeans for each k, score with silhouette_score(Z, labels), return the best k
    scores = [silhouette_score(Z, KMeans(k, n_init=10, random_state=0).fit_predict(Z)) for k in ks]
    return ks[int(np.argmax(scores))]
    # @end


def linkage_scores(Z, y, k):
    """Dict linkage → ARI of AgglomerativeClustering(n_clusters=k, linkage=…) against y, for ward/complete/average/single."""
    # @blank links
    # zh: 四种 linkage 各拟合一次，用 adjusted_rand_score(y, labels) 打分
    # en: Fit once per linkage and score with adjusted_rand_score(y, labels)
    return {L: adjusted_rand_score(y, AgglomerativeClustering(n_clusters=k, linkage=L).fit_predict(Z))
            for L in ("ward", "complete", "average", "single")}
    # @end
