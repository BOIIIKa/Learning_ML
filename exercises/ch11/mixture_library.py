"""Gaussian mixtures in scikit-learn: choose the number of components and the covariance type by BIC,
then compare with k-means on clusters that are not round."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture


def bic_table(X, ks, types=("full", "tied", "diag", "spherical")):
    """Dict (k, covariance_type) → BIC of GaussianMixture(k, covariance_type=…, n_init=3, random_state=0)."""
    # @blank bic
    # zh: 两重循环拟合，调用 .bic(X)
    # en: Fit in a double loop and call .bic(X)
    return {(k, c): GaussianMixture(k, covariance_type=c, n_init=3, random_state=0).fit(X).bic(X) for k in ks for c in types}
    # @end


def best_model(X, table):
    """Refit and return the GaussianMixture with the smallest BIC in the table."""
    # @blank best
    # zh: min(table, key=table.get) 给出最好的 (k, 类型)，用同样的参数重新拟合
    # en: min(table, key=table.get) gives the best (k, type); refit with the same settings
    k, c = min(table, key=table.get)
    return GaussianMixture(k, covariance_type=c, n_init=3, random_state=0).fit(X)
    # @end


def compare(X, y, k):
    """ARI against y of (KMeans(k, n_init=10, random_state=0),
    GaussianMixture(k, covariance_type="tied", n_init=3, random_state=0)) — the shared covariance BIC prefers here.
    (With covariance_type="full", the default k-means initialization traps EM in a poor local optimum on these data.)"""
    # @blank compare
    # zh: 两个模型都 fit_predict，再用 adjusted_rand_score 打分
    # en: fit_predict both models and score with adjusted_rand_score
    km = KMeans(k, n_init=10, random_state=0).fit_predict(X)
    gm = GaussianMixture(k, covariance_type="tied", n_init=3, random_state=0).fit_predict(X)
    return adjusted_rand_score(y, km), adjusted_rand_score(y, gm)
    # @end
