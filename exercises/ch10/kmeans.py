"""k-means from scratch: assignment, update, k-means++ seeding and the objective."""

import numpy as np


def assign(X, C):
    """Index of the nearest centre for every row of X."""
    # @blank assign
    # zh: 用广播算 (n, K) 的平方距离矩阵，再按行 argmin
    # en: Broadcast the (n, K) matrix of squared distances, then take the argmin of each row
    D = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    return D.argmin(axis=1)
    # @end


def update(X, labels, C):
    """Each centre moves to the mean of its cluster (an empty cluster keeps its old centre)."""
    # @blank update
    # zh: 对每个 k 取 X[labels == k] 的均值；没有成员时保留 C[k]
    # en: For each k take the mean of X[labels == k]; keep C[k] if the cluster is empty
    return np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else C[k] for k in range(len(C))])
    # @end


def objective(X, labels, C):
    """Within-cluster sum of squares."""
    return float(((X - C[labels]) ** 2).sum())


def kmeans_pp(X, K, rng):
    """k-means++ seeding: first centre uniform, then each new one with probability ∝ D(x)²."""
    C = [X[rng.integers(len(X))]]
    while len(C) < K:
        # @blank pp
        # zh: D² = 每个点到已选中心的最小平方距离；按 D² / ΣD² 的概率用 rng.choice 抽一个点
        # en: D² = each point's smallest squared distance to the chosen centres; draw a point with rng.choice using probabilities D² / ΣD²
        D2 = np.min(((X[:, None, :] - np.array(C)[None, :, :]) ** 2).sum(axis=2), axis=1)
        C.append(X[rng.choice(len(X), p=D2 / D2.sum())])
        # @end
    return np.array(C)


def lloyd(X, C, max_iter=100, trace=None):
    """Alternate assign and update until the labels stop changing. Returns (labels, centres)."""
    labels = assign(X, C)
    for t in range(max_iter):
        C = update(X, labels, C)
        if trace is not None:
            trace(t, labels, C)
        new = assign(X, C)
        if np.array_equal(new, labels):
            break
        labels = new
    return labels, C
