"""k-nearest-neighbour classification from scratch (numpy only).

Model: the training set itself (no parameters to fit). Strategy: majority vote, i.e. the class
that minimizes 0-1 loss among the k neighbours. Algorithm: brute-force distance computation.
Labels are ±1 and k is odd, so a vote can never tie.
"""

import numpy as np


def pairwise_distances(A, B, p=2):
    """Matrix D with D[i, j] = ‖A[i] − B[j]‖_p for p = 1, 2 or np.inf."""
    diff = A[:, None, :] - B[None, :, :]  # shape (len(A), len(B), d)
    # @blank dist
    # zh: 沿最后一维求范数：p = 1 绝对值之和，p = 2 平方和开方，p = ∞ 绝对值的最大值
    # en: Take the norm along the last axis: p = 1 sum of |·|, p = 2 root of the sum of squares, p = ∞ max of |·|
    if p == 1:
        return np.abs(diff).sum(axis=-1)
    if p == 2:
        return np.sqrt((diff ** 2).sum(axis=-1))
    return np.abs(diff).max(axis=-1)
    # @end


def knn_score(X_train, y_train, X, k, p=2):
    """For each row of X, the mean ±1 label of its k nearest training points (> 0 votes for +1)."""
    D = pairwise_distances(X, X_train, p)
    # @blank vote
    # zh: 每行按距离排序取前 k 个下标（argsort, kind="stable"），再对这些邻居的标签取平均
    # en: Sort each row and keep the first k indices (argsort, kind="stable"), then average those neighbours' labels
    idx = np.argsort(D, axis=1, kind="stable")[:, :k]
    return y_train[idx].mean(axis=1)
    # @end


def knn_predict(X_train, y_train, X, k, p=2):
    """Majority vote: +1 if more than half of the k neighbours are +1, else −1."""
    return np.where(knn_score(X_train, y_train, X, k, p) > 0, 1, -1)


def loo_error(X, y, k, p=2):
    """Leave-one-out error: classify each training point by its k nearest *other* training points."""
    D = pairwise_distances(X, X, p)
    # @blank loo
    # zh: 把对角线设为 +∞（点不能做自己的邻居），然后像上面一样投票，返回判错的比例
    # en: Set the diagonal to +∞ (a point may not be its own neighbour), vote as above, and return the share of mistakes
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1, kind="stable")[:, :k]
    pred = np.where(y[idx].sum(axis=1) > 0, 1, -1)
    return float(np.mean(pred != y))
    # @end


def choose_k(X, y, ks, p=2, trace=None, grid=None):
    """Return the k in `ks` with the smallest leave-one-out error (the first one on ties).

    With a trace, record each k's decision field on `grid` together with its training and LOO errors.
    """
    best_k, best_err = None, np.inf
    for k in ks:
        err = loo_error(X, y, k, p)
        if trace is not None:
            train_err = float(np.mean(knn_predict(X, y, X, k, p) != y))
            trace.record(field=knn_score(X, y, grid.points, k, p), metrics={"train": train_err, "loo": err}, note=f"k = {k}")
        # @blank best
        # zh: 若 err 严格小于当前最优，就更新 best_k 与 best_err
        # en: If err is strictly smaller than the best so far, update best_k and best_err
        if err < best_err:
            best_k, best_err = k, err
        # @end
    return best_k
