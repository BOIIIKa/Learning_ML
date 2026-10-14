"""k nearest neighbours, a kd-tree search and a cubic regression spline, from scratch.

Labels are y ∈ {−1, +1}. The kd-tree is given as nested dicts
{"i": point index, "axis": 0 or 1, "left": subtree or None, "right": subtree or None}.
"""

import numpy as np


def pairwise(Q, X, p=2):
    """Matrix of L_p distances, D[a, b] = ‖Q[a] − X[b]‖_p, for p = 1, 2 or np.inf."""
    # @blank dist
    # zh: 先用广播得到 |Q[:, None, :] − X[None, :, :]|；p = inf 时取最大值，否则求 p 次方和再开 p 次方
    # en: Broadcast |Q[:, None, :] − X[None, :, :]|; for p = inf take the max, otherwise sum the p-th powers and take the p-th root
    A = np.abs(Q[:, None, :] - X[None, :, :])
    if p == np.inf:
        return A.max(axis=2)
    return (A ** p).sum(axis=2) ** (1 / p)
    # @end


def knn_predict(X, y, Q, k, p=2):
    """Majority vote of the k nearest training points for each row of Q (ties → +1)."""
    D = pairwise(Q, X, p)
    # @blank vote
    # zh: 用 np.argsort(D, axis=1, kind="stable")[:, :k] 取最近 k 个，对它们的标签求和，非负判为 +1
    # en: Take the k nearest with np.argsort(D, axis=1, kind="stable")[:, :k], sum their labels, predict +1 when the sum is ≥ 0
    idx = np.argsort(D, axis=1, kind="stable")[:, :k]
    return np.where(y[idx].sum(axis=1) >= 0, 1, -1)
    # @end


def build_kd(P, idx=None, depth=0):
    """2-D kd-tree split at the median (element ⌊n/2⌋ after sorting) on x, y, x, …"""
    if idx is None:
        idx = list(range(len(P)))
    if not idx:
        return None
    axis = depth % 2
    s = sorted(idx, key=lambda i: (P[i][axis], i))
    m = len(s) // 2
    return {"i": s[m], "axis": axis, "left": build_kd(P, s[:m], depth + 1), "right": build_kd(P, s[m + 1:], depth + 1)}


def kd_nearest(node, P, q, best=None, visited=None):
    """Nearest neighbour of q. Returns (index, distance, visited list)."""
    if best is None:
        best, visited = [-1, np.inf], []
    if node is None:
        return best[0], best[1], visited
    i, a = node["i"], node["axis"]
    visited.append(i)
    d = float(np.linalg.norm(np.asarray(P[i]) - q))
    if d < best[1]:
        best[0], best[1] = i, d
    diff = q[a] - P[i][a]
    near, far = (node["left"], node["right"]) if diff < 0 else (node["right"], node["left"])
    kd_nearest(near, P, q, best, visited)
    # @blank prune
    # zh: 只有当以 q 为圆心、当前最近距离为半径的球越过切分线（|diff| < 最近距离）时，才去搜索另一侧
    # en: Search the far side only if the ball around q with the best distance as radius crosses the splitting line (|diff| < best distance)
    if abs(diff) < best[1]:
        kd_nearest(far, P, q, best, visited)
    # @end
    return best[0], best[1], visited


def spline_basis(x, knots):
    """Cubic regression-spline basis: columns 1, x, x², x³, (x − ξ_k)³₊ for every knot ξ_k."""
    # @blank basis
    # zh: 前四列是 x 的 0 到 3 次幂；之后每个节点一列 max(x − ξ, 0)³
    # en: The first four columns are powers 0 to 3 of x; then one column max(x − ξ, 0)³ per knot
    cols = [x ** j for j in range(4)] + [np.maximum(x - k, 0.0) ** 3 for k in knots]
    return np.column_stack(cols)
    # @end


def fit_spline(x, y, knots):
    """Least-squares coefficients of the spline (use np.linalg.lstsq)."""
    return np.linalg.lstsq(spline_basis(x, knots), y, rcond=None)[0]
