"""Linear soft-margin SVM trained with SMO (sequential minimal optimization).

The dual problem:  max_α  Σα_i − ½ ΣΣ α_i α_j y_i y_j K_ij
                   s.t.   0 ≤ α_i ≤ C,  Σ α_i y_i = 0

SMO repeatedly picks a pair (α_i, α_j), solves the two-variable subproblem in closed
form, and clips the result back into the box. Fill in the blanks of the pair update.
"""

import numpy as np


def bounds(a_i, a_j, y_i, y_j, C):
    """Feasible interval [L, H] for the new α_j on the line Σ α_k y_k = const."""
    # @blank bounds
    # zh: y_i ≠ y_j 时 α_j − α_i 守恒，否则 α_i + α_j 守恒；据此写出 L、H
    # en: If y_i ≠ y_j then α_j − α_i is conserved, otherwise α_i + α_j is; derive L and H
    if y_i != y_j:
        L, H = max(0.0, a_j - a_i), min(C, C + a_j - a_i)
    else:
        L, H = max(0.0, a_i + a_j - C), min(C, a_i + a_j)
    # @end
    return L, H


def eta(K, i, j):
    """Curvature of the dual objective along the constraint line."""
    # @blank eta
    # zh: 计算 η = K_ii + K_jj − 2K_ij
    # en: Compute η = K_ii + K_jj − 2K_ij
    return K[i, i] + K[j, j] - 2.0 * K[i, j]
    # @end


def new_alpha_j(a_j, y_j, E_i, E_j, eta_ij, L, H):
    """Unconstrained optimum along the line, then clipped into [L, H]."""
    # @blank alpha-j
    # zh: 先求未剪辑解 α_j + y_j (E_i − E_j) / η，再裁剪到 [L, H]
    # en: Take the unclipped optimum α_j + y_j (E_i − E_j) / η, then clip it into [L, H]
    unclipped = a_j + y_j * (E_i - E_j) / eta_ij
    return min(H, max(L, unclipped))
    # @end


def new_alpha_i(a_i, a_j_old, a_j_new, y_i, y_j):
    """Move α_i so that α_i y_i + α_j y_j stays constant."""
    # @blank alpha-i
    # zh: 由约束 Σ α_k y_k = 0 推出 α_i 的变化量
    # en: Use the constraint Σ α_k y_k = 0 to get the change in α_i
    return a_i + y_i * y_j * (a_j_old - a_j_new)
    # @end


def new_b(b, E_i, E_j, y_i, y_j, d_i, d_j, K, i, j, a_i_new, a_j_new, C):
    """Threshold after the update; d_i, d_j are the changes in α_i, α_j."""
    b1 = b - E_i - y_i * d_i * K[i, i] - y_j * d_j * K[i, j]
    b2 = b - E_j - y_i * d_i * K[i, j] - y_j * d_j * K[j, j]
    # @blank b
    # zh: 若 0 < α_i < C 取 b1；否则若 0 < α_j < C 取 b2；否则取二者平均
    # en: Use b1 if 0 < α_i < C; else b2 if 0 < α_j < C; otherwise their average
    if 0 < a_i_new < C:
        return b1
    if 0 < a_j_new < C:
        return b2
    return (b1 + b2) / 2.0
    # @end


def smo(X, y, C=1.0, tol=1e-3, max_steps=500, trace=None):
    """Train a linear SVM. Returns (alpha, b). Records one trace step per pair update."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n = len(y)
    K = X @ X.T
    alpha, b = np.zeros(n), 0.0

    def record(pair, note):
        if trace is not None:
            w = (alpha * y) @ X
            dual = alpha.sum() - 0.5 * (alpha * y) @ K @ (alpha * y)
            support = np.flatnonzero(alpha > 1e-6)
            trace.record(w=w, b=b, highlight=pair, support=support, metrics={"dual": dual}, note=note)

    record([], "init")
    for _ in range(max_steps):
        E = (alpha * y) @ K + b - y
        r = y * E  # y_k f(x_k) − 1
        violation = np.where(((r < -tol) & (alpha < C)) | ((r > tol) & (alpha > 0)), np.abs(r), 0.0)
        if violation.max() <= 0:
            break
        progressed = False
        # Outer loop: worst KKT violator first. Inner loop: largest |E_i − E_j| first.
        for i in np.argsort(-violation):
            if violation[i] <= 0:
                break
            for j in np.argsort(-np.abs(E - E[i])):
                if j == i:
                    continue
                L, H = bounds(alpha[i], alpha[j], y[i], y[j], C)
                k = eta(K, i, j)
                if L >= H or k <= 1e-12:
                    continue
                a_j = new_alpha_j(alpha[j], y[j], E[i], E[j], k, L, H)
                if abs(a_j - alpha[j]) < 1e-8:
                    continue
                a_i = new_alpha_i(alpha[i], alpha[j], a_j, y[i], y[j])
                b = new_b(b, E[i], E[j], y[i], y[j], a_i - alpha[i], a_j - alpha[j], K, i, j, a_i, a_j, C)
                alpha[i], alpha[j] = a_i, a_j
                record([int(i), int(j)], "step")
                progressed = True
                break
            if progressed:
                break
        if not progressed:
            break
    return alpha, b
