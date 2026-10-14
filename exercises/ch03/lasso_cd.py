"""Ridge in closed form and the lasso by coordinate descent, written from scratch.

Both minimize (1/2n)‖y − Xw‖² plus a penalty: (λ/2)‖w‖² for ridge, λ‖w‖₁ for the lasso.
X is assumed standardized and y centred, so there is no intercept.
"""

import numpy as np


def ridge(X, y, lam):
    """Closed form: (XᵀX/n + λI)⁻¹ Xᵀy/n."""
    # @blank ridge
    # zh: 解线性方程组 (XᵀX/n + λI) w = Xᵀy/n（用 np.linalg.solve，不要求逆）
    # en: Solve (XᵀX/n + λI) w = Xᵀy/n with np.linalg.solve (don't invert)
    n, p = X.shape
    return np.linalg.solve(X.T @ X / n + lam * np.eye(p), X.T @ y / n)
    # @end


def soft_threshold(z, g):
    """S(z, g) = sign(z) · max(|z| − g, 0)."""
    # @blank soft
    # zh: 用 np.sign 和 np.maximum 写成一行
    # en: One line with np.sign and np.maximum
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)
    # @end


def lasso_cd(X, y, lam, w0=None, sweeps=500, tol=1e-10):
    """Cyclic coordinate descent for the lasso, keeping the residual r = y − Xw up to date."""
    n, p = X.shape
    w = np.zeros(p) if w0 is None else np.array(w0, dtype=float)
    r = y - X @ w
    sq = (X ** 2).sum(axis=0) / n
    for _ in range(sweeps):
        biggest = 0.0
        for j in range(p):
            # @blank coord
            # zh: rho = X[:, j] · (r + X[:, j] w_j) / n；新值 = soft_threshold(rho, lam) / sq[j]；再用变化量更新残差 r
            # en: rho = X[:, j] · (r + X[:, j] w_j) / n; new = soft_threshold(rho, lam) / sq[j]; then update the residual r by the change
            rho = X[:, j] @ (r + X[:, j] * w[j]) / n
            new = soft_threshold(rho, lam) / sq[j]
            r -= X[:, j] * (new - w[j])
            # @end
            biggest = max(biggest, abs(new - w[j]))
            w[j] = new
        if biggest < tol:
            break
    return w


def lasso_path(X, y, lams):
    """Solutions for each λ (largest first), warm-starting each from the previous one."""
    ws, w = [], None
    for lam in lams:
        w = lasso_cd(X, y, lam, w0=w)
        ws.append(w.copy())
    return ws
