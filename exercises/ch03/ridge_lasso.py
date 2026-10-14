"""Ridge and lasso from scratch.

Ridge:  minimize ‖y − Xw‖² + λ‖w‖²       → closed form.
Lasso:  minimize ½‖y − Xw‖² + λ‖w‖₁      → cyclic coordinate descent with soft thresholding.
X is assumed centred (no intercept column) and y centred, so the intercept is ȳ and is not penalized.
"""

import numpy as np


def ridge(X, y, lam):
    """Ridge coefficients: solve (XᵀX + λI) w = Xᵀy."""
    # @blank ridge
    # zh: 用 np.linalg.solve 解 (XᵀX + λI) w = Xᵀy（不要显式求逆）
    # en: Use np.linalg.solve on (XᵀX + λI) w = Xᵀy (do not form the inverse)
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)
    # @end


def soft_threshold(z, lam):
    """S_λ(z) = sign(z) · max(|z| − λ, 0), elementwise."""
    # @blank soft
    # zh: 先向 0 平移 λ，越过 0 的部分截成 0
    # en: Shrink towards 0 by λ, clipping anything that would cross 0 to exactly 0
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)
    # @end


def lasso_objective(X, y, w, lam):
    return 0.5 * float(np.sum((y - X @ w) ** 2)) + lam * float(np.sum(np.abs(w)))


def lasso_cd(X, y, lam, sweeps=100, on_sweep=None):
    """Cyclic coordinate descent from w = 0. Calls on_sweep(k, w) after each full sweep."""
    n, p = X.shape
    w = np.zeros(p)
    col_sq = np.sum(X ** 2, axis=0)  # ‖x_j‖²
    r = y - X @ w  # running residual
    for k in range(sweeps):
        for j in range(p):
            # @blank coord
            # zh: 部分残差相关 ρ = x_jᵀ r + ‖x_j‖² w_j；新值 = S_λ(ρ) / ‖x_j‖²；同步更新残差 r
            # en: Partial-residual correlation ρ = x_jᵀ r + ‖x_j‖² w_j; new value = S_λ(ρ) / ‖x_j‖²; keep r up to date
            rho = X[:, j] @ r + col_sq[j] * w[j]
            new = soft_threshold(rho, lam) / col_sq[j]
            r -= X[:, j] * (new - w[j])
            w[j] = new
            # @end
        if on_sweep is not None:
            on_sweep(k, w.copy())
    return w
