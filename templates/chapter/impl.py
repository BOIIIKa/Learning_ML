"""TODO(scaffold) One-line summary of what the student implements.

Each `# @blank id` … `# @end` region becomes a TODO in the student's skeleton; the comment lines
`# zh:` / `# en:` are the hint shown in each language. Keep blanks small (1–4 lines) and give
the shapes in the docstring.
"""

import numpy as np


def loss(w, X, y):
    """Mean squared error of the linear model X @ w."""
    # @blank loss
    # zh: 残差 X @ w − y 的平方的平均
    # en: The mean of the squared residuals X @ w − y
    return float(np.mean((X @ w - y) ** 2))
    # @end


def fit(X, y, lr=0.1, steps=200, trace=None):
    """Gradient descent from zero, recording every 10 steps if a trace is given."""
    w = np.zeros(X.shape[1])
    for t in range(steps):
        if trace is not None and t % 10 == 0:
            trace.record(w=[w[1], -1.0], b=w[0], metrics={"loss": loss(w, X, y)}, note=f"step {t}")
        # @blank step
        # zh: w ← w − lr · (2/n) Xᵀ(Xw − y)
        # en: w ← w − lr · (2/n) Xᵀ(Xw − y)
        w = w - lr * 2 / len(y) * X.T @ (X @ w - y)
        # @end
    return w
