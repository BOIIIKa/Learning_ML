"""Sequence labelling with scikit-learn: a per-day classifier, then the same scores decoded with Viterbi.

Each day has a noisy 3-number lab panel; the hidden condition (0 = stable, 1 = worse) is sticky.
A classifier that looks at one day at a time ignores that stickiness; combining its log-probabilities
with transition log-probabilities and decoding the whole sequence (a simple hybrid of an HMM and a
discriminative model) uses it.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def window(X, w=1):
    """Features of day t = the panels of days t−w … t+w side by side (edges padded by repetition)."""
    # @blank window
    # zh: 用 np.pad(X, ((w, w), (0, 0)), mode="edge") 补边，再把 2w+1 个平移后的切片按列拼接
    # en: Pad with np.pad(X, ((w, w), (0, 0)), mode="edge"), then concatenate the 2w+1 shifted slices column-wise
    P = np.pad(X, ((w, w), (0, 0)), mode="edge")
    return np.hstack([P[k:k + len(X)] for k in range(2 * w + 1)])
    # @end


def per_day_model():
    """StandardScaler → LogisticRegression(max_iter=1000)."""
    # @blank model
    # zh: 一个标准化 + 逻辑回归的管道
    # en: A standardize + logistic regression pipeline
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    # @end


def transition_logprob(labels, alpha=1.0):
    """log of the 2×2 transition matrix estimated from a label sequence with add-α smoothing."""
    # @blank trans
    # zh: 数相邻两天的 (y_t, y_{t+1}) 计数，加 α 后按行归一化，再取对数
    # en: Count consecutive pairs (y_t, y_{t+1}), add α, normalize each row, take the log
    C = np.full((2, 2), alpha)
    for a, b in zip(labels[:-1], labels[1:]):
        C[a, b] += 1
    return np.log(C / C.sum(axis=1, keepdims=True))
    # @end


def decode(logp, logA):
    """Viterbi over per-day scores logp (T, 2) and transitions logA; returns the label sequence."""
    T = len(logp)
    d = logp[0].copy()
    back = np.zeros((T, 2), dtype=int)
    for t in range(1, T):
        cand = d[:, None] + logA
        back[t] = cand.argmax(axis=0)
        d = cand.max(axis=0) + logp[t]
    path = [int(d.argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t][path[-1]]))
    return np.array(path[::-1])
