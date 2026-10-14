"""AdaBoost with decision stumps, written from scratch.

Each round fits the stump with the lowest weighted error, gives it a vote
α = ½ ln((1 − e) / e), and re-weights the samples so the next stump focuses on
the points the ensemble still gets wrong.
"""

import numpy as np


def stump_predict(X, feature, threshold, polarity):
    """±1 prediction of a one-split tree on a single feature."""
    return polarity * np.where(X[:, feature] <= threshold, 1.0, -1.0)


def weighted_error(w, y, pred):
    """Weighted training error of `pred`; the weights w sum to 1."""
    # @blank error
    # zh: 加权误差 = 被分错样本的权重之和
    # en: Weighted error = sum of the weights of misclassified samples
    return float(np.sum(w[pred != y]))
    # @end


def best_stump(X, y, w):
    """Exhaustive search over features, midpoints and polarities."""
    best = None
    for feature in range(X.shape[1]):
        values = np.unique(X[:, feature])
        for threshold in (values[:-1] + values[1:]) / 2:
            for polarity in (1.0, -1.0):
                err = weighted_error(w, y, stump_predict(X, feature, threshold, polarity))
                if best is None or err < best[0]:
                    best = (err, feature, threshold, polarity)
    return best


def stump_weight(err):
    """The stump's vote α; better stumps get larger votes."""
    err = min(max(err, 1e-10), 1 - 1e-10)
    # @blank alpha
    # zh: α = ½ ln((1 − e) / e)
    # en: α = ½ ln((1 − e) / e)
    return 0.5 * np.log((1 - err) / err)
    # @end


def update_weights(w, alpha, y, pred):
    """Up-weight mistakes, down-weight hits, then renormalize."""
    # @blank weights
    # zh: w_i ← w_i · exp(−α y_i G(x_i))，再除以规范化因子 Z 使权重和为 1
    # en: w_i ← w_i · exp(−α y_i G(x_i)), then divide by the normalizer Z so the weights sum to 1
    w = w * np.exp(-alpha * y * pred)
    return w / w.sum()
    # @end


def predict(model, X):
    """Sign of the weighted vote Σ α_m G_m(x)."""
    # @blank predict
    # zh: 对所有弱分类器的加权投票 Σ α_m G_m(x) 取符号
    # en: Take the sign of the weighted vote Σ α_m G_m(x) over all stumps
    score = sum(alpha * stump_predict(X, f, thr, pol) for alpha, f, thr, pol in model)
    return np.where(score >= 0, 1.0, -1.0)
    # @end


def adaboost(X, y, rounds=20, trace=None):
    """Returns the model: a list of (alpha, feature, threshold, polarity)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    w = np.full(len(y), 1.0 / len(y))
    model, bound = [], 1.0
    field = None if trace is None or trace.grid is None else np.zeros(len(trace.grid.points))
    for _ in range(rounds):
        err, f, thr, pol = best_stump(X, y, w)
        alpha = stump_weight(err)
        pred = stump_predict(X, f, thr, pol)
        model.append((alpha, f, thr, pol))
        if trace is not None:
            bound *= 2 * np.sqrt(err * (1 - err))
            train_err = np.mean(predict(model, X) != y)
            if field is not None:
                field = field + alpha * stump_predict(trace.grid.points, f, thr, pol)
            trace.record(field=field, weights=w * len(y), highlight=np.flatnonzero(pred != y),
                         metrics={"error": train_err, "bound": bound, "alpha": alpha},
                         note=f"x{f + 1} ≤ {thr:.2f}")
        w = update_weights(w, alpha, y, pred)
    return model
