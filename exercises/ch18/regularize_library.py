"""Regularising a neural network with scikit-learn: the L2 penalty `alpha` of MLPClassifier chosen on
a validation set, and built-in early stopping."""

import numpy as np
from sklearn.neural_network import MLPClassifier


def fit_alpha(X, y, alpha, seed=0):
    """A 2-hidden-layer MLP (64, 64), ReLU, Adam, at most 500 epochs, with L2 penalty `alpha`."""
    # @blank alpha
    # zh: MLPClassifier(hidden_layer_sizes=(64, 64), alpha=alpha, max_iter=500, random_state=seed)，再 fit
    # en: MLPClassifier(hidden_layer_sizes=(64, 64), alpha=alpha, max_iter=500, random_state=seed), then fit
    return MLPClassifier(hidden_layer_sizes=(64, 64), alpha=alpha, max_iter=500, random_state=seed).fit(X, y)
    # @end


def weight_norm(model):
    """Euclidean norm of all weight matrices (biases excluded) of a fitted MLPClassifier."""
    # @blank norm
    # zh: 权重矩阵在 model.coefs_ 里
    # en: The weight matrices are in model.coefs_
    return float(np.sqrt(sum((W ** 2).sum() for W in model.coefs_)))
    # @end


def pick_alpha(X, y, Xv, yv, alphas):
    """Fit one model per alpha; return (best alpha, its model) by accuracy on (Xv, yv); ties go to the smaller alpha."""
    # @blank pick
    # zh: 逐个训练，用 model.score(Xv, yv) 比较，只在严格更好时替换
    # en: Fit each, compare model.score(Xv, yv), and replace only on a strict improvement
    best = None
    for a in alphas:
        m = fit_alpha(X, y, a)
        if best is None or m.score(Xv, yv) > best[1].score(Xv, yv):
            best = (a, m)
    return best
    # @end


def fit_early_stopping(X, y, seed=0):
    """The same MLP with alpha = 1e-4, holding out 20% of (X, y) and stopping after 20 epochs without improvement."""
    # @blank early
    # zh: early_stopping=True、validation_fraction=0.2、n_iter_no_change=20
    # en: early_stopping=True, validation_fraction=0.2, n_iter_no_change=20
    return MLPClassifier(hidden_layer_sizes=(64, 64), alpha=1e-4, max_iter=500, early_stopping=True,
                         validation_fraction=0.2, n_iter_no_change=20, random_state=seed).fit(X, y)
    # @end
