"""Logistic regression vs the perceptron in scikit-learn, and softmax for more than two classes.

Both are linear classifiers, but only logistic regression models P(y | x): it has predict_proba,
the perceptron only a sign. With K > 2 classes LogisticRegression fits the multinomial
(softmax) model, the dual of the maximum-entropy problem.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_both(X, y, C=1.0, seed=0):
    """Return (logistic, perceptron): two pipelines that standardize the features first."""
    # @blank models
    # zh: 用 make_pipeline 分别把 StandardScaler() 接上 LogisticRegression(C=C) 和 Perceptron(random_state=seed)，并各自 fit
    # en: With make_pipeline, chain StandardScaler() with LogisticRegression(C=C) and with Perceptron(random_state=seed); fit both
    logistic = make_pipeline(StandardScaler(), LogisticRegression(C=C)).fit(X, y)
    perceptron = make_pipeline(StandardScaler(), Perceptron(random_state=seed)).fit(X, y)
    # @end
    return logistic, perceptron


def risk_scores(model, X):
    """The predicted probability of the positive class (the column for label 1)."""
    # @blank proba
    # zh: predict_proba 返回每一类的概率，取第 1 列
    # en: predict_proba returns one column per class; take column 1
    return model.predict_proba(X)[:, 1]
    # @end


def softmax(scores):
    """Row-wise softmax of an (n, K) score matrix, stable for large scores."""
    # @blank softmax
    # zh: 每行先减去最大值，再取指数并按行归一化
    # en: Subtract each row's maximum, exponentiate, then normalize each row
    z = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
    # @end
