"""Logistic regression the way practitioners use it in scikit-learn.

Part 1: a softmax (multinomial) model on the wine data (178 wines, 13 measurements, 3 cultivars),
with the regularization strength C chosen by cross-validation.
Part 2: the same linear model trained by SGD with different losses, the "loss zoo" of the chapter.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOSSES = ["perceptron", "hinge", "log_loss", "modified_huber"]


def build_cv():
    """Pipeline: StandardScaler → LogisticRegressionCV(Cs=np.logspace(−3, 3, 13), cv=5, max_iter=2000)."""
    # @blank cv
    # zh: 用 make_pipeline 组装；C 的候选网格是 np.logspace(-3, 3, 13)
    # en: Assemble with make_pipeline; the grid of C values is np.logspace(-3, 3, 13)
    return make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-3, 3, 13), cv=5, max_iter=2000))
    # @end


def confidence(model, X):
    """For each row: the predicted probability of the predicted class."""
    # @blank proba
    # zh: predict_proba 每行是各类概率（和为 1），取每行最大值
    # en: Each row of predict_proba holds the class probabilities (summing to 1); take the row maximum
    return model.predict_proba(X).max(axis=1)
    # @end


def c_path(X, y, Cs):
    """Coefficient vector of a binary LogisticRegression(C=C, max_iter=2000) for each C in Cs."""
    # @blank path
    # zh: C 越大正则越弱；每个 C 拟合一次，取 coef_[0]
    # en: Larger C means weaker regularization; fit once per C and keep coef_[0]
    return [LogisticRegression(C=C, max_iter=2000).fit(X, y).coef_[0] for C in Cs]
    # @end


def sgd_models():
    """Dict loss → SGDClassifier(loss=loss, alpha=1e-3, average=True, max_iter=2000, tol=1e-6, random_state=0), for every loss in LOSSES.

    average=True returns the average of all SGD iterates (for loss="perceptron": the averaged perceptron),
    which is far more stable than the last iterate when the classes overlap.
    """
    # @blank sgd
    # zh: 用字典推导式，一个损失对应一个模型；别忘了 average=True
    # en: A dict comprehension, one model per loss; don't forget average=True
    return {loss: SGDClassifier(loss=loss, alpha=1e-3, average=True, max_iter=2000, tol=1e-6, random_state=0) for loss in LOSSES}
    # @end
