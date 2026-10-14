"""Generative classifiers in scikit-learn: discriminant analysis on tabular data, naive Bayes on text.

Part 1 compares LDA, QDA, Gaussian naive Bayes and logistic regression by cross-validation.
Part 2 assembles the standard text pipeline: word counts → multinomial naive Bayes.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def models():
    """Dict name → estimator: "LDA" (shrinkage="auto", solver="lsqr"), "QDA" (reg_param=0.1), "GaussianNB",
    and "Logistic" (StandardScaler → LogisticRegression(max_iter=2000))."""
    # @blank models
    # zh: 四个模型放进一个字典；逻辑回归要先标准化
    # en: Put the four models in a dict; logistic regression needs standardization first
    return {
        "LDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.1),
        "GaussianNB": GaussianNB(),
        "Logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    }
    # @end


def cv_scores(ms, X, y):
    """Dict name → mean 5-fold cross-validated accuracy."""
    # @blank cv
    # zh: 对每个模型调用 cross_val_score(m, X, y, cv=5) 并取平均
    # en: Call cross_val_score(m, X, y, cv=5) for each model and take the mean
    return {name: float(np.mean(cross_val_score(m, X, y, cv=5))) for name, m in ms.items()}
    # @end


def text_model(alpha=1.0):
    """Pipeline: CountVectorizer(stop_words="english") → MultinomialNB(alpha=alpha)."""
    # @blank text
    # zh: 词频向量化（去掉 and、of 这类停用词）+ 多项式朴素贝叶斯；alpha 就是拉普拉斯平滑
    # en: Word counts (dropping stop words such as "and", "of") + multinomial naive Bayes; alpha is the Laplace smoothing
    return make_pipeline(CountVectorizer(stop_words="english"), MultinomialNB(alpha=alpha))
    # @end


def top_words(pipe, k=3):
    """For class 1: the k words with the largest log P(word | 1) − log P(word | 0)."""
    # @blank words
    # zh: vocab = pipe[0].get_feature_names_out()；对数概率在 pipe[-1].feature_log_prob_，取差值最大的 k 个
    # en: vocab = pipe[0].get_feature_names_out(); log-probabilities are in pipe[-1].feature_log_prob_; take the k largest differences
    vocab = pipe[0].get_feature_names_out()
    lp = pipe[-1].feature_log_prob_
    return [str(vocab[i]) for i in np.argsort(lp[1] - lp[0])[::-1][:k]]
    # @end
