"""Generative vs discriminative with scikit-learn: LDA, QDA, Gaussian naive Bayes and logistic regression,
their learning curves, and a multinomial naive Bayes text classifier with Laplace smoothing.
"""

from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline


def build_models():
    """Unfitted classifiers keyed by name: "LDA", "QDA", "NB" (Gaussian) and "LR" (almost unregularized)."""
    # @blank models
    # zh: 返回字典：LinearDiscriminantAnalysis()、QuadraticDiscriminantAnalysis()、GaussianNB()、LogisticRegression(C=100, max_iter=2000)
    # en: Return a dict: LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(), GaussianNB(), LogisticRegression(C=100, max_iter=2000)
    return {
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "NB": GaussianNB(),
        "LR": LogisticRegression(C=100, max_iter=2000),
    }
    # @end


def learning_curve(model, X_train, y_train, X_test, y_test, sizes):
    """Test error of a fresh copy of `model` trained on the first m training samples, for each m in `sizes`."""
    # @blank curve
    # zh: 对每个 m：clone(model) 后在 X_train[:m], y_train[:m] 上 fit，记录 1 − score(X_test, y_test)
    # en: For each m: clone(model), fit it on X_train[:m], y_train[:m], and record 1 − score(X_test, y_test)
    return [1.0 - clone(model).fit(X_train[:m], y_train[:m]).score(X_test, y_test) for m in sizes]
    # @end


def message_classifier(alpha=1.0):
    """Word counts followed by multinomial naive Bayes with additive (Laplace) smoothing α."""
    # @blank text
    # zh: 用 make_pipeline 串起 CountVectorizer() 与 MultinomialNB(alpha=alpha)
    # en: Chain CountVectorizer() and MultinomialNB(alpha=alpha) with make_pipeline
    return make_pipeline(CountVectorizer(), MultinomialNB(alpha=alpha))
    # @end
