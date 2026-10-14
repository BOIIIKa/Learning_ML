"""Assemble an RBF-kernel SVM with scikit-learn and tune it by cross-validation.

Features of the breast-cancer data live on very different scales, so the kernel
needs standardized inputs; C and γ are then chosen by grid search.
"""

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_model(C=1.0, gamma="scale"):
    """A pipeline: standardize the features, then an RBF-kernel SVC."""
    # @blank pipeline
    # zh: 用 make_pipeline 把 StandardScaler() 与 SVC(kernel="rbf", C=C, gamma=gamma) 串起来
    # en: Chain StandardScaler() and SVC(kernel="rbf", C=C, gamma=gamma) with make_pipeline
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma=gamma))
    # @end


def tune(X, y):
    """5-fold grid search over C and γ; returns the fitted GridSearchCV object."""
    # @blank grid
    # zh: 参数名形如 "svc__C"；在 C ∈ {0.1, 1, 10, 100}、γ ∈ {0.001, 0.01, 0.1} 上做 5 折网格搜索并 fit
    # en: Parameter names look like "svc__C"; grid-search C ∈ {0.1, 1, 10, 100}, γ ∈ {0.001, 0.01, 0.1} with 5 folds, then fit
    grid = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.001, 0.01, 0.1]}
    search = GridSearchCV(build_model(), grid, cv=5)
    search.fit(X, y)
    # @end
    return search
