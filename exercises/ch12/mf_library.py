"""Matrix factorization in scikit-learn: PCA on handwritten digits, then LSA and NMF on a tiny corpus."""

import numpy as np
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def components_for(X, frac=0.9):
    """Smallest number of PCA components whose cumulative explained variance ratio reaches `frac`."""
    # @blank ncomp
    # zh: 拟合完整的 PCA，对 explained_variance_ratio_ 做累加，找第一个 ≥ frac 的位置（加一）
    # en: Fit a full PCA, cumulatively sum explained_variance_ratio_, find the first position ≥ frac (plus one)
    cum = np.cumsum(PCA().fit(X).explained_variance_ratio_)
    return int(np.searchsorted(cum, frac) + 1)
    # @end


def lsa(docs, k=2):
    """TF-IDF (stop_words="english") → TruncatedSVD(k, random_state=0). Returns (vectorizer, svd, document coordinates)."""
    # @blank lsa
    # zh: 先 fit_transform 得到 TF-IDF 矩阵，再用 TruncatedSVD 降到 k 维
    # en: fit_transform to the TF-IDF matrix, then reduce to k dimensions with TruncatedSVD
    vec = TfidfVectorizer(stop_words="english")
    A = vec.fit_transform(docs)
    svd = TruncatedSVD(k, random_state=0)
    return vec, svd, svd.fit_transform(A)
    # @end


def nmf_topics(docs, k=2, top=3):
    """NMF(k, init="nndsvda", random_state=0, max_iter=500) on TF-IDF; return the top words of each topic."""
    # @blank nmf
    # zh: components_ 的每一行是一个话题在各词上的权重；取最大的 top 个词
    # en: Each row of components_ is a topic's weight on each word; take the top words
    vec = TfidfVectorizer(stop_words="english")
    A = vec.fit_transform(docs)
    H = NMF(k, init="nndsvda", random_state=0, max_iter=500).fit(A).components_
    vocab = vec.get_feature_names_out()
    return [[str(vocab[i]) for i in np.argsort(h)[::-1][:top]] for h in H]
    # @end
