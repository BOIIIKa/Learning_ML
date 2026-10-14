"""Markov chain methods with libraries: LDA in scikit-learn, PageRank as a numpy eigenproblem, and a
Bayesian logistic-regression posterior written in numpy (sampled by a ready-made Metropolis sampler)."""

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline


def lda_pipeline(k):
    """Pipeline: CountVectorizer() → LatentDirichletAllocation(k topics, random_state=0)."""
    # @blank lda
    # zh: LDA 的输入是词频，所以用 CountVectorizer 而不是 TF-IDF
    # en: LDA takes word counts, so use CountVectorizer rather than TF-IDF
    return make_pipeline(CountVectorizer(), LatentDirichletAllocation(n_components=k, random_state=0))
    # @end


def top_words(pipe, n):
    """For each topic of a fitted pipeline, its n most heavily weighted words (a list of lists)."""
    # @blank top
    # zh: 词表在 countvectorizer.get_feature_names_out()；话题–词权重在 latentdirichletallocation.components_
    # en: The vocabulary is countvectorizer.get_feature_names_out(); topic–word weights are latentdirichletallocation.components_
    words = pipe.named_steps["countvectorizer"].get_feature_names_out()
    comps = pipe.named_steps["latentdirichletallocation"].components_
    return [[str(words[i]) for i in np.argsort(row)[::-1][:n]] for row in comps]
    # @end


def pagerank_eig(links, d=0.85):
    """PageRank as the eigenvector of G = (1 − d)/n + d M for eigenvalue 1 (dangling pages link to all), summing to 1."""
    n = len(links)
    M = np.zeros((n, n))
    for j, out in enumerate(links):
        M[out if out else list(range(n)), j] = 1 / (len(out) or n)
    # @blank eig
    # zh: np.linalg.eig(G)；取实部最大的特征值对应的特征向量，取实部再除以它的和
    # en: np.linalg.eig(G); take the eigenvector of the eigenvalue with the largest real part, its real part, divided by its sum
    G = (1 - d) / n + d * M
    w, v = np.linalg.eig(G)
    p = np.real(v[:, np.argmax(np.real(w))])
    return p / p.sum()
    # @end


def log_posterior(theta, X, y, tau=3.0):
    """Unnormalized log posterior of logistic regression, theta = (w₁, …, w_p, b), labels y ∈ {−1, +1},
    prior N(0, τ² I) on all of theta: Σ log σ(y (w·x + b)) − ‖theta‖² / (2τ²)."""
    # @blank post
    # zh: log σ(m) = −log(1 + e^{−m}) = −np.logaddexp(0, −m)，数值稳定
    # en: log σ(m) = −log(1 + e^{−m}) = −np.logaddexp(0, −m), numerically stable
    m = y * (X @ theta[:-1] + theta[-1])
    return float(-np.logaddexp(0, -m).sum() - theta @ theta / (2 * tau ** 2))
    # @end
