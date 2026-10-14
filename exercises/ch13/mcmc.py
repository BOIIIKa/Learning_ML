"""Markov chain methods from scratch: random-walk Metropolis, PageRank by power iteration, and a collapsed Gibbs sampler for LDA."""

import numpy as np


def metropolis(logp, x0, step, n, seed=0):
    """Random-walk Metropolis with N(0, step² I) proposals. Returns (samples (n, dim), acceptance rate)."""
    rng = np.random.default_rng(seed)
    x = np.atleast_1d(np.asarray(x0, dtype=float))
    lp = logp(x)
    out, acc = np.empty((n, x.size)), 0
    for t in range(n):
        y = x + step * rng.normal(size=x.size)
        # @blank accept
        # zh: 在对数尺度上比较：log u < logp(y) − logp(x) 就接受（更新 x、lp 与 acc）
        # en: Compare on the log scale: accept if log u < logp(y) − logp(x) (update x, lp and acc)
        ly = logp(y)
        if np.log(rng.uniform()) < ly - lp:
            x, lp, acc = y, ly, acc + 1
        # @end
        out[t] = x
    return out, acc / n


def pagerank(links, d=0.85, iters=100):
    """PageRank of pages 0..n−1 given out-link lists; a page without out-links links to every page."""
    n = len(links)
    r = np.full(n, 1 / n)
    for _ in range(iters):
        # @blank pagerank
        # zh: new = (1 − d)/n；每个页面 j 把 d·r[j] 平均分给它的外链（没有外链就分给所有页面）
        # en: new = (1 − d)/n; each page j splits d·r[j] evenly among its out-links (or among all pages if it has none)
        new = np.full(n, (1 - d) / n)
        for j, out in enumerate(links):
            if out:
                new[out] += d * r[j] / len(out)
            else:
                new += d * r[j] / n
        r = new
        # @end
    return r


def lda_gibbs(docs, K, V, sweeps, alpha=0.5, beta=0.1, seed=0, trace=None):
    """Collapsed Gibbs sampling for LDA. docs: lists of word ids. Returns (z, ndk (D, K), nkw (K, V))."""
    rng = np.random.default_rng(seed)
    z = [rng.integers(K, size=len(doc)) for doc in docs]
    ndk = np.zeros((len(docs), K))
    nkw = np.zeros((K, V))
    for d, doc in enumerate(docs):
        for w, k in zip(doc, z[d]):
            ndk[d, k] += 1
            nkw[k, w] += 1
    nk = nkw.sum(axis=1)
    for s in range(sweeps):
        for d, doc in enumerate(docs):
            for i, w in enumerate(doc):
                k = z[d][i]
                ndk[d, k] -= 1; nkw[k, w] -= 1; nk[k] -= 1
                # @blank conditional
                # zh: p ∝ (ndk[d] + α)(nkw[:, w] + β)/(nk + Vβ)，归一化后用 rng.choice 抽新话题
                # en: p ∝ (ndk[d] + α)(nkw[:, w] + β)/(nk + Vβ); normalize and draw the new topic with rng.choice
                p = (ndk[d] + alpha) * (nkw[:, w] + beta) / (nk + V * beta)
                k = rng.choice(K, p=p / p.sum())
                # @end
                z[d][i] = k
                ndk[d, k] += 1; nkw[k, w] += 1; nk[k] += 1
        if trace is not None:
            trace(s, ndk, nkw)
    return z, ndk, nkw
