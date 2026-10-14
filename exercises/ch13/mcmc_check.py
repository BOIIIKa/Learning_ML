"""Checks for mcmc.py. Runs identically under CPython and Pyodide."""

import numpy as np

from learnml import Grid, Trace


def themed_corpus(seed=0, D=20, L=30):
    """Words 0–3 medical, 4–7 financial; document d has a share (0.9, 0.1 or 0.5) of medical words."""
    rng = np.random.default_rng(seed)
    docs, share = [], []
    for d in range(D):
        m = (0.9, 0.1, 0.5)[d % 3]
        docs.append([int(rng.integers(4)) if rng.uniform() < m else int(4 + rng.integers(4)) for _ in range(L)])
        share.append(m)
    return docs, np.array(share)


def bumps(x):
    """Unnormalized log density of an equal mixture of N((−1, −0.5), 0.3 I) and N((1.2, 0.8), 0.3 I)."""
    a = -np.sum((x - [-1, -0.5]) ** 2) / 0.6
    b = -np.sum((x - [1.2, 0.8]) ** 2) / 0.6
    return np.logaddexp(a, b)


def check(s, t):
    def t_normal():
        xs, acc = s.metropolis(lambda x: -0.5 * float(x @ x), [3.0], 2.4, 20000, seed=1)
        xs = xs[1000:, 0]
        assert abs(xs.mean()) < 0.1 and abs(xs.var() - 1) < 0.15, f"mean {xs.mean():.3f}, variance {xs.var():.3f}; the target is N(0, 1)"
        assert 0.3 < acc < 0.6, f"acceptance {acc:.2f} with step 2.4 should be about 0.44"

    def t_tradeoff():
        _, small = s.metropolis(lambda x: -0.5 * float(x @ x), [0.0], 0.1, 3000)
        _, big = s.metropolis(lambda x: -0.5 * float(x @ x), [0.0], 30.0, 3000)
        assert small > 0.9 and big < 0.1, f"acceptance {small:.2f} (step 0.1) and {big:.2f} (step 30): tiny steps are almost always accepted, huge ones almost never"

    def t_pagerank():
        r = s.pagerank([[1, 2], [2], [0]])
        assert np.allclose(r, np.array([686, 380, 703]) / 1769, atol=1e-9), f"r = {np.round(r, 4)}; the notes get (686, 380, 703)/1769"
        links = [[1, 2], [], [0, 1], [2]]
        r = s.pagerank(links, d=0.8, iters=200)
        n = len(links)
        M = np.zeros((n, n))
        for j, out in enumerate(links):
            M[out if out else list(range(n)), j] = 1 / (len(out) or n)
        G = 0.2 / n + 0.8 * M
        w, v = np.linalg.eig(G)
        p = np.real(v[:, np.argmax(np.real(w))])
        assert np.allclose(r, p / p.sum(), atol=1e-9), "with a dangling page, r should still be the eigenvector of G for eigenvalue 1"

    def t_lda():
        docs, share = themed_corpus()
        _, ndk, nkw = s.lda_gibbs(docs, 2, 8, 40, seed=0)
        assert nkw.sum() == sum(len(d) for d in docs) and np.all(ndk.sum(axis=1) == [len(d) for d in docs]), "counts must stay consistent with the corpus"
        med = int(np.argmax(nkw[:, :4].sum(axis=1)))
        purity = (nkw[med, :4].sum() + nkw[1 - med, 4:].sum()) / nkw.sum()
        assert purity > 0.85, f"topic purity {purity:.2f}; one topic should hold the medical words, the other the financial ones"
        theta = (ndk[:, med] + 0.5) / (ndk.sum(axis=1) + 1)
        assert np.corrcoef(theta, share)[0, 1] > 0.9, "each document's medical-topic share should track its true make-up"

    t.test("N(0, 1) 的均值与方差", "Mean and variance of N(0, 1)", t_normal)
    t.test("步长与接受率的权衡", "Step size versus acceptance", t_tradeoff)
    t.test("PageRank：例题与悬挂页面", "PageRank: worked example and a dangling page", t_pagerank)
    t.test("LDA：话题自己分开", "LDA: topics separate on their own", t_lda)

    def trace():
        rng = np.random.default_rng(0)
        lab = rng.integers(2, size=300)
        X = np.where(lab[:, None] == 1, [1.2, 0.8], [-1, -0.5]) + np.sqrt(0.3) * rng.normal(size=(300, 2))
        grid = Grid(X, n=40)
        tr = Trace("classifier2d", X, 2 * lab - 1, grid=grid, show_margin=False,
                   title={"zh": "你的链访问过的地方（颜色越暖越常去）", "en": "Where your chain has been (warmer = more often)"})
        xs, _ = s.metropolis(bumps, [-2.5, 2.0], 0.9, 4000, seed=3)
        for n in (20, 100, 400, 1500, 4000):
            # A Gaussian kernel density of the first n samples; it factorizes over the two axes.
            Ex = np.exp(-(grid.xs[:, None] - xs[None, :n, 0]) ** 2 / 0.18)
            Ey = np.exp(-(grid.ys[:, None] - xs[None, :n, 1]) ** 2 / 0.18)
            dens = (Ey @ Ex.T).ravel()
            moved = float(np.mean(np.any(xs[1:n] != xs[:n - 1], axis=1)))
            tr.record(field=2 * dens / dens.max() - 1, metrics={"acc": moved}, note=f"{n} steps")
        return tr

    t.trace(trace)
