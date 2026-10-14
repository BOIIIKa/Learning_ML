"""Checks for hmm.py. Runs identically under CPython and Pyodide."""

import itertools

import numpy as np

from learnml import Trace

PI = np.array([0.6, 0.4])
A = np.array([[0.7, 0.3], [0.4, 0.6]])
B = np.array([[0.9, 0.1], [0.2, 0.8]])


def sample(T, seed):
    rng = np.random.default_rng(seed)
    S, O = [rng.choice(2, p=PI)], []
    for t in range(T):
        if t:
            S.append(rng.choice(2, p=A[S[-1]]))
        O.append(rng.choice(2, p=B[S[-1]]))
    return np.array(S), np.array(O)


def brute(O):
    best, bp, tot = None, -1, 0.0
    for s in itertools.product(range(2), repeat=len(O)):
        p = PI[s[0]] * B[s[0], O[0]] * np.prod([A[s[t - 1], s[t]] * B[s[t], O[t]] for t in range(1, len(O))])
        tot += p
        if p > bp:
            best, bp = list(s), p
    return tot, best, bp


def check(s, t):
    def t_example():
        _, logp = s.forward(PI, A, B, [0, 1, 1])
        assert abs(np.exp(logp) - 0.10007) < 1e-12, f"P(O) = {np.exp(logp):.6f}; the notes get 0.10007"
        path, lp = s.viterbi(PI, A, B, [0, 1, 1])
        assert path == [0, 1, 1] and abs(np.exp(lp) - 0.062208) < 1e-12, f"path {path}, prob {np.exp(lp):.6f}"

    def t_brute():
        _, O = sample(9, 3)
        tot, best, bp = brute(O)
        _, logp = s.forward(PI, A, B, O)
        assert abs(logp - np.log(tot)) < 1e-9, "forward must equal the brute-force sum over all paths"
        path, lp = s.viterbi(PI, A, B, O)
        assert path == best and abs(lp - np.log(bp)) < 1e-9, "Viterbi must find the brute-force best path"

    def t_long():
        _, O = sample(2000, 5)
        _, logp = s.forward(PI, A, B, O)
        assert np.isfinite(logp) and logp < 0, "scaling must keep a 2000-step sequence finite"

    def t_em():
        _, O = sample(500, 7)
        p, a, b = np.array([0.5, 0.5]), np.array([[0.6, 0.4], [0.5, 0.5]]), np.array([[0.7, 0.3], [0.4, 0.6]])
        lls = []
        for _ in range(30):
            p, a, b, ll = s.baum_welch_step(p, a, b, O)
            lls.append(ll)
        assert all(y >= x - 1e-8 for x, y in zip(lls, lls[1:])), "EM must never decrease the log-likelihood"
        assert np.allclose(a.sum(axis=1), 1) and np.allclose(b.sum(axis=1), 1), "rows of A and B must sum to 1"
        assert abs(b[0, 0] - 0.9) < 0.12 and abs(b[1, 1] - 0.8) < 0.12, f"emissions {np.round(b, 2)} should approach the truth"

    t.test("例题：P(O) 与维特比路径", "Worked example: P(O) and the Viterbi path", t_example)
    t.test("与穷举一致", "Matches brute-force enumeration", t_brute)
    t.test("长序列不下溢", "No underflow on long sequences", t_long)
    t.test("Baum–Welch 单调上升", "Baum–Welch increases the likelihood", t_em)

    def trace():
        S, O = sample(60, 11)
        X = np.column_stack([np.arange(60), O + 0.15 * (S - 0.5)])
        tr = Trace("classifier2d", X, 2 * S - 1, show_margin=False,
                   title={"zh": "EM 迭代中的维特比解码（黄圈 = 判为恶化）", "en": "Viterbi decoding during EM (ring = decoded as worse)"})
        p, a, b = np.array([0.5, 0.5]), np.array([[0.6, 0.4], [0.5, 0.5]]), np.array([[0.7, 0.3], [0.4, 0.6]])
        for k in range(10):
            path, _ = s.viterbi(p, a, b, O)
            _, ll = s.forward(p, a, b, O)
            tr.record(highlight=[i for i, z in enumerate(path) if z == 1], metrics={"loglik": ll}, note=f"EM {k}")
            p, a, b, _ = s.baum_welch_step(p, a, b, O)
        return tr

    t.trace(trace)
