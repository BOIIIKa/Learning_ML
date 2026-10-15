#!/usr/bin/env python3
"""Precompute the two chapter-18 visuals that are too slow to train in the browser.

    python3 scripts/precompute_ch18.py grokking     # ≈ 5 min on 4 cores
    python3 scripts/precompute_ch18.py landscape    # ≈ 1 min

Writes site/src/engines/precomputed/{grokking,landscape}.json, which the widgets replay and
site/src/engines/dl-generalization.test.ts checks (shapes and every claim the chapter makes).

grokking: modular addition a + b (mod 31) learned by a one-hidden-layer ReLU network from half of
the 961 pairs, full-batch AdamW, for several weight-decay strengths.
landscape: two-moons classifiers; the loss on a 2-D slice through the trained weights along two
random filter-normalised directions (Li et al. 2018), for a plain vs residual 20-layer network and
for small-batch vs full-batch training.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "site" / "src" / "engines" / "precomputed"

# ---------------------------------------------------------------- grokking

P_MOD, HIDDEN, FRAC, LR, STEPS = 31, 256, 0.5, 3e-3, 20000
WDS = [0.0, 0.1, 0.3, 1.0, 3.0]
LOG_STEPS = sorted({int(round(v)) for v in np.geomspace(1, STEPS, 90)})


def grok(wd: float, seed: int = 0):
    p = P_MOD
    r = np.random.default_rng(seed)
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a, b = a.ravel(), b.ravel()
    y = (a + b) % p
    idx = r.permutation(p * p)
    tr, te = idx[: int(FRAC * p * p)], idx[int(FRAC * p * p):]
    X = np.zeros((p * p, 2 * p))
    X[np.arange(p * p), a] = 1
    X[np.arange(p * p), p + b] = 1
    W1 = r.normal(0, np.sqrt(1 / 2), (HIDDEN, 2 * p))   # each input row has two ones
    W2 = r.normal(0, np.sqrt(1 / HIDDEN), (p, HIDDEN))
    params = [W1, W2]
    m = [np.zeros_like(q) for q in params]
    v = [np.zeros_like(q) for q in params]
    Xt, yt = X[tr], y[tr]
    rec = {"step": [], "train": [], "test": [], "norm": []}

    def acc(ii):
        return float(np.mean((np.maximum(X[ii] @ params[0].T, 0) @ params[1].T).argmax(1) == y[ii]))

    for s in range(1, STEPS + 1):
        z = Xt @ params[0].T
        h = np.maximum(z, 0)
        lo = h @ params[1].T
        lo -= lo.max(1, keepdims=True)
        pr = np.exp(lo)
        pr /= pr.sum(1, keepdims=True)
        pr[np.arange(len(yt)), yt] -= 1
        d = pr / len(yt)
        g2 = d.T @ h
        g1 = ((d @ params[1]) * (z > 0)).T @ Xt
        for i, g in enumerate([g1, g2]):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.98 * v[i] + 0.02 * g * g
            # AdamW: the decay is decoupled from the adaptive step.
            params[i] = params[i] - LR * ((m[i] / (1 - 0.9 ** s)) / (np.sqrt(v[i] / (1 - 0.98 ** s)) + 1e-8) + wd * params[i])
        if s in LOG_STEPS:
            rec["step"].append(s)
            rec["train"].append(round(acc(tr), 4))
            rec["test"].append(round(acc(te), 4))
            rec["norm"].append(round(float(np.sqrt(sum((q * q).sum() for q in params))), 3))
    return rec


def grokking():
    with ProcessPoolExecutor() as ex:
        runs = list(ex.map(grok, WDS))
        seeds = list(ex.map(grok, [1.0, 1.0, 0.0, 0.0], [1, 2, 1, 2]))
    data = {
        "p": P_MOD, "hidden": HIDDEN, "frac": FRAC, "lr": LR, "steps": STEPS,
        "nTrain": int(FRAC * P_MOD * P_MOD), "wds": WDS, "runs": runs,
        # Extra seeds for the robustness checks in the tests (wd = 1 groks, wd = 0 does not).
        "seeds": {"wd1": seeds[:2], "wd0": seeds[2:]},
    }
    (OUT / "grokking.json").write_text(json.dumps(data, separators=(",", ":")))


# ---------------------------------------------------------------- landscape

def moons():
    from sklearn.datasets import make_moons
    X, y = make_moons(200, noise=0.2, random_state=0)
    Xt, yt = make_moons(1000, noise=0.2, random_state=1)
    return X, y.astype(float), Xt, yt.astype(float)


def init(depth, width, seed, res):
    r = np.random.default_rng(seed)
    P = [[r.normal(0, 1.0, (width, 2)), np.zeros(width)]]
    for _ in range(depth):
        # He init; residual branches start scaled down by 4 so the sum does not blow up with depth.
        P.append([r.normal(0, np.sqrt(2 / width) * (0.25 if res else 1), (width, width)), np.zeros(width)])
    P.append([r.normal(0, np.sqrt(1 / width), (1, width)), np.zeros(1)])
    return P


def fwd(P, X, res, cache=False):
    z0 = X @ P[0][0].T + P[0][1]
    h = np.maximum(z0, 0)
    C = [(X, z0)]
    for W, b in P[1:-1]:
        z = h @ W.T + b
        a = np.maximum(z, 0)
        C.append((h, z))
        h = h + a if res else a
    C.append((h, None))
    f = (h @ P[-1][0].T + P[-1][1])[:, 0]
    return (f, C) if cache else f


def bce(P, X, y, res):
    f = fwd(P, X, res)
    return float(np.mean(np.logaddexp(0, f) - y * f))


def grad(P, X, y, res):
    f, C = fwd(P, X, res, True)
    d = ((1 / (1 + np.exp(-f)) - y) / len(y))[:, None]
    G = [None] * len(P)
    G[-1] = [d.T @ C[-1][0], d.sum(0)]
    dh = d @ P[-1][0]
    for l in range(len(P) - 2, 0, -1):
        hin, z = C[l]
        dz = dh * (z > 0)
        G[l] = [dz.T @ hin, dz.sum(0)]
        dh = (dh if res else 0) + dz @ P[l][0]
    hin, z = C[0]
    dz = dh * (z > 0)
    G[0] = [dz.T @ hin, dz.sum(0)]
    return G


def train(X, y, depth, res, batch, epochs, lr, seed):
    P = init(depth, 16, seed, res)
    r = np.random.default_rng(seed + 1)
    V = [[np.zeros_like(a) for a in p] for p in P]
    n = len(y)
    for _ in range(epochs):
        order = r.permutation(n)
        for s in range(0, n - batch + 1, batch):
            i = order[s:s + batch]
            for p, g, v in zip(P, grad(P, X[i], y[i], res), V):
                for k in range(2):
                    v[k] = 0.9 * v[k] - lr * g[k]
                    p[k] += v[k]
    return P


def direction(P, r):
    """A Gaussian direction with every neuron's incoming weights rescaled to that neuron's norm (filter normalisation)."""
    out = []
    for W, b in P:
        D = r.normal(size=W.shape)
        D *= np.linalg.norm(W, axis=1, keepdims=True) / np.linalg.norm(D, axis=1, keepdims=True)
        out.append([D, np.zeros_like(b)])
    return out


K, R = 25, 1.0
CONFIGS = {
    "plain": dict(depth=20, res=False, batch=20, epochs=60, lr=0.02),
    "residual": dict(depth=20, res=True, batch=20, epochs=60, lr=0.02),
    "small": dict(depth=4, res=False, batch=10, epochs=100, lr=0.02),
    "large": dict(depth=4, res=False, batch=200, epochs=1000, lr=0.2),
}


def landscape_one(name, seed):
    X, y, Xt, yt = moons()
    c = CONFIGS[name]
    P = train(X, y, seed=seed, **c)
    r = np.random.default_rng(100 + seed)
    d1, d2 = direction(P, r), direction(P, r)
    a = np.linspace(-R, R, K)
    S = np.zeros((K, K))
    for i, al in enumerate(a):
        for j, be in enumerate(a):
            Q = [[p[k] + al * u[k] + be * v[k] for k in range(2)] for p, u, v in zip(P, d1, d2)]
            S[j, i] = bce(Q, X, y, c["res"])
    return {
        "loss": [[round(float(v), 4) for v in row] for row in S],
        "trainAcc": round(float(np.mean((fwd(P, X, c["res"]) > 0) == (y == 1))), 4),
        "testAcc": round(float(np.mean((fwd(P, Xt, c["res"]) > 0) == (yt == 1))), 4),
    }


def landscape():
    names = list(CONFIGS)
    jobs = [(n, s) for n in names for s in (0, 1, 2)]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(landscape_one, *zip(*jobs)))
    data = {"K": K, "R": R, "configs": CONFIGS, "runs": {n: [res[i] for i, (m, _) in enumerate(jobs) if m == n] for n in names}}
    (OUT / "landscape.json").write_text(json.dumps(data, separators=(",", ":")))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for what in sys.argv[1:] or ["grokking", "landscape"]:
        {"grokking": grokking, "landscape": landscape}[what]()
        print("wrote", OUT / f"{what}.json")
