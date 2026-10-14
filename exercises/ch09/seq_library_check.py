"""Checks for seq_library.py. Runs identically under CPython and Pyodide."""

import numpy as np

from learnml import Trace


def patients(T, seed):
    """A sticky two-state chain with noisy 3-d lab panels."""
    rng = np.random.default_rng(seed)
    y = [0]
    for _ in range(T - 1):
        y.append(y[-1] if rng.random() < 0.9 else 1 - y[-1])
    y = np.array(y)
    X = rng.normal(0, 1.3, (T, 3)) + np.outer(y, [1.0, 0.8, 0.6])
    return X, y


def check(s, t):
    Xtr, ytr = patients(1500, 0)
    Xte, yte = patients(800, 1)
    state = {}

    def t_window():
        X = np.arange(12.0).reshape(4, 3)
        W = s.window(X, 1)
        assert W.shape == (4, 9), f"shape {W.shape}; expected (4, 9)"
        assert np.allclose(W[0], [0, 1, 2, 0, 1, 2, 3, 4, 5]) and np.allclose(W[3], [6, 7, 8, 9, 10, 11, 9, 10, 11]), "edges are padded by repetition"

    def t_day():
        m = s.per_day_model().fit(Xtr, ytr)
        acc = m.score(Xte, yte)
        assert 0.6 < acc < 0.85, f"per-day accuracy {acc:.3f}"
        mw = s.per_day_model().fit(s.window(Xtr), ytr)
        accw = mw.score(s.window(Xte), yte)
        assert accw > acc + 0.03, f"windowed features ({accw:.3f}) should beat single days ({acc:.3f})"
        state.update(m=m, acc=acc, accw=accw)

    def t_trans():
        lA = s.transition_logprob(ytr)
        P = np.exp(lA)
        assert np.allclose(P.sum(axis=1), 1), "rows must sum to 1"
        assert P[0, 0] > 0.85 and P[1, 1] > 0.85, f"the chain is sticky: diagonal {np.round(np.diag(P), 3)}"

    def t_decode():
        m = state.get("m")
        assert m is not None, "fix the previous tests first"
        logp = np.log(m.predict_proba(Xte) + 1e-12) - np.log(np.bincount(ytr) / len(ytr))
        path = s.decode(logp, s.transition_logprob(ytr))
        acc = float(np.mean(path == yte))
        assert acc > state["acc"] + 0.05, f"structured decoding ({acc:.3f}) should clearly beat the per-day classifier ({state['acc']:.3f})"

    t.test("滑动窗口特征", "Sliding-window features", t_window)
    t.test("逐日分类器", "Per-day classifier", t_day)
    t.test("转移概率", "Transition probabilities", t_trans)
    t.test("维特比解码胜过逐日分类", "Viterbi decoding beats per-day classification", t_decode)

    def trace():
        X, y = patients(80, 3)
        m = s.per_day_model().fit(Xtr, ytr)
        P = np.column_stack([np.arange(80), m.predict_proba(X)[:, 1]])
        tr = Trace("classifier2d", P, 2 * y - 1, show_margin=False,
                   title={"zh": "纵轴：逐日分类器给出的 P(恶化)；黄圈：判为恶化", "en": "Vertical: per-day P(worse); rings: decoded as worse"})
        per_day = (P[:, 1] > 0.5).astype(int)
        logp = np.log(m.predict_proba(X) + 1e-12) - np.log(np.bincount(ytr) / len(ytr))
        vit = s.decode(logp, s.transition_logprob(ytr))
        for name, lab in (("per-day", per_day), ("Viterbi", vit)):
            tr.record(highlight=[i for i, v in enumerate(lab) if v == 1], metrics={"acc": float(np.mean(lab == y))}, note=name)
        return tr

    t.trace(trace)
