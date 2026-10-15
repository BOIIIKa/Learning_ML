"""Checks for regularize_library.py. Runs identically under CPython and Pyodide."""

import warnings

import numpy as np
from sklearn.datasets import make_moons
from sklearn.exceptions import ConvergenceWarning

from learnml import Grid, Trace


def check(s, t):
    X, y = make_moons(60, noise=0.35, random_state=0)
    Xv, yv = make_moons(300, noise=0.35, random_state=10)

    def quiet(fn):
        def run():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                fn()
        return run

    def t_norm():
        models = [s.fit_alpha(X, y, a) for a in (1e-4, 1.0, 10.0)]
        m = models[0]
        assert m.alpha == 1e-4 and m.hidden_layer_sizes == (64, 64), "use alpha and hidden_layer_sizes=(64, 64)"
        assert m.score(X, y) > 0.9, f"with almost no penalty the network should fit the 60 points (train accuracy {m.score(X, y):.2f})"
        manual = np.sqrt(sum((W ** 2).sum() for W in m.coefs_))
        assert abs(s.weight_norm(m) - manual) < 1e-9, "weight_norm should be the norm of all coefs_"
        norms = [s.weight_norm(q) for q in models]
        assert norms[0] > norms[1] > norms[2], f"a larger alpha should give smaller weights: {[round(v, 2) for v in norms]}"

    def t_pick():
        alphas = [1e-4, 0.1, 1.0]
        a, m = s.pick_alpha(X, y, Xv, yv, alphas)
        scores = [s.fit_alpha(X, y, b).score(Xv, yv) for b in alphas]
        best = alphas[int(np.argmax(scores))]
        assert a == best, f"validation accuracies {[round(v, 3) for v in scores]} pick alpha = {best}, got {a}"
        assert abs(m.score(Xv, yv) - max(scores)) < 1e-12, "return the model fitted with the chosen alpha"

    def t_early():
        m = s.fit_early_stopping(X, y)
        assert m.early_stopping and m.validation_fraction == 0.2 and m.n_iter_no_change == 20, "set early_stopping, validation_fraction and n_iter_no_change"
        best = int(np.argmax(m.validation_scores_)) + 1
        assert m.n_iter_ < 500, f"early stopping should end well before 500 epochs (ran {m.n_iter_})"
        assert m.n_iter_ - best >= 20, f"it stops only after 20 epochs without improvement (best epoch {best}, stopped at {m.n_iter_})"

    t.test("alpha 与权重范数", "alpha and the weight norm", quiet(t_norm))
    t.test("在验证集上选 alpha", "Choosing alpha on a validation set", quiet(t_pick))
    t.test("内置的早停", "Built-in early stopping", quiet(t_early))

    def trace():
        grid = Grid(X, n=48)
        tr = Trace("classifier2d", X, np.where(y > 0, 1, -1), grid=grid, show_margin=False,
                   title={"zh": "alpha 从小到大：分界线从扭曲到平滑，再到欠拟合", "en": "Growing alpha: the boundary goes from twisted to smooth to underfit"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for a in (1e-4, 0.1, 1.0, 10.0):
                m = s.fit_alpha(X, y, a)
                p = m.predict_proba(grid.points)[:, 1]
                tr.record(field=np.log(p + 1e-12) - np.log(1 - p + 1e-12), metrics={"val": 1 - m.score(Xv, yv)}, note=f"alpha = {a:g}")
        return tr

    t.trace(trace)
