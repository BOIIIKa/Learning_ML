"""Checks for library.py. Runs identically under CPython and Pyodide."""

import numpy as np

from learnml import Trace


def check(s, t):
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 60)
    y = 1 + 2 * x + rng.normal(0, 0.2, 60)

    def t_fit():
        m = s.build_model().fit(x[:, None], y)
        assert abs(m.coef_[0] - 2) < 0.2, f"slope {m.coef_[0]:.2f} should be close to 2"

    t.test("组装并训练", "Assemble and fit", t_fit)

    def trace():
        tr = Trace("classifier2d", np.column_stack([x, y]), np.ones(len(y)), show_margin=False,
                   title={"zh": "TODO(scaffold)", "en": "TODO(scaffold)"})
        for k in range(5, 61, 5):
            m = s.build_model().fit(x[:k, None], y[:k])
            tr.record(w=[m.coef_[0], -1.0], b=m.intercept_, metrics={"loss": float(np.mean((m.predict(x[:, None]) - y) ** 2))}, note=f"n = {k}")
        return tr

    t.trace(trace)
