"""Checks for smo.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.svm import SVC

from learnml import Trace

C = 1.0


def dataset(seed=7, n=40):
    """Two slightly overlapping Gaussian blobs, labels in {−1, +1}."""
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal([-1.2, -0.8], 0.9, (n // 2, 2)), rng.normal([1.2, 0.9], 0.9, (n // 2, 2))])
    y = np.array([-1.0] * (n // 2) + [1.0] * (n // 2))
    return X, y


def check(s, t):
    def t_bounds():
        assert np.allclose(s.bounds(0.3, 0.5, 1, -1, 1.0), (0.2, 1.0)), "y_i ≠ y_j"
        assert np.allclose(s.bounds(0.3, 0.5, 1, 1, 1.0), (0.0, 0.8)), "y_i = y_j"
        assert np.allclose(s.bounds(0.9, 0.6, -1, -1, 1.0), (0.5, 1.0)), "y_i = y_j, a_i + a_j > C"

    def t_eta():
        K = np.array([[2.0, 0.5], [0.5, 1.0]])
        assert np.isclose(s.eta(K, 0, 1), 2.0), f"η = {s.eta(K, 0, 1)}, expected 2.0"

    def t_alpha_j():
        assert np.isclose(s.new_alpha_j(0.2, 1, 0.5, -0.1, 2.0, 0.0, 1.0), 0.5), "unclipped case"
        assert np.isclose(s.new_alpha_j(0.2, 1, 3.0, -1.0, 2.0, 0.0, 1.0), 1.0), "should clip to H"
        assert np.isclose(s.new_alpha_j(0.2, -1, 3.0, -1.0, 2.0, 0.0, 1.0), 0.0), "should clip to L"

    def t_alpha_i():
        assert np.isclose(s.new_alpha_i(0.4, 0.2, 0.5, 1, -1), 0.7), "y_i ≠ y_j"
        assert np.isclose(s.new_alpha_i(0.4, 0.2, 0.5, 1, 1), 0.1), "y_i = y_j"

    def t_b():
        K = np.eye(2)
        args = dict(b=0.0, E_i=0.2, E_j=-0.4, y_i=1, y_j=-1, d_i=0.1, d_j=0.1, K=K, i=0, j=1, C=1.0)
        assert np.isclose(s.new_b(**args, a_i_new=0.5, a_j_new=0.0), -0.3), "α_i free ⇒ b1"
        assert np.isclose(s.new_b(**args, a_i_new=1.0, a_j_new=0.5), 0.5), "α_j free ⇒ b2"
        assert np.isclose(s.new_b(**args, a_i_new=0.0, a_j_new=1.0), 0.1), "both at bounds ⇒ average"

    def t_end_to_end():
        X, y = dataset()
        alpha, b = s.smo(X, y, C=C)
        w = (alpha * y) @ X
        ref = SVC(kernel="linear", C=C, tol=1e-6).fit(X, y)
        assert np.allclose(w, ref.coef_[0], atol=0.03), f"w = {np.round(w, 3)}, sklearn = {np.round(ref.coef_[0], 3)}"
        assert abs(b - ref.intercept_[0]) < 0.05, f"b = {b:.3f}, sklearn = {ref.intercept_[0]:.3f}"
        assert abs(alpha @ y) < 1e-6, "Σ α_i y_i should stay 0"

    t.test("可行区间 L、H", "Feasible interval L, H", t_bounds)
    t.test("η 的计算", "Computing η", t_eta)
    t.test("α_j 的更新与裁剪", "Updating and clipping α_j", t_alpha_j)
    t.test("α_i 的更新", "Updating α_i", t_alpha_i)
    t.test("阈值 b 的更新", "Updating the threshold b", t_b)
    t.test("与 scikit-learn 的解一致", "Matches scikit-learn's solution", t_end_to_end)

    def student_trace():
        X, y = dataset()
        trace = Trace("classifier2d", X, y, title={"zh": "你的 SMO", "en": "Your SMO"})
        s.smo(X, y, C=C, trace=trace)
        return trace

    def library_trace():
        X, y = dataset()
        ref = SVC(kernel="linear", C=C, tol=1e-6).fit(X, y)
        trace = Trace("classifier2d", X, y, title={"zh": "scikit-learn（libsvm）", "en": "scikit-learn (libsvm)"})
        trace.record(w=ref.coef_[0], b=ref.intercept_[0], support=ref.support_, note="final")
        return trace

    t.trace(student_trace)
    t.trace(library_trace)
