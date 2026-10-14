"""Checks for gen_library.py. Runs identically under CPython and Pyodide."""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline

from learnml import Grid, Trace

# A tiny made-up corpus of triage notes: 1 = needs a follow-up.
NOTES = [
    ("persistent fever and cough for two weeks", 1),
    ("high fever chest pain shortness of breath", 1),
    ("cough with blood and night sweats", 1),
    ("fever cough weight loss fatigue", 1),
    ("shortness of breath and chest pain at rest", 1),
    ("routine checkup no complaints", 0),
    ("mild headache after long day", 0),
    ("sprained ankle while running", 0),
    ("routine vaccination no complaints", 0),
    ("seasonal allergy runny nose", 0),
]


def check(s, t):
    data = load_breast_cancer()
    state = {}

    def t_models():
        ms = s.models()
        assert sorted(ms) == ["GaussianNB", "LDA", "Logistic", "QDA"], f"keys {sorted(ms)}"
        assert isinstance(ms["Logistic"], Pipeline), "logistic regression should sit in a pipeline after a scaler"

    def t_cv():
        sc = s.cv_scores(s.models(), data.data, data.target)
        state["sc"] = sc
        for name, v in sc.items():
            assert v > 0.9, f"{name}: CV accuracy {v:.3f} should exceed 0.9"
        assert sc["Logistic"] >= sc["GaussianNB"], "with 30 correlated features, logistic regression should not lose to naive Bayes"

    def t_text():
        texts, labels = zip(*NOTES)
        pipe = s.text_model(1.0).fit(texts, labels)
        assert list(pipe.predict(["fever and chest pain", "routine checkup"])) == [1, 0], "classify two new notes"
        p = pipe.predict_proba(["sprained cough"])[0, 1]
        assert 0 < p < 1, "with smoothing no probability is exactly 0 or 1"
        state["pipe"] = pipe

    def t_words():
        pipe = state.get("pipe")
        assert pipe is not None, "fix the previous test first"
        words = s.top_words(pipe, 3)
        assert len(words) == 3 and set(words) <= {"fever", "cough", "chest", "pain", "breath", "shortness"}, f"top words {words}"

    t.test("组装四个模型", "Assembling the four models", t_models)
    t.test("乳腺癌数据上的交叉验证", "Cross-validation on breast cancer", t_cv)
    t.test("病历文本的朴素贝叶斯", "Naive Bayes on triage notes", t_text)
    t.test("最能指示“复查”的词", "Words that point to a follow-up", t_words)

    def trace():
        rng = np.random.default_rng(4)
        A = np.array([[1.0, 0.0], [0.55, 0.45]])
        X = np.vstack([rng.normal(size=(80, 2)) * 0.55 + [-0.9, -0.5], rng.normal(size=(80, 2)) @ A.T + [0.9, 0.6]])
        y = np.array([0] * 80 + [1] * 80)
        grid = Grid(X, n=40)
        tr = Trace("classifier2d", X, 2 * y - 1, grid=grid, show_margin=False,
                   title={"zh": "同一份数据，四种分类器", "en": "Same data, four classifiers"})
        for name, m in s.models().items():
            m.fit(X, y)
            p = m.predict_proba(grid.points)[:, 1]
            tr.record(field=np.log(p + 1e-12) - np.log(1 - p + 1e-12), metrics={"acc": float(m.score(X, y))}, note=name)
        return tr

    t.trace(trace)
