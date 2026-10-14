---
name: exercise
description: How to write Learning_ML practice exercises - numerically-checked derivation drills in exercises.yaml, and in-browser Python code exercises (implement / library) with @blank regions, bilingual checks, and replayable traces. Use when adding or changing anything under exercises/ or python/learnml.
---

# Writing exercises

Each chapter has `exercises/ch<id>/exercises.yaml` plus Python files. `scripts/build_exercises.py`
turns them into `site/src/generated/exercises.json` (skeletons + solutions). The same checks run in
CPython (`tests/test_exercises.py`) and in the browser (Pyodide, `site/src/runtime/python.worker.ts`).

## Derivation drills (推导理解) — 4 per chapter

```yaml
derivation:
  - id: shrinkage
    title: {zh: "…", en: "…"}
    prompt: {zh: "… 可用变量：`d, lam`。", en: "… Variables: `d, lam`."}
    answer: "d^2/(d^2 + lam)"          # mathjs syntax
    vars: {d: [0.1, 4], lam: [0.01, 5]} # sampling ranges for the numeric equivalence check
    solution: {zh: "完整推导 …", en: "Full derivation …"}
```

- The student's expression is compared with `answer` at random points of `vars` — any equivalent
  form is accepted. Keep ranges inside the domain (no log of negatives, no division by ~0).
- Prompts state exactly which variables may be used. Solutions give the complete derivation and
  end by linking to the chapter (a widget, a theorem, a later chapter).
- Good drills check the chapter's key steps: one estimator property, one optimality condition, one
  probability/closed form, one "why does the algorithm work".
- YAML: escape backslashes in double-quoted strings (`\\frac`).

## Code exercises (训练实战) — `implement` + `library`

```python
def soft_threshold(z, g):
    """S(z, g) = sign(z) · max(|z| − g, 0)."""
    # @blank soft
    # zh: 用 np.sign 和 np.maximum 写成一行
    # en: One line with np.sign and np.maximum
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)
    # @end
```

- Blanks: 1–4 lines each, 3–4 per exercise; hints in both languages. The skeleton replaces each
  region with a TODO line and `raise NotImplementedError("id")`; double-Tab fills the reference.
- `implement`: the algorithm from scratch in numpy (gradient step, coordinate update, split search…).
- `library`: assemble a real library (scikit-learn, LightGBM, scipy) the way practitioners do —
  pipelines, CV, validation sets. Must be single-threaded / small enough for Pyodide (seconds).
- Checks (`<file>_check.py`): `def check(s, t)` where `s` is the student module.
  `t.test("中文名", "English name", fn)` — `fn` asserts with messages that say what was expected
  (`f"w = {w}, least squares = {ref}"`). Prefer **mathematical** checks (KKT conditions, gradient
  vs finite differences, comparison with a closed form) over matching a library's output.
- Traces: `t.trace(fn)` where `fn` returns a `learnml.Trace("classifier2d", X, y, grid=Grid(X),
  show_margin=…, title={zh, en})` and calls `trace.record(w=[w1, w2], b=…, field=…, highlight=[…],
  support=[…], weights=[…], metrics={…}, note="…")` per step. 1-D regression fits use
  `field = f(grid_x) − grid_y` so the zero contour is the fitted curve (see ch02/ch03).
  `metrics` keys must match the `metrics:` list in the YAML. Traces of one exercise may record
  different metrics (e.g. `updates` for a perceptron trace, `nll` for a Newton trace): the player
  takes each metric from the first trace that records it.
- Keep runtimes small: the whole check, including the trace, should run in a few seconds in Pyodide.

## Test

```bash
python -m pytest -q tests -k ch<id>      # solution passes every test, skeleton fails every blank test (zh + en)
python3 scripts/build_exercises.py
node site/scripts/verify/chapter.mjs <slug>   # fill-all + run in real Pyodide
```
