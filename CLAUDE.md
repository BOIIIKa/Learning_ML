# Learning_ML — notes for agents

**Visible Machine Learning** (看得见的机器学习): a bilingual (zh/en), open, non-commercial course that
turns ISL/ISLP, Prince's *Understanding Deep Learning* and 李航《统计学习方法》 into rigorous,
interactive lessons for mathematically prepared students who want research ability fast.
Live site: https://boiiika.github.io/Learning_ML/ (deployed from `main`).

## Non-negotiables

- **Copyright**: never copy book text or figures; everything is rewritten and redrawn. Cite books only by chapter.
- **Bilingual**: every user-facing string exists in zh and en, written natively (not word-for-word).
- **Rigor**: every number in notes / decks / recaps / exercises is checked in code; every widget claim has a unit test.
- **Two formats per chapter**: 竖版讲义 notes (review) and 横版动画课程 animated course (first-time learners, 3Blue1Brown style).

## Layout

| Path | What |
|---|---|
| `curriculum/outline.yaml` | single source of truth for the 30 chapters (titles, goals, triad, visuals, frontier) → `scripts/build_outline.py` renders `OUTLINE.{zh,en}.md` |
| `site/` | Astro 7 + React 19 + MDX site |
| `site/src/content/{zh,en}/<slug>.mdx` | notes; a chapter is published once both languages exist |
| `site/src/slides/{zh,en}/<slug>.mdx` | animated-course decks |
| `site/src/recaps`, `site/src/handbook` | one-page recaps; 实战手册 handbook entries |
| `site/src/components/chapters/chNN/` | chapter widgets; `components/viz`, `components/ui` shared blocks |
| `site/src/engines/` | pure-TS maths behind widgets, with vitest tests |
| `site/src/manim/` | manim-like animation layer for the course (clock, Write, Morph, drive DSL) |
| `exercises/chNN/` | exercises.yaml + Python implement/library exercises and checks |
| `python/learnml/` | blanks parser, checker, Trace protocol (shared by CPython tests and Pyodide) |
| `templates/chapter/`, `scripts/new_chapter.py` | scaffolding for a new chapter |
| `site/scripts/verify/` | Playwright checks (notes pages, course walk/film, themes) |

## Skills (read the one that fits before working)

- `.claude/skills/new-chapter` — the full chapter workflow and quality bar
- `.claude/skills/course-deck` — writing the animated course (beats, Morph, drive, pacing)
- `.claude/skills/widget` — widgets, engines, hydration and theming rules
- `.claude/skills/exercise` — derivation drills and Python exercises
- `.claude/skills/verify-site` — dev server, Pyodide, verify scripts, pre-push checks

## Commands

```bash
cd site && npx tsc --noEmit -p . && npx vitest run && npx astro build
python -m pytest -q tests && python3 scripts/build_outline.py --check
python3 scripts/new_chapter.py <id>          # scaffold the next chapter
```

## Status

Done: chapters 00–16 (notes, course, recap, widgets, exercises, handbook).
Next in order: 17 backprop-init (backpropagation and initialization), then 18–21; reuse `site/src/engines/mlp.ts` (forward + backprop, gradient-checked) and `optimization.ts` (optimizers, `detCos`).
See `docs/PROJECT_SUMMARY.md` for the full picture and roadmap, and `docs/HANDOFF.md` for the user's preferences, environment notes and the concrete plan for the next chapter.
