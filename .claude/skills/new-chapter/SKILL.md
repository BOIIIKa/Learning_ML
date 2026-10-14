---
name: new-chapter
description: End-to-end workflow for adding one chapter of the Learning_ML course (notes + animated course + recap + widgets + exercises + handbook, in zh and en). Use whenever the user asks to write, continue, or finish a chapter (e.g. "写第 04 章", "do chapter 10"), or to scaffold the next chapter of curriculum/outline.yaml.
---

# Adding a chapter

A chapter is done when **all** of these exist in both languages and pass verification:
notes page, animated course (deck), one-page recap, ≥3 widgets, 4 derivation exercises,
1 implement + 1 library code exercise, a handbook entry. The bar is the quality of chapters
00–03, 07, 08 — read one of them (e.g. `site/src/content/zh/linear-regression.mdx`) before writing.

## 0. Plan (before any file)

1. Read the chapter's entry in `curriculum/outline.yaml`: goals, triad, derivations, visuals,
   frontier. These are promises the chapter must keep.
2. Pick the **running scenario**. Reuse the course's characters when possible (the teacher and her
   students' study hours → ch01–03; the hospital screening → ch07; the doctor asking "why" → ch08).
   The scenario widget must make the reader *feel* the chapter's question before any formula.
3. Decide 3–4 widgets: scenario (hands-on), method/geometry, math (makes a theorem visible),
   extension/frontier. One idea per widget.
4. Design the worked example **with numbers that you verify in Python first** (exact fractions
   where possible). Every number in notes, deck and recap must agree.

## 1. Scaffold

```bash
python3 scripts/new_chapter.py <id> --widget <ScenarioWidgetName>
```

This writes every file with `TODO(scaffold)` markers. `site/src/lib/scaffold.test.ts` fails
until none remain, so CI blocks a half-written chapter. Delete template files you don't use.

## 2. Engine + widgets  → skill `widget`

- Maths in `site/src/engines/<slug>.ts` (pure TS, seeded `rng`, `r6()` data), tests in
  `<slug>.test.ts` covering **every claim a caption makes** ("bias falls with degree", "lasso is
  sparse"…). Monte-Carlo estimates need enough draws and bias corrections (see ch02's
  `biasVariance`, which subtracts Var/reps).
- Widgets in `site/src/components/chapters/ch<id>/`. Look at the result in the browser before
  writing prose about it — several widget bugs in this project were only visible in screenshots.

## 3. Notes (竖版讲义)  `site/src/content/{zh,en}/<slug>.mdx`

Six-part structure; the page appends Practice (05) and Handbook (06) automatically:

| Section `k=` | Content |
|---|---|
| `scenario` | Story + scenario widget + the 2–3 questions of the chapter |
| `method` | Definition box(es): model / strategy / algorithm; method widget; an insight box |
| `math` | Full derivations, theorem boxes, the math widget; link back to earlier chapters |
| `cases` | Worked example (table, hand-checkable) + extensions/frontier widget + bullet list |

Writing rules (both languages are first-class; not translations of each other word by word):
- Rewrite everything in our own words; cite books only by chapter (copyright).
- Rigorous but direct. Every derivation step must be checkable; use exercises for the gaps
  ("(see derivation exercise 2)").
- Chinese: full-width punctuation, spaces around Latin/math. English: plain, short sentences.
- MDX gotchas: display math only as multi-line `$$` blocks; no `{`, `<` in prose (`$p \lt n$`);
  no `style=""` strings; `\|` inside tables renders as `|`.
- Boxes: `<Box kind="definition|theorem|insight|example" title="…">`.

## 4. Animated course  `site/src/slides/{zh,en}/<slug>.mdx`  → skill `course-deck`

~10 slides, ~40–60 beats, ≈5–7 min. Every widget slide has `drive` scripts; every derivation
is a `<Morph>` chain. Zh and en decks must have the **same slides and beats in the same order**.

## 5. Recap  `site/src/recaps/{zh,en}/<slug>.mdx`

6–8 bullets, each `**Term**: one sentence + key formula`. Shown collapsible in the notes and as
the course's last slide.

## 6. Exercises  `exercises/ch<id>/`  → skill `exercise`

4 derivations (numerically checked expressions), `implement` (complete the algorithm, with a
replayable trace) and `library` (assemble scikit-learn / LightGBM / …). Then
`python3 scripts/build_exercises.py`.

## 7. Handbook  `site/src/handbook/{zh,en}/<slug>.mdx`

Go-to code, tuning tips that actually bite (scaling, leakage, API naming traps), frontier
reading taken from the outline's `frontier` list.

## 8. Verify  → skill `verify-site`

```bash
cd site && npx tsc --noEmit -p . && npx vitest run && npx astro build
python -m pytest -q ../tests          # solutions pass, skeletons fail
python3 ../scripts/build_outline.py --check
node scripts/verify/chapter.mjs <slug> zh dark     # widgets + both code exercises in Pyodide
node scripts/verify/course.mjs walk zh <slug>      # every beat, no console errors
node scripts/verify/course.mjs walk en <slug>
node scripts/verify/pages.mjs <slug>               # themes, hydration, mobile overflow
```

Look at the screenshots, not just the exit codes.

## 9. Ship

One commit per chapter ("Add chapter NN: …"), push to the working branch, update the PR
checklist. Only merge when the user asks; GitHub Pages deploys from `main`.
