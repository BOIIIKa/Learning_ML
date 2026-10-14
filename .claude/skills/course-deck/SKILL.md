---
name: course-deck
description: How to write or edit an animated-course deck (横版动画课程, site/src/slides/{zh,en}/*.mdx) in the 3Blue1Brown / manim style - beats, narration pacing, Write/Morph/Indicate effects, and drive scripts that operate widgets. Use when creating a deck for a chapter, adding slides, or changing how the course animates.
---

# Writing an animated-course deck

The course plays itself like a 3B1B video: each `<Step>` is a **beat**. A beat enters with a
manim-like animation, its `say` narration runs as a subtitle (optionally spoken), its `drive`
script operates the slide's widget, and the timeline moves on when all three are done
(+0.45 s, + `hold`). Engine: `site/src/manim/` and `site/src/components/course/controller.ts`.

## Building blocks

```mdx
import Slide from "../../components/course/Slide.astro";
import Step from "../../components/course/Step.astro";
import Morph from "../../components/course/Morph.astro";

<Slide lang="zh" part="scenario" title="标题" say="进入这页时的旁白">
  <Step say="旁白" drive="s1=11/4">屏幕上的一行要点 →</Step>
  <div slot="visual"><MyWidget client:visible lang="zh" /></div>
</Slide>
```

- `Slide` props: `part` = intro | scenario | method | math | cases | practice (sets the colour
  and the progress-bar segment); `layout` = split (default: text left, `visual` slot right) |
  center | title; `title`; `kicker` (title layout); `say`.
- `Step` props:
  - `effect`: `write` (default — glyphs drawn as outlines, then filled), `fade`, `pop`
    (buttons), `focus` (write + Indicate: key statements, usually a `<p class="big-q sm">`),
    `morph` (only inside `<Morph>`).
  - `drive`: widget script (below). `hold`: extra seconds to linger.
- `**bold**` inside a beat gets a manim Indicate (yellow flash) after it is written.
- The last slide (recap) is appended automatically from `src/recaps/<lang>/<slug>.mdx`.

## Morph: derivations that transform in place

```mdx
<Morph>
<Step effect="morph" say="…">

$$
\mathbb E[(Y-c)^2]
$$

</Step>
<Step effect="morph" say="…">

$$
\mathbb E[(Y-\mu+\mu-c)^2]
$$

</Step>
</Morph>
```

Matching glyphs (same character, in order) fly to their new positions; the rest fade out/in —
manim's TransformMatchingTex. Design chains so most symbols survive each step (3–5 lines).
Every chapter should have at least one Morph chain for its central derivation.

## drive: operating the widget (manim ValueTracker)

Commands run in order, separated by `;`:

| Command | Meaning |
|---|---|
| `s2=0.3/4` | animate the 2nd `<Slider>` of the visual to 0.3 over 4 s (default 2.5 s, smooth) |
| `s1=80%/3` | … to 80 % of its range (use when the range is data-dependent) |
| `click:chip3` | Circumscribe + click the 3rd `.chip` (also `btn2`, `check1`, or any CSS selector such as `.tp-play`) |
| `wait:1.5` | pause |

Counting is in DOM order inside the visual. Check a widget's controls before writing drives:
`grep -n "<Slider\|className=\"chip\|className=\"btn\|checkbox" <widget>.tsx`. Every widget slide
needs drives — a static figure next to narration that says "drag the slider" is a bug in the
animated course. Narration may still invite the viewer to try it themselves afterwards.

## Pacing (节奏适中)

- Narration per beat: **zh 15–60 characters, en 10–40 words** (≈ 215 ms per zh character, 330 ms per
  en word). Split longer thoughts into two beats.
- On-screen text per beat: one line (a phrase + formula), never a paragraph. The subtitle carries
  the sentence; the canvas carries the structure.
- 3–5 beats per slide, ~10 slides per chapter, 5–7 minutes total (the start screen shows the estimate).
- Drives take time too: a beat lasts max(narration, entrance + drive).
- Rhythm per slide: question → build (beats) → payoff (`focus` beat or a drive that reveals it).

## Deck skeleton (per chapter)

1. `intro` title slide — kicker "第 NN 章 · 动画课程", one-line subtitle.
2. `scenario` — the scenario widget, driven to show the problem; then a `layout="center"` slide
   with the chapter question as a `focus` beat.
3. `method` — model / strategy / algorithm beats; method widget driven.
4. `math` — Morph derivation chain(s); math widget.
5. `cases` — worked example beats with the exact numbers of the notes; extension widget.
6. `practice` — list exercises, then a `pop` beat with the button to `#s-practice`.

## Bilingual rule

zh and en decks must have identical slides and beats in the same order (drives are written once
and must work for both). Narration is written natively in each language, not translated literally.

## Check

```bash
node site/scripts/verify/course.mjs walk zh <slug>     # every beat runs, console clean
node site/scripts/verify/course.mjs film <slug> zh "#5" "2,6,10,16"   # screenshots over time
```
Watch for: visual overflow (widgets auto-shrink, but very tall widgets get tiny), drives that
target the wrong control, morph chains where nothing matches (looks like a crossfade). Film a
Morph slide at times *during* a transformation (e.g. 0.3 s into a beat): a broken morph still
looks fine once it has finished.
