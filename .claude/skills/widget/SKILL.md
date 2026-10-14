---
name: widget
description: How to build an interactive visualization widget (React + SVG) for the Learning_ML site - the shared Widget/Slider/Stat/Plot2D/CurvePlot/MetricChart/TracePlayer building blocks, the pure-TS engine pattern, hydration-safe rendering, theming through CSS variables, and making controls drivable by the animated course. Use when adding or fixing anything under site/src/components/chapters or site/src/engines.
---

# Building a widget

## Shape of a widget

```tsx
const TEXT = { title: { zh: "…", en: "…" }, caption: { zh: "…", en: "…" }, … };   // all strings bilingual
export default function MyWidget({ lang }: { lang: Lang }) {
  const [k, setK] = useState(3);
  const data = useMemo(() => sample(30, 2), []);          // deterministic, seeded
  const fit = useMemo(() => polyFit(data.xs, data.ys, k), [data, k]);
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="scenario|interactive|frontier">
      <CurvePlot … />
      <Slider label={…} value={k} min={0} max={11} step={1} onChange={setK} />
      <div className="w-stats"><Stat label={…} value={x.toFixed(2)} tone="accent" /></div>
    </Widget>
  );
}
```

Used from MDX as `<MyWidget client:visible lang="zh" />` (notes and decks alike).

- **One idea per widget.** The caption says what to look at, what to drag, and what the reader will
  notice. End synthetic-data captions with "数据为合成示意 / Synthetic illustrative data" when realistic.
- Maths lives in `site/src/engines/<slug>.ts` — pure functions, no React — and is unit-tested
  (`<slug>.test.ts`): every claim the caption or notes make about the widget gets a test.

## Building blocks (`site/src/components/viz`, `components/ui`)

| Component | Use for |
|---|---|
| `Widget`, `Slider`, `Stat` (`ui/Widget.tsx`) | frame, sliders (automatically drivable by the course), stat tiles |
| `CurvePlot` | 1-D functions: points, curves (`dashed`, `opacity`), residual sticks, draggable handles, vlines, extra hollow points |
| `Plot2D` | 2-D classification: points by class, a separating line `step.w/b` (+margins), a decision `field` on a `grid` (contours), extra lines/points, draggable points |
| `MetricChart` | small line chart over steps with a cursor (`zero={false}` for log scales) |
| `TracePlayer` | replays a `Trace` (from Python exercises or TS engines) with play/scrub/speed |
| chips | `<div className="chips"><button className={`chip${on ? " on" : ""}`}>…` |
| custom SVG | `<svg className="lm-svg" viewBox=…>` + `scaleLinear` from d3-scale; add the class to the shared rules in `ui/widget.css` and `styles/blueprint.css` if new |

Both plots draw a CAD crosshair with coordinates on hover and "draw in" their curves.

## Rules that prevent real bugs seen in this project

1. **Hydration**: the server render and the first client render must match exactly.
   - Round SVG coordinates: `const f1 = (v: number) => v.toFixed(1)` for every computed `x/y/cx/d`.
   - Round generated data with `r6()`; use seeded `rng(seed)`, never `Math.random()` in render.
   - Never put a multi-value `style` (e.g. `backgroundSize: "a, b"`) or CSS custom properties in SSR
     style props — set them in `useEffect` via a ref.
   - Don't mutate island DOM before hydration (the course's animations use the Web Animations API
     and CSS for exactly this reason).
2. **Colours only from CSS variables**: `var(--accent)`, `--accent-2`, `--accent-3`, `--ink`,
   `--text-muted`, `--class-pos`, `--class-neg`, `--success`, `--danger`, `--border`. The notes have
   two blueprint themes and the course has the manim palette; hard-coded colours break one of them.
3. **Layout**: grid cells need `min-width: 0`; test at 390 px width (`node site/scripts/verify/pages.mjs`).
4. **Drivable controls**: the course drives `Slider`s by order (`s1`, `s2`…) and clicks chips /
   buttons / checkboxes by order. Prefer these standard controls over custom ones; drag-only
   interactions cannot be driven (fine for "try it yourself", but give the widget at least one
   slider or button that shows the key effect).
5. **Performance**: everything recomputes on every slider tick — keep engines O(n·grid) and memoize
   expensive sweeps (`useMemo` over the whole slider range, then index).
6. **Monte-Carlo honesty**: if a widget estimates something by resampling, check the estimator's
   own noise (ch02's bias² needed a Var/reps correction to stop rising at high degree).

## Look at it

Screenshots in both themes and inside the course are part of the work:
`node site/scripts/verify/chapter.mjs <slug> zh light` and
`node site/scripts/verify/course.mjs film <slug> zh "#<slide>" "2,6,10"`.
