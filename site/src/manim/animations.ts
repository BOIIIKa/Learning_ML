// manim-style animations on HTML + SVG, built on the Web Animations API and a shared Clock.
//   write      — glyphs appear as outlines, then fill in, left to right (manim Write)
//   fadeIn     — fade with a small shift (FadeIn(shift=UP))
//   create     — stroke draws itself (Create) for SVG paths and lines
//   growDots   — dots grow from their centres with a lag (LaggedStart(GrowFromCenter))
//   indicate   — scale up and flash yellow, then back (Indicate)
//   circumscribe — a rectangle draws around an element and fades (Circumscribe)
//   morph      — TransformMatchingTex between two formulas: matching glyphs fly to their new place
import type { Clock } from "./clock";
import { EASE } from "./rate";

const LEAF_SEL = ".mj-g";
const KATEX_ATOMS = /\b(mord|mbin|mrel|mopen|mclose|mpunct|mop|minner|mspace)\b/;

/** Wrap every visible character in `.mj-g` spans (once). KaTeX leaves become glyphs as they are. */
export function splitGlyphs(root: HTMLElement): HTMLElement[] {
  if (!root.dataset.mjSplit) {
    root.dataset.mjSplit = "1";
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const texts: Text[] = [];
    while (walker.nextNode()) texts.push(walker.currentNode as Text);
    for (const node of texts) {
      const parent = node.parentElement!;
      if (parent.closest(".katex-mathml, svg, code, .mj-g")) continue;
      const inKatex = parent.closest(".katex-html");
      if (inKatex) {
        // A KaTeX leaf (an atom span with only text) becomes one glyph.
        if (parent.childElementCount === 0 && KATEX_ATOMS.test(parent.className) && node.data.trim()) parent.classList.add("mj-g");
        else if (parent.childElementCount === 0 && node.data.trim()) parent.classList.add("mj-g");
        continue;
      }
      if (!node.data.trim()) continue;
      const frag = document.createDocumentFragment();
      // Latin words stay whole (so line breaking is unchanged); CJK characters split one by one.
      for (const piece of node.data.match(/[　-鿿＀-￯]|[^\s　-鿿＀-￯]+|\s+/g) ?? []) {
        if (/^\s+$/.test(piece)) frag.append(piece);
        else {
          const s = document.createElement("span");
          s.className = "mj-g";
          s.textContent = piece;
          frag.append(s);
        }
      }
      node.replaceWith(frag);
    }
    // Strokes inside KaTeX (fraction bars, radicals) and inline code animate as units.
    root.querySelectorAll<HTMLElement>(".katex-html .frac-line, .katex-html svg, code, .katex-html .overline-line, .katex-html .underline-line")
      .forEach((el) => el.classList.add("mj-g", "mj-stroke"));
  }
  return [...root.querySelectorAll<HTMLElement>(LEAF_SEL)].filter((g) => !g.parentElement!.closest(".mj-stroke"));
}

const colorOf = (el: Element) => getComputedStyle(el).color;

/** manim Write: each glyph is drawn as an outline, then filled; glyphs start with a lag. */
export function write(clock: Clock, root: HTMLElement, { runTime = 1, lag = 0.6, signal }: { runTime?: number; lag?: number; signal?: AbortSignal } = {}) {
  const gs = splitGlyphs(root);
  if (!gs.length) return fadeIn(clock, root, { signal });
  // Longer text takes longer, but never drags: 0.7 s for a symbol, up to ~2.6 s for a sentence.
  const total = runTime * Math.max(700, Math.min(2600, 500 + gs.length * 45));
  const per = Math.max(250, Math.min(700, total * (1 - lag)));
  const step = gs.length > 1 ? (total - per) / (gs.length - 1) : 0;
  return Promise.all(gs.map((g, i) => {
    const c = colorOf(g);
    const stroke = g.classList.contains("mj-stroke");
    const kf: Keyframe[] = stroke
      ? [{ opacity: 0, transform: "scaleX(0)", transformOrigin: "0 50%" }, { opacity: 1, transform: "scaleX(1)", transformOrigin: "0 50%" }]
      : [
          { opacity: 0, color: "transparent", webkitTextStrokeColor: c, webkitTextStrokeWidth: "0.6px" } as Keyframe,
          { opacity: 1, color: "transparent", webkitTextStrokeColor: c, webkitTextStrokeWidth: "0.6px", offset: 0.55 } as Keyframe,
          { opacity: 1, color: c, webkitTextStrokeColor: c, webkitTextStrokeWidth: "0px" } as Keyframe,
        ];
    return clock.track(g.animate(kf, { duration: per, delay: i * step, easing: EASE.linear, fill: "backwards" }), signal);
  })).then(() => {});
}

/** manim FadeIn(shift=UP*0.3). */
export function fadeIn(clock: Clock, el: Element, { dy = 14, runTime = 0.8, delay = 0, scale = 1, signal }: { dy?: number; runTime?: number; delay?: number; scale?: number; signal?: AbortSignal } = {}) {
  return clock.track(el.animate(
    [{ opacity: 0, transform: `translateY(${dy}px) scale(${scale})` }, { opacity: 1, transform: "none" }],
    { duration: runTime * 1000, delay: delay * 1000, easing: EASE.smooth, fill: "backwards" },
  ), signal);
}

export function fadeOut(clock: Clock, el: Element, { dy = -10, runTime = 0.45, signal }: { dy?: number; runTime?: number; signal?: AbortSignal } = {}) {
  return clock.track(el.animate(
    [{ opacity: 1, transform: "none" }, { opacity: 0, transform: `translateY(${dy}px)` }],
    { duration: runTime * 1000, easing: EASE.smooth, fill: "forwards" },
  ), signal);
}

/** manim Indicate: scale to 1.2 and turn yellow, then back. */
export function indicate(clock: Clock, el: HTMLElement, { color = "var(--mn-yellow)", runTime = 1, signal }: { color?: string; runTime?: number; signal?: AbortSignal } = {}) {
  const c = colorOf(el);
  if (getComputedStyle(el).display === "inline") el.style.display = "inline-block";
  return clock.track(el.animate(
    [{ transform: "scale(1)", color: c }, { transform: "scale(1.18)", color, offset: 0.5 }, { transform: "scale(1)", color: c }],
    { duration: runTime * 1000, easing: EASE.smooth },
  ), signal);
}

/** manim Circumscribe: draw a rectangle around `el` (inside `layer`), then fade it out. */
export function circumscribe(clock: Clock, el: Element, layer: HTMLElement, { runTime = 1.3, signal }: { runTime?: number; signal?: AbortSignal } = {}) {
  const r = el.getBoundingClientRect();
  const L = layer.getBoundingClientRect();
  const sc = L.width / layer.offsetWidth || 1;
  const pad = 6;
  const w = r.width / sc + 2 * pad;
  const h = r.height / sc + 2 * pad;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "mj-circ");
  Object.assign(svg.style, { left: `${(r.left - L.left) / sc - pad}px`, top: `${(r.top - L.top) / sc - pad}px`, width: `${w}px`, height: `${h}px` });
  const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  rect.setAttribute("x", "1"); rect.setAttribute("y", "1");
  rect.setAttribute("width", String(w - 2)); rect.setAttribute("height", String(h - 2));
  rect.setAttribute("pathLength", "1");
  svg.append(rect);
  layer.append(svg);
  const a = rect.animate(
    [{ strokeDasharray: "0 1", strokeDashoffset: 0, opacity: 1 }, { strokeDasharray: "1 0", strokeDashoffset: 0, opacity: 1, offset: 0.6 }, { strokeDasharray: "1 0", opacity: 0 }],
    { duration: runTime * 1000, easing: EASE.smooth, fill: "forwards" },
  );
  return clock.track(a, signal).then(() => svg.remove());
}

/** Create for every stroked SVG shape in `root`, plus GrowFromCenter for small dots, FadeIn for fills/text. */
export function createSvg(clock: Clock, root: Element, { runTime = 1.6, signal }: { runTime?: number; signal?: AbortSignal } = {}) {
  const jobs: Promise<void>[] = [];
  const ms = runTime * 1000;
  for (const svg of root.querySelectorAll("svg")) {
    if (svg.closest(".katex, .mj-circ")) continue;
    const shapes = [...svg.querySelectorAll<SVGGeometryElement>("path, line, polyline, polygon, circle, ellipse, rect")];
    const dots = shapes.filter((s) => s.tagName === "circle" && +(s.getAttribute("r") ?? 0) <= 9);
    shapes.forEach((s) => {
      if (dots.includes(s as SVGCircleElement)) return;
      const cs = getComputedStyle(s);
      const stroked = cs.stroke !== "none" && parseFloat(cs.strokeWidth) > 0;
      const filled = cs.fill !== "none" && cs.fillOpacity !== "0";
      const big = s.tagName === "rect" && +(s.getAttribute("width") ?? 0) > 200;
      if (big) { jobs.push(fadeIn(clock, s, { dy: 0, runTime: 0.5, signal })); return; }
      if (stroked && !filled && typeof s.getTotalLength === "function") {
        let len = 0;
        try { len = s.getTotalLength(); } catch { len = 0; }
        if (len > 0) {
          jobs.push(clock.track(s.animate(
            [{ strokeDasharray: `${len} ${len}`, strokeDashoffset: len }, { strokeDasharray: `${len} ${len}`, strokeDashoffset: 0 }],
            { duration: ms * 0.8, delay: ms * 0.1, easing: EASE.smooth, fill: "backwards" },
          ), signal));
          return;
        }
      }
      jobs.push(fadeIn(clock, s, { dy: 0, runTime: runTime * 0.6, delay: runTime * 0.2, signal }));
    });
    dots.forEach((d, i) => {
      jobs.push(clock.track(d.animate(
        [{ transform: "scale(0)", opacity: 0 }, { transform: "scale(1.25)", opacity: 1, offset: 0.7 }, { transform: "scale(1)", opacity: 1 }],
        { duration: 450, delay: ms * 0.25 + (ms * 0.6 * i) / Math.max(1, dots.length), easing: EASE.smooth, fill: "backwards" },
      ), signal));
    });
    svg.querySelectorAll("text").forEach((t) => jobs.push(fadeIn(clock, t, { dy: 0, runTime: 0.6, delay: runTime * 0.5, signal })));
  }
  return Promise.all(jobs).then(() => {});
}

/** Glyph identity for matching across formulas: its text (KaTeX glyph) or a stroke kind. */
const keyOf = (g: HTMLElement) => (g.classList.contains("mj-stroke") ? `#${g.className}` : g.textContent ?? "");

/**
 * TransformMatchingTex: `to` takes the place of `from`. Glyphs with the same text (matched in
 * order of appearance) fly from their old position to the new one; the rest fade out / in.
 */
export async function morph(clock: Clock, from: HTMLElement, to: HTMLElement, { runTime = 1.5, signal }: { runTime?: number; signal?: AbortSignal } = {}) {
  const a = splitGlyphs(from);
  const b = splitGlyphs(to);
  const ms = runTime * 1000;
  const scale = from.getBoundingClientRect().width / (from.offsetWidth || 1) || 1;
  const rectA = new Map(a.map((g) => [g, g.getBoundingClientRect()]));
  // Show the target (both occupy the same grid cell), hide the source after measuring.
  to.classList.add("shown", "morph-top");
  from.classList.remove("morph-top");
  from.classList.add("morph-out");
  const pool = new Map<string, HTMLElement[]>();
  a.forEach((g) => { const k = keyOf(g); if (!pool.has(k)) pool.set(k, []); pool.get(k)!.push(g); });
  const used = new Set<HTMLElement>();
  const jobs: Promise<void>[] = [];
  for (const g of b) {
    const src = pool.get(keyOf(g))?.shift();
    const rb = g.getBoundingClientRect();
    if (src) {
      used.add(src);
      const ra = rectA.get(src)!;
      const dx = (ra.left - rb.left) / scale;
      const dy = (ra.top - rb.top) / scale;
      const sx = rb.width ? ra.width / rb.width : 1;
      jobs.push(clock.track(g.animate(
        [{ transform: `translate(${dx}px, ${dy}px) scale(${sx})`, transformOrigin: "0 0" }, { transform: "none", transformOrigin: "0 0" }],
        { duration: ms, easing: EASE.smooth, fill: "backwards" },
      ), signal));
    } else {
      jobs.push(clock.track(g.animate(
        [{ opacity: 0, transform: "translateY(-8px)" }, { opacity: 0, offset: 0.35 }, { opacity: 1, transform: "none" }],
        { duration: ms, easing: EASE.smooth, fill: "backwards" },
      ), signal));
    }
  }
  // Unmatched source glyphs: a ghost copy of the old formula fades them out in place.
  const ghost = from.cloneNode(true) as HTMLElement;
  ghost.classList.add("mj-ghost");
  ghost.classList.remove("morph-out");
  from.parentElement!.append(ghost);
  const ghostGlyphs = [...ghost.querySelectorAll<HTMLElement>(LEAF_SEL)].filter((g) => !g.parentElement!.closest(".mj-stroke"));
  a.forEach((g, i) => {
    const gg = ghostGlyphs[i];
    if (!gg) return;
    if (used.has(g)) gg.style.visibility = "hidden";
    else jobs.push(clock.track(gg.animate([{ opacity: 1 }, { opacity: 0, transform: "translateY(8px)" }], { duration: ms * 0.5, easing: EASE.smooth, fill: "forwards" }), signal));
  });
  await Promise.all(jobs);
  ghost.remove();
}
