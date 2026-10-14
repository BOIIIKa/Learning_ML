import { contours } from "d3-contour";
import { scaleLinear } from "d3-scale";
import { useId, useMemo, useRef, useState } from "react";
import { Crosshair } from "./Crosshair";
import type { TraceStep } from "../../engines/trace";
import { useTween } from "./useTween";
import "./plot.css";

export interface ExtraLine {
  w: [number, number];
  b: number;
  color: string;
  label?: string;
  dashed?: boolean;
}

export interface Plot2DProps {
  X: number[][];
  y: number[];
  step?: TraceStep;
  grid?: { x: number[]; y: number[] };
  /** Draw the ±1 margin lines around the separating line. */
  showMargin?: boolean;
  extraLines?: ExtraLine[];
  /** Extra hollow points (e.g. unseen test data); `bad` marks them with a red cross-ring. */
  extraPoints?: { X: number[][]; y: number[]; bad?: boolean[] };
  onDragPoint?: (index: number, p: [number, number]) => void;
  /** Tint each side of an axis-aligned split with the colour of its majority class. */
  highlightSide?: { feature: number; threshold: number };
  /** Fixed data domain; defaults to the data's bounding box plus padding. */
  domain?: { x: [number, number]; y: [number, number] };
  height?: number;
  tweenMs?: number;
  /** Fixed positive contour levels for `field` (mirrored for negative values); default: thirds of max |field|. */
  levels?: number[];
  /** Extra SVG drawn above the points, given the data → pixel scales. */
  overlay?: (sx: (v: number) => number, sy: (v: number) => number) => React.ReactNode;
}

const W = 460;
// Rounded coordinates keep server- and client-rendered SVG identical (no hydration mismatch).
const f1 = (v: number) => v.toFixed(1);

/** Segment of the line w·p + c = 0 clipped to the box, or null if it misses it. */
function clipLine(w: [number, number], c: number, [x0, x1]: [number, number], [y0, y1]: [number, number]) {
  const pts: [number, number][] = [];
  if (Math.abs(w[1]) > 1e-12) {
    for (const x of [x0, x1]) {
      const y = -(w[0] * x + c) / w[1];
      if (y >= y0 - 1e-9 && y <= y1 + 1e-9) pts.push([x, y]);
    }
  }
  if (Math.abs(w[0]) > 1e-12) {
    for (const y of [y0, y1]) {
      const x = -(w[1] * y + c) / w[0];
      if (x >= x0 - 1e-9 && x <= x1 + 1e-9) pts.push([x, y]);
    }
  }
  if (pts.length < 2) return null;
  // Pick the two intersections farthest apart (a line through a corner yields duplicates).
  let best: [[number, number], [number, number]] = [pts[0], pts[1]];
  let dmax = -1;
  for (const a of pts) for (const b of pts) {
    const d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
    if (d > dmax) { dmax = d; best = [a, b]; }
  }
  return best;
}

export default function Plot2D({
  X, y, step, grid, showMargin = true, extraLines = [], extraPoints, onDragPoint, highlightSide, domain, height = 380, tweenMs = 450,
  levels: fixedLevels, overlay,
}: Plot2DProps) {
  const uid = useId().replace(/:/g, "");
  const svgRef = useRef<SVGSVGElement>(null);
  const dragging = useRef<number | null>(null);
  const [hover, setHover] = useState<[number, number] | null>(null);

  const dom = useMemo(() => {
    if (domain) return domain;
    if (grid) return { x: [grid.x[0], grid.x[grid.x.length - 1]], y: [grid.y[0], grid.y[grid.y.length - 1]] } as const;
    const xs = X.map((p) => p[0]);
    const ys = X.map((p) => p[1]);
    return { x: [Math.min(...xs) - 0.8, Math.max(...xs) + 0.8], y: [Math.min(...ys) - 0.8, Math.max(...ys) + 0.8] } as const;
  }, [domain, grid, X]);

  const H = height;
  const sx = useMemo(() => scaleLinear().domain(dom.x).range([0, W]), [dom]);
  const sy = useMemo(() => scaleLinear().domain(dom.y).range([H, 0]), [dom, H]);

  const lineVec = useMemo(() => (step?.w && step.b !== undefined ? [step.w[0], step.w[1], step.b] : undefined), [step]);
  const line = useTween(lineVec, tweenMs);
  const field = useTween(step?.field, tweenMs);
  const weights = step?.weights;

  const fieldPaths = useMemo(() => {
    if (!field || !grid) return null;
    const nx = grid.x.length;
    const ny = grid.y.length;
    const dx = (grid.x[nx - 1] - grid.x[0]) / (nx - 1);
    const dy = (grid.y[ny - 1] - grid.y[0]) / (ny - 1);
    const toPath = (rings: number[][][][]) =>
      rings
        .flatMap((poly) =>
          poly.map(
            (ring) =>
              "M" + ring.map(([gx, gy]) => `${sx(grid.x[0] + (gx - 0.5) * dx).toFixed(1)},${sy(grid.y[0] + (gy - 0.5) * dy).toFixed(1)}`).join("L") + "Z",
          ),
        )
        .join("");
    const m = Math.max(1e-9, ...field.map(Math.abs));
    const gen = contours().size([nx, ny]);
    const levels = fixedLevels ?? [1e-9, m / 3, (2 * m) / 3];
    const neg = field.map((v) => -v);
    return {
      pos: levels.map((l) => toPath(gen.thresholds([l])(field)[0].coordinates as number[][][][])),
      neg: levels.map((l) => toPath(gen.thresholds([l])(neg)[0].coordinates as number[][][][])),
      zero: toPath(gen.thresholds([0])(field)[0].coordinates as number[][][][]),
    };
  }, [field, grid, sx, sy, fixedLevels]);

  const toData = (e: React.PointerEvent): [number, number] => {
    const svg = svgRef.current!;
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.getScreenCTM()!.inverse());
    return [sx.invert(pt.x), sy.invert(pt.y)];
  };

  const support = new Set(step?.support ?? []);
  const highlight = new Set(step?.highlight ?? []);
  const linePath = (w: [number, number], c: number) => {
    const seg = clipLine(w, c, dom.x as [number, number], dom.y as [number, number]);
    return seg ? `M${f1(sx(seg[0][0]))},${f1(sy(seg[0][1]))}L${f1(sx(seg[1][0]))},${f1(sy(seg[1][1]))}` : "";
  };

  return (
    <svg
      ref={svgRef}
      className={`plot2d${onDragPoint ? " draggable" : ""}`}
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      onPointerMove={(e) => {
        const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(svgRef.current!.getScreenCTM()!.inverse());
        setHover([pt.x, pt.y]);
        if (dragging.current !== null && onDragPoint) onDragPoint(dragging.current, toData(e));
      }}
      onPointerUp={() => (dragging.current = null)}
      onPointerLeave={() => { dragging.current = null; setHover(null); }}
    >
      <defs>
        <pattern id={`dots-${uid}`} width="23" height="23" patternUnits="userSpaceOnUse">
          <path d="M23 0H0V23" className="plot-gridline" />
        </pattern>
      </defs>
      <rect width={W} height={H} className="plot-bg" />
      <rect width={W} height={H} fill={`url(#dots-${uid})`} />

      {fieldPaths && (
        <g className="plot-field">
          {fieldPaths.neg.map((d, i) => <path key={`n${i}`} d={d} className="field-neg" />)}
          {fieldPaths.pos.map((d, i) => <path key={`p${i}`} d={d} className="field-pos" />)}
          <path d={fieldPaths.zero} className="field-zero" />
        </g>
      )}

      {highlightSide && (() => {
        const { feature: f, threshold: th } = highlightSide;
        const side = (le: boolean) => {
          const idx = X.flatMap((p, i) => ((p[f] <= th) === le ? [i] : []));
          const pos = idx.filter((i) => y[i] > 0).length;
          return idx.length === 0 ? "none" : pos * 2 >= idx.length ? "pos" : "neg";
        };
        const c = f === 0 ? f1(sx(th)) : f1(sy(th));
        const [a, b] = f === 0
          ? [{ x: 0, y: 0, width: +c, height: H }, { x: +c, y: 0, width: W - +c, height: H }]
          : [{ x: 0, y: +c, width: W, height: H - +c }, { x: 0, y: 0, width: W, height: +c }];
        return (
          <g>
            <rect {...a} className={`side-${side(true)}`} />
            <rect {...b} className={`side-${side(false)}`} />
          </g>
        );
      })()}

      {extraLines.map((l, i) => (
        <path key={i} d={linePath(l.w, l.b)} stroke={l.color} className={`extra-line${l.dashed ? " dashed" : ""}`} />
      ))}

      {line && (line[0] !== 0 || line[1] !== 0) && (
        <g>
          {showMargin && (
            <>
              <path d={linePath([line[0], line[1]], line[2] - 1)} className="margin-line pos" />
              <path d={linePath([line[0], line[1]], line[2] + 1)} className="margin-line neg" />
            </>
          )}
          <path d={linePath([line[0], line[1]], line[2])} pathLength={1} className="sep-line draw" />
        </g>
      )}

      {extraPoints?.X.map((p, i) => (
        <g key={`e${i}-${p[0]}`} transform={`translate(${f1(sx(p[0]))},${f1(sy(p[1]))})`}>
          <circle r={4.5} className={`pt-new ${extraPoints.y[i] > 0 ? "pos" : "neg"}${extraPoints.bad?.[i] ? " bad" : ""}`}
            style={{ animationDelay: `${(i % 30) * 25}ms` }} />
        </g>
      ))}

      {X.map((p, i) => {
        const r = Math.max(2.5, Math.min(16, 5.5 * Math.sqrt(weights?.[i] ?? 1)));
        return (
          <g key={i} transform={`translate(${f1(sx(p[0]))},${f1(sy(p[1]))})`}>
            {support.has(i) && <circle r={r + 5} className="sv-ring" />}
            {highlight.has(i) && <circle r={r + 3} className="hl-ring" />}
            <circle
              r={r}
              className={`pt ${y[i] > 0 ? "pos" : "neg"}`}
              onPointerDown={(e) => {
                if (!onDragPoint) return;
                (e.target as Element).setPointerCapture?.(e.pointerId);
                dragging.current = i;
              }}
            />
          </g>
        );
      })}
      {overlay?.(sx, sy)}
      <Crosshair p={hover} W={W} H={H} label={hover ? `(${sx.invert(hover[0]).toFixed(2)}, ${sy.invert(hover[1]).toFixed(2)})` : ""} />
    </svg>
  );
}
