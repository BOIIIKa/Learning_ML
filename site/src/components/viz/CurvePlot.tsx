import { scaleLinear } from "d3-scale";
import { useRef, useState } from "react";
import { Crosshair } from "./Crosshair";
import "./plot.css";

export interface Curve {
  f: (x: number) => number;
  color: string;
  dashed?: boolean;
  width?: number;
  label?: string;
  opacity?: number;
}

export interface CurvePlotProps {
  xs: number[];
  ys: number[];
  curves?: Curve[];
  /** Draw residual sticks from each point to this function. */
  residualsTo?: (x: number) => number;
  /** Indices drawn as outliers (hollow red). */
  outliers?: number[];
  xDomain: [number, number];
  yDomain: [number, number];
  xLabel?: string;
  yLabel?: string;
  /** Draggable handles (data coordinates). */
  handles?: { p: [number, number]; onDrag: (p: [number, number]) => void }[];
  height?: number;
  vlines?: { x: number; color: string; label?: string }[];
  /** Per-point fill colours (overrides the default class colour). */
  pointColors?: string[];
  /** Extra hollow points, e.g. unseen test data. */
  extraPoints?: { xs: number[]; ys: number[] };
}

const W = 460;
const f1 = (v: number) => v.toFixed(1);

/** A 1-D function plot: scatter points, curves, residual sticks and draggable handles. */
export default function CurvePlot({ xs, ys, curves = [], residualsTo, outliers = [], xDomain, yDomain, xLabel, yLabel, handles = [], height = 320, vlines = [], pointColors, extraPoints }: CurvePlotProps) {
  const H = height;
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<number | null>(null);
  const [hover, setHover] = useState<[number, number] | null>(null);
  const sx = scaleLinear().domain(xDomain).range([34, W - 10]);
  const sy = scaleLinear().domain(yDomain).range([H - 26, 10]);
  const grid = Array.from({ length: 161 }, (_, i) => xDomain[0] + ((xDomain[1] - xDomain[0]) * i) / 160);
  const clampY = (v: number) => Math.max(-50, Math.min(H + 50, sy(v)));
  const out = new Set(outliers);
  const toData = (e: React.PointerEvent): [number, number] => {
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.current!.getScreenCTM()!.inverse());
    return [sx.invert(p.x), sy.invert(p.y)];
  };

  return (
    <svg ref={svg} className="curve-plot" viewBox={`0 0 ${W} ${H}`} role="img"
      onPointerMove={(e) => {
        const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.current!.getScreenCTM()!.inverse());
        setHover([pt.x, pt.y]);
        if (drag.current !== null) handles[drag.current]?.onDrag(toData(e));
      }}
      onPointerUp={() => (drag.current = null)} onPointerLeave={() => { drag.current = null; setHover(null); }}>
      <defs><clipPath id="cp-clip"><rect x={34} y={10} width={W - 44} height={H - 36} /></clipPath></defs>
      <rect width={W} height={H} rx={12} className="plot-bg" />
      {sx.ticks(5).map((t) => <text key={`x${t}`} x={f1(sx(t))} y={H - 8} textAnchor="middle" className="cp-tick">{t}</text>)}
      {sy.ticks(5).map((t) => (
        <g key={`y${t}`}>
          <line x1={34} x2={W - 10} y1={f1(sy(t))} y2={f1(sy(t))} className="cp-grid" />
          <text x={28} y={f1(sy(t))} textAnchor="end" dominantBaseline="middle" className="cp-tick">{t}</text>
        </g>
      ))}
      {xLabel && <text x={W - 12} y={H - 30} textAnchor="end" className="cp-label">{xLabel}</text>}
      {yLabel && <text x={40} y={22} className="cp-label">{yLabel}</text>}
      <g clipPath="url(#cp-clip)">
        {vlines.map((v, i) => (
          <g key={`v${i}`}>
            <line x1={f1(sx(v.x))} x2={f1(sx(v.x))} y1={10} y2={H - 26} stroke={v.color} className="cp-vline" />
            {v.label && <text x={f1(sx(v.x) + 4)} y={24 + 14 * i} fill={v.color} className="cp-vlabel">{v.label}</text>}
          </g>
        ))}
        {residualsTo && xs.map((x, i) => (
          <line key={`r${i}`} x1={f1(sx(x))} x2={f1(sx(x))} y1={f1(sy(ys[i]))} y2={f1(clampY(residualsTo(x)))} className="cp-res" />
        ))}
        {curves.map((c, i) => (
          <path key={`c${i}`} d={"M" + grid.map((x) => `${f1(sx(x))},${f1(clampY(c.f(x)))}`).join("L")}
            stroke={c.color} strokeWidth={c.width ?? 2.6} strokeDasharray={c.dashed ? "6 4" : undefined} strokeOpacity={c.opacity}
            pathLength={c.dashed ? undefined : 1} className={c.dashed ? "cp-curve" : "cp-curve draw"} />
        ))}
        {extraPoints?.xs.map((x, i) => <circle key={`e${i}`} cx={f1(sx(x))} cy={f1(clampY(extraPoints.ys[i]))} r={3.5} className="cp-extra" />)}
        {xs.map((x, i) => <circle key={`p${i}`} cx={f1(sx(x))} cy={f1(sy(ys[i]))} r={4} className={out.has(i) ? "cp-out" : "cp-pt"} style={pointColors ? { fill: pointColors[i] } : undefined} />)}
      </g>
      {handles.map((h, i) => (
        <circle key={`h${i}`} cx={f1(sx(h.p[0]))} cy={f1(sy(h.p[1]))} r={9} className="cp-handle"
          onPointerDown={(e) => { (e.target as Element).setPointerCapture?.(e.pointerId); drag.current = i; }} />
      ))}
      <Crosshair p={hover} W={W} H={H} label={hover ? `x=${sx.invert(hover[0]).toFixed(2)}  y=${sy.invert(hover[1]).toFixed(2)}` : ""} />
    </svg>
  );
}
