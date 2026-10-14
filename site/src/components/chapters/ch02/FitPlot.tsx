import { scaleLinear } from "d3-scale";
import { useId } from "react";
import "./ch02.css";

export interface FitCurve { ys: number[]; className: string; key?: string }
export interface PointGroup { xs: number[]; ys: number[]; className: string; ring?: boolean; r?: number }

interface Props {
  /** x positions at which every curve's `ys` are given. */
  grid: number[];
  curves?: FitCurve[];
  points?: PointGroup[];
  xDomain: [number, number];
  yDomain: [number, number];
  xLabel?: string;
  yLabel?: string;
  height?: number;
}

const W = 460;
const f1 = (v: number) => v.toFixed(1);

/** Scatter + curves on a fixed 1-D domain (study hours → score), clipped to the plot area. */
export default function FitPlot({ grid, curves = [], points = [], xDomain, yDomain, xLabel, yLabel, height = 280 }: Props) {
  const H = height;
  const clip = `g2-clip-${useId().replace(/:/g, "")}`;
  const sx = scaleLinear().domain(xDomain).range([34, W - 10]);
  const sy = scaleLinear().domain(yDomain).range([H - 26, 10]);
  const cy = (v: number) => f1(Math.max(-40, Math.min(H + 40, sy(Number.isFinite(v) ? v : 1e9))));
  return (
    <svg className="g2-svg" viewBox={`0 0 ${W} ${H}`} role="img">
      <defs><clipPath id={clip}><rect x={34} y={10} width={W - 44} height={H - 36} /></clipPath></defs>
      <rect width={W} height={H} rx={12} className="plot-bg" />
      {sx.ticks(5).map((t) => <text key={`x${t}`} x={sx(t)} y={H - 8} textAnchor="middle" className="g2-tick">{t}</text>)}
      {sy.ticks(5).map((t) => (
        <g key={`y${t}`}>
          <line x1={34} x2={W - 10} y1={sy(t)} y2={sy(t)} className="g2-grid" />
          <text x={28} y={sy(t)} textAnchor="end" dominantBaseline="middle" className="g2-tick">{t}</text>
        </g>
      ))}
      {xLabel && <text x={W - 12} y={H - 30} textAnchor="end" className="g2-label">{xLabel}</text>}
      {yLabel && <text x={40} y={22} className="g2-label">{yLabel}</text>}
      <g clipPath={`url(#${clip})`}>
        {curves.map((c, i) => (
          <path key={c.key ?? i} d={"M" + grid.map((x, j) => `${f1(sx(x))},${cy(c.ys[j])}`).join("L")} className={`g2-curve ${c.className}`} />
        ))}
        {points.map((g, gi) => g.xs.map((x, i) => (
          <g key={`${gi}-${i}`} transform={`translate(${f1(sx(x))},${cy(g.ys[i])})`}>
            {g.ring && <circle r={(g.r ?? 4.5) + 3} className="g2-val-ring" />}
            <circle r={g.r ?? 4.5} className={g.className} />
          </g>
        )))}
      </g>
    </svg>
  );
}
