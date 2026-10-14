import { scaleLinear } from "d3-scale";

const f1 = (v: number) => v.toFixed(1);

export interface Series {
  key: string;
  label: string;
  values: number[];
  color: string;
  dashed?: boolean;
}

/** Small line chart of per-step metrics with a cursor at the current step. */
export default function MetricChart({ series, cursor, height = 150, zero = true }: { series: Series[]; cursor: number; height?: number; zero?: boolean }) {
  const W = 460;
  const H = height;
  const pad = { l: 36, r: 10, t: 10, b: 22 };
  const n = Math.max(...series.map((s) => s.values.length));
  const all = series.flatMap((s) => s.values).filter(Number.isFinite);
  if (n < 2 || all.length === 0) return null;
  const lo = zero ? Math.min(0, ...all) : Math.min(...all);
  const hi = Math.max(...all);
  const sx = scaleLinear().domain([0, n - 1]).range([pad.l, W - pad.r]);
  const sy = scaleLinear().domain([lo, hi === lo ? lo + 1 : hi]).range([H - pad.b, pad.t]).nice();
  const ticks = sy.ticks(4);

  return (
    <figure className="metric-chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img">
        {ticks.map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={W - pad.r} y1={f1(sy(v))} y2={f1(sy(v))} className="mc-grid" />
            <text x={pad.l - 6} y={f1(sy(v))} className="mc-tick" textAnchor="end" dominantBaseline="middle">{+v.toFixed(3)}</text>
          </g>
        ))}
        {series.map((s) => (
          <path
            key={s.key}
            d={"M" + s.values.map((v, i) => `${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join("L")}
            stroke={s.color}
            strokeDasharray={s.dashed ? "5 4" : undefined}
            pathLength={s.dashed ? undefined : 1}
            className={s.dashed ? "mc-line" : "mc-line draw"}
          />
        ))}
        <line x1={f1(sx(cursor))} x2={f1(sx(cursor))} y1={pad.t} y2={H - pad.b} className="mc-cursor" />
        {series.map((s) =>
          s.values[cursor] === undefined ? null : (
            <circle key={s.key} cx={f1(sx(cursor))} cy={f1(sy(s.values[cursor]))} r={4} fill={s.color} className="mc-dot" />
          ),
        )}
      </svg>
      <figcaption>
        {series.map((s) => (
          <span key={s.key} className="mc-legend">
            <i style={{ background: s.color }} /> {s.label}
            {s.values[cursor] !== undefined && <b> {s.values[cursor].toFixed(3)}</b>}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
