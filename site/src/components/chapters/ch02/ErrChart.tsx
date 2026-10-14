import { scaleLinear, scaleLog } from "d3-scale";
import { useId } from "react";
import "./ch02.css";

export interface ErrSeries { key: string; label: string; values: number[]; color: string; dashed?: boolean; width?: number }

interface Props {
  xs: number[];
  series: ErrSeries[];
  xLog?: boolean;
  yLog?: boolean;
  yDomain?: [number, number];
  /** Index into `xs` of the highlighted point. */
  cursor?: number;
  vlines?: { x: number; label: string; color: string }[];
  xLabel?: string;
  yLabel?: string;
  height?: number;
  format?: (v: number) => string;
}

const W = 460;
const f1 = (v: number) => v.toFixed(1);
const fmtTick = (v: number) => (Math.abs(v) >= 1e6 ? `${+(v / 1e6).toPrecision(3)}M` : Math.abs(v) >= 1000 ? `${+(v / 1000).toPrecision(3)}k` : `${+v.toPrecision(3)}`);

/** Error-versus-complexity chart: several series over a shared x axis (linear or log), with a cursor. */
export default function ErrChart({ xs, series, xLog, yLog, yDomain, cursor, vlines = [], xLabel, yLabel, height = 190, format = (v) => v.toFixed(1) }: Props) {
  const H = height;
  const clip = `g2-ec-${useId().replace(/:/g, "")}`;
  const pad = { l: 40, r: 10, t: 10, b: 26 };
  const all = series.flatMap((s) => s.values).filter((v) => Number.isFinite(v) && (!yLog || v > 0));
  const dom: [number, number] = yDomain ?? [yLog ? Math.min(...all) : Math.min(0, ...all), Math.max(...all)];
  const sx = (xLog ? scaleLog() : scaleLinear()).domain([xs[0], xs[xs.length - 1]]).range([pad.l, W - pad.r]);
  const sy = (yLog ? scaleLog() : scaleLinear()).domain(dom).range([H - pad.b, pad.t]);
  const yt = yLog ? sy.ticks(4).filter((t) => Number.isInteger(Math.log10(t))) : sy.ticks(4);
  const xt = xLog ? sx.ticks(4).filter((t) => Number.isInteger(Math.log10(t))) : xs.length <= 13 ? xs : sx.ticks(6);
  const cy = (v: number) => f1(Math.max(-30, Math.min(H + 30, Number.isFinite(v) && (!yLog || v > 0) ? sy(v) : -30)));
  return (
    <figure className="g2-chart">
      <svg className="g2-svg" viewBox={`0 0 ${W} ${H}`} role="img">
        <defs><clipPath id={clip}><rect x={pad.l} y={pad.t - 4} width={W - pad.l - pad.r} height={H - pad.t - pad.b + 8} /></clipPath></defs>
        <rect width={W} height={H} rx={12} className="plot-bg" />
        {yt.map((t) => (
          <g key={`y${t}`}>
            <line x1={pad.l} x2={W - pad.r} y1={f1(sy(t))} y2={f1(sy(t))} className="g2-grid" />
            <text x={pad.l - 6} y={f1(sy(t))} textAnchor="end" dominantBaseline="middle" className="g2-tick">{fmtTick(t)}</text>
          </g>
        ))}
        {xt.map((t) => <text key={`x${t}`} x={f1(sx(t))} y={H - 9} textAnchor="middle" className="g2-tick">{fmtTick(t)}</text>)}
        {xLabel && <text x={W - pad.r - 2} y={H - pad.b - 5} textAnchor="end" className="g2-label">{xLabel}</text>}
        {yLabel && <text x={pad.l + 6} y={pad.t + 12} className="g2-label">{yLabel}</text>}
        <g clipPath={`url(#${clip})`}>
          {vlines.map((v) => (
            <g key={v.label}>
              <line x1={f1(sx(v.x))} x2={f1(sx(v.x))} y1={pad.t} y2={H - pad.b} stroke={v.color} className="g2-vline" />
              <text x={f1(sx(v.x) - 5)} y={pad.t + 12} textAnchor="end" fill={v.color} className="g2-vlabel">{v.label}</text>
            </g>
          ))}
          {cursor !== undefined && <line x1={f1(sx(xs[cursor]))} x2={f1(sx(xs[cursor]))} y1={pad.t} y2={H - pad.b} className="g2-cursor" />}
          {series.map((s) => (
            <path key={s.key} d={"M" + s.values.map((v, i) => `${f1(sx(xs[i]))},${cy(v)}`).join("L")} stroke={s.color}
              strokeWidth={s.width ?? 2.4} strokeDasharray={s.dashed ? "6 4" : undefined} className="g2-curve" />
          ))}
          {cursor !== undefined && series.map((s) => (
            <circle key={s.key} cx={f1(sx(xs[cursor]))} cy={cy(s.values[cursor])} r={4.5} fill={s.color} className="g2-dot" />
          ))}
        </g>
      </svg>
      <figcaption className="g2-legend">
        {series.map((s) => (
          <span key={s.key} style={{ color: s.color }}>
            <i className={s.dashed ? "dashed" : ""} style={{ background: s.color }} />
            <span style={{ color: "var(--text-muted)" }}>{s.label}</span>
            {cursor !== undefined && Number.isFinite(s.values[cursor]) && <b style={{ color: "var(--text)" }}> {format(s.values[cursor])}</b>}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
