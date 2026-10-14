import { useEffect, useRef, useState } from "react";
import type { Trace } from "../../engines/trace";
import { pick, t, type Lang } from "../../i18n/ui";
import MetricChart, { type Series } from "./MetricChart";
import Plot2D from "./Plot2D";
import "./player.css";

export interface MetricSpec {
  key: string;
  label: string;
  color: string;
  dashed?: boolean;
}

interface Props {
  /** One trace, or several to compare side by side on a shared timeline. */
  traces: Trace[];
  lang: Lang;
  metrics?: MetricSpec[];
  /** Base time per step in ms at 1× speed. */
  stepMs?: number;
  autoplay?: boolean;
}

const SPEEDS = [0.5, 1, 2, 4];

export default function TracePlayer({ traces, lang, metrics = [], stepMs = 650, autoplay = false }: Props) {
  const total = Math.max(...traces.map((tr) => tr.steps.length));
  const [k, setK] = useState(autoplay ? 0 : total - 1);
  const [playing, setPlaying] = useState(autoplay);
  const [speed, setSpeed] = useState(1);
  const timer = useRef<number>(0);
  const scrub = useRef<HTMLInputElement>(null);

  // Set after mount: an SSR'd multi-value style attribute trips React's hydration check.
  useEffect(() => {
    scrub.current!.style.backgroundSize = `${(100 * k) / Math.max(1, total - 1)}% 100%, 100% 100%`;
  }, [k, total]);

  useEffect(() => setK((cur) => Math.min(cur, total - 1)), [total]);

  useEffect(() => {
    if (!playing) return;
    timer.current = window.setTimeout(() => {
      if (k >= total - 1) setPlaying(false);
      else setK(k + 1);
    }, stepMs / speed);
    return () => clearTimeout(timer.current);
  }, [playing, k, total, speed, stepMs]);

  const toggle = () => {
    if (!playing && k >= total - 1) setK(0);
    setPlaying(!playing);
  };

  const primary = traces[0];
  const series: Series[] = metrics.map((m) => ({
    ...m,
    values: primary.steps.map((s) => s.metrics?.[m.key] ?? NaN),
  }));

  return (
    <div className="trace-player">
      <div className={`tp-plots n${traces.length}`}>
        {traces.map((tr, i) => {
          const idx = Math.min(k, tr.steps.length - 1);
          const step = tr.steps[idx];
          return (
            <figure key={i} className="tp-plot">
              {tr.title && <figcaption className="tp-title">{pick(tr.title, lang)}</figcaption>}
              <Plot2D X={tr.X} y={tr.y} step={step} grid={tr.grid} showMargin={tr.showMargin !== false} tweenMs={Math.min(450, stepMs / speed)} />
              <div className="tp-note">
                <span className="tp-badge">{t(lang, "step", { k: idx + 1, n: tr.steps.length })}</span>
                {step?.note && step.note !== "step" && <span className="tp-note-text">{step.note}</span>}
              </div>
            </figure>
          );
        })}
      </div>

      {series.length > 0 && <MetricChart series={series} cursor={Math.min(k, primary.steps.length - 1)} />}

      <div className="tp-controls">
        <button className="btn tp-play" onClick={toggle} aria-label={playing ? t(lang, "pause") : t(lang, "play")}>
          {playing ? "❚❚" : "▶"}
        </button>
        <button className="btn tp-icon" onClick={() => { setPlaying(false); setK(Math.max(0, k - 1)); }} aria-label="-1">‹</button>
        <input
          ref={scrub}
          className="tp-scrub"
          type="range"
          min={0}
          max={total - 1}
          value={k}
          onChange={(e) => { setPlaying(false); setK(+e.target.value); }}
        />
        <button className="btn tp-icon" onClick={() => { setPlaying(false); setK(Math.min(total - 1, k + 1)); }} aria-label="+1">›</button>
        <button className="btn tp-speed" onClick={() => setSpeed(SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length])} title={t(lang, "speed")}>
          {speed}×
        </button>
      </div>
    </div>
  );
}
