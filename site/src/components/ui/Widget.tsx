import { useEffect, useRef, type ReactNode } from "react";
import "./widget.css";

/** A framed interactive figure: title, body, optional caption and a stats/controls row. */
export default function Widget({ title, caption, children, tag }: { title: string; caption?: ReactNode; children: ReactNode; tag?: string }) {
  return (
    <figure className="widget card">
      <header className="widget-head">
        <span className="widget-dot" />
        <span className="widget-title">{title}</span>
        {tag && <span className="widget-tag">{tag}</span>}
      </header>
      <div className="widget-body">{children}</div>
      {caption && <figcaption className="widget-caption">{caption}</figcaption>}
    </figure>
  );
}

export function Slider({ label, value, min, max, step, onChange, format }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; format?: (v: number) => string;
}) {
  // Set after mount: an SSR'd multi-value style attribute trips React's hydration check.
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    input.current!.style.backgroundSize = `${(100 * (value - min)) / (max - min)}% 100%, 100% 100%`;
  }, [value, min, max]);
  return (
    <label className="w-slider">
      <span className="w-slider-label">{label}</span>
      <input ref={input} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} />
      <span className="w-slider-value">{format ? format(value) : value}</span>
    </label>
  );
}

export function Stat({ label, value, tone }: { label: string; value: ReactNode; tone?: "good" | "bad" | "accent" }) {
  return (
    <div className={`w-stat ${tone ?? ""}`}>
      <span className="w-stat-value">{value}</span>
      <span className="w-stat-label">{label}</span>
    </div>
  );
}
