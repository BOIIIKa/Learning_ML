import { scaleLinear } from "d3-scale";
import { useState } from "react";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";

const TEXT = {
  title: { zh: "Jensen 不等式：弦总在凸函数上方", en: "Jensen's inequality: chords sit above a convex curve" },
  x1: { zh: "x₁", en: "x₁" },
  x2: { zh: "x₂", en: "x₂" },
  p: { zh: "权重 p", en: "Weight p" },
  ef: { zh: "E[f(X)]", en: "E[f(X)]" },
  fe: { zh: "f(E[X])", en: "f(E[X])" },
  gap: { zh: "差（≥ 0）", en: "Gap (≥ 0)" },
  caption: {
    zh: "f(x) = −log x 是凸函数。随机变量 X 以概率 p 取 x₁、以 1−p 取 x₂。弦上的点（E[f(X)]）永远不低于曲线上的点（f(E[X])）。把 X 取为 q(x)/p(x)、期望取在 p 下，这个不等式就直接给出 KL 散度非负。",
    en: "f(x) = −log x is convex. Let X be x₁ with probability p and x₂ with probability 1−p. The point on the chord, E[f(X)], never lies below the point on the curve, f(E[X]). Taking X = q(x)/p(x) with the expectation under p, this inequality directly gives KL ≥ 0.",
  },
};

const W = 460;
const H = 300;
const f = (x: number) => -Math.log(x);

export default function JensenKL({ lang }: { lang: Lang }) {
  const [x1, setX1] = useState(0.35);
  const [x2, setX2] = useState(2.6);
  const [p, setP] = useState(0.5);
  const sx = scaleLinear().domain([0.1, 3.2]).range([20, W - 10]);
  const sy = scaleLinear().domain([-1.3, 2.4]).range([H - 16, 10]);
  const curve = "M" + Array.from({ length: 121 }, (_, i) => { const x = 0.1 + (i / 120) * 3.1; return `${sx(x).toFixed(1)},${sy(f(x)).toFixed(1)}`; }).join("L");
  const m = p * x1 + (1 - p) * x2;
  const ef = p * f(x1) + (1 - p) * f(x2);
  const fe = f(m);
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <svg className="jk-svg" viewBox={`0 0 ${W} ${H}`} role="img">
        <rect width={W} height={H} rx={12} className="plot-bg" />
        <line x1={0} x2={W} y1={sy(0)} y2={sy(0)} className="lm-axis" />
        <path d={curve} className="jk-curve" />
        <line x1={sx(x1)} y1={sy(f(x1))} x2={sx(x2)} y2={sy(f(x2))} className="jk-chord" />
        <line x1={sx(m)} x2={sx(m)} y1={sy(ef)} y2={sy(fe)} className="jk-gap" />
        <circle cx={sx(x1)} cy={sy(f(x1))} r={5} className="jk-end" />
        <circle cx={sx(x2)} cy={sy(f(x2))} r={5} className="jk-end" />
        <circle cx={sx(m)} cy={sy(ef)} r={6} className="jk-ef" />
        <circle cx={sx(m)} cy={sy(fe)} r={6} className="jk-fe" />
        <text x={sx(m) + 9} y={sy(ef) - 6} className="jk-lab ef">E[f(X)]</text>
        <text x={sx(m) + 9} y={sy(fe) + 16} className="jk-lab fe">f(E[X])</text>
      </svg>
      <div className="w-row">
        <Slider label={TEXT.x1[lang]} value={x1} min={0.15} max={3.1} step={0.01} onChange={setX1} format={(v) => v.toFixed(2)} />
        <Slider label={TEXT.x2[lang]} value={x2} min={0.15} max={3.1} step={0.01} onChange={setX2} format={(v) => v.toFixed(2)} />
        <Slider label={TEXT.p[lang]} value={p} min={0} max={1} step={0.01} onChange={setP} format={(v) => v.toFixed(2)} />
      </div>
      <div className="w-stats">
        <Stat label={TEXT.ef[lang]} value={ef.toFixed(3)} />
        <Stat label={TEXT.fe[lang]} value={fe.toFixed(3)} />
        <Stat label={TEXT.gap[lang]} value={(ef - fe).toFixed(3)} tone="accent" />
      </div>
    </Widget>
  );
}
