import { scaleLinear } from "d3-scale";
import { useState } from "react";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";

const TEXT = {
  title: { zh: "贝叶斯分类器与贝叶斯错误率", en: "The Bayes classifier and the Bayes error" },
  prior: { zh: "先验 P(良性)", en: "Prior P(benign)" },
  sep: { zh: "两类均值间距", en: "Distance between class means" },
  thr: { zh: "你的阈值", en: "Your threshold" },
  bayes: { zh: "贝叶斯阈值", en: "Bayes threshold" },
  yourErr: { zh: "你的错误率", en: "Your error rate" },
  bayesErr: { zh: "贝叶斯错误率（下限）", en: "Bayes error (the floor)" },
  caption: {
    zh: "两类的密度乘以先验后画成两座山（蓝 = 良性，橙 = 恶性）。阈值左边判良性、右边判恶性；阴影是犯错的概率。贝叶斯阈值取在两座山相交处，此时阴影面积最小——任何分类器都做不到更低。拖动你的阈值，它永远不会低于贝叶斯错误率。",
    en: "Each class density times its prior is drawn as a hill (blue = benign, orange = malignant). Left of the threshold we predict benign, right of it malignant; the shaded area is the probability of a mistake. The Bayes threshold sits where the hills cross and minimizes that area — no classifier can do better. Drag your threshold: it never beats the Bayes error.",
  },
};

const W = 460;
const H = 260;
const phi = (x: number, m: number) => Math.exp(-0.5 * (x - m) ** 2) / Math.sqrt(2 * Math.PI);
// Standard normal CDF via the Abramowitz–Stegun erf approximation (|error| < 1.5e-7).
function Phi(z: number) {
  const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * Math.exp(-(z * z) / 2);
  return z >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
}

export default function BayesClassifier({ lang }: { lang: Lang }) {
  const [p0, setP0] = useState(0.6);
  const [d, setD] = useState(2);
  const [thr, setThr] = useState(0.4);
  const m0 = -d / 2;
  const m1 = d / 2;
  const p1 = 1 - p0;
  const bayes = (m0 + m1) / 2 + Math.log(p0 / p1) / (m1 - m0);
  const err = (t: number) => p0 * (1 - Phi(t - m0)) + p1 * Phi(t - m1);
  const sx = scaleLinear().domain([-4.5, 4.5]).range([10, W - 10]);
  const top = Math.max(p0, p1) * phi(0, 0) * 1.15;
  const sy = scaleLinear().domain([0, top]).range([H - 24, 10]);
  const grid = Array.from({ length: 181 }, (_, i) => -4.5 + (9 * i) / 180);
  const path = (g: (x: number) => number) => "M" + grid.map((x) => `${sx(x).toFixed(1)},${sy(g(x)).toFixed(1)}`).join("L");
  const area = (g: (x: number) => number, lo: number, hi: number) => {
    const pts = grid.filter((x) => x >= lo && x <= hi);
    if (pts.length < 2) return "";
    return `M${sx(pts[0]).toFixed(1)},${sy(0)}` + pts.map((x) => `L${sx(x).toFixed(1)},${sy(g(x)).toFixed(1)}`).join("") + `L${sx(pts.at(-1)!).toFixed(1)},${sy(0)}Z`;
  };
  const g0 = (x: number) => p0 * phi(x, m0);
  const g1 = (x: number) => p1 * phi(x, m1);

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <svg className="bayes-svg" viewBox={`0 0 ${W} ${H}`} role="img">
        <rect width={W} height={H} rx={12} className="plot-bg" />
        <path d={area(g0, thr, 4.5)} className="bc-err neg" />
        <path d={area(g1, -4.5, thr)} className="bc-err pos" />
        <path d={path(g0)} className="bc-curve neg" />
        <path d={path(g1)} className="bc-curve pos" />
        <line x1={sx(bayes)} x2={sx(bayes)} y1={10} y2={H - 24} className="bc-bayes" />
        <line x1={sx(thr)} x2={sx(thr)} y1={10} y2={H - 24} className="bc-thr" />
        <text x={sx(bayes) + 4} y={22} className="bc-lab bayes">Bayes</text>
        <line x1={10} x2={W - 10} y1={sy(0)} y2={sy(0)} className="lm-axis" />
      </svg>
      <div className="w-row">
        <Slider label={TEXT.thr[lang]} value={thr} min={-3} max={3} step={0.01} onChange={setThr} format={(v) => v.toFixed(2)} />
        <Slider label={TEXT.prior[lang]} value={p0} min={0.1} max={0.9} step={0.01} onChange={setP0} format={(v) => v.toFixed(2)} />
        <Slider label={TEXT.sep[lang]} value={d} min={0.5} max={4} step={0.05} onChange={setD} format={(v) => v.toFixed(2)} />
      </div>
      <div className="w-stats">
        <Stat label={TEXT.yourErr[lang]} value={`${(100 * err(thr)).toFixed(2)}%`} tone={err(thr) - err(bayes) < 5e-4 ? "good" : undefined} />
        <Stat label={TEXT.bayesErr[lang]} value={`${(100 * err(bayes)).toFixed(2)}%`} tone="accent" />
        <Stat label={TEXT.bayes[lang]} value={bayes.toFixed(2)} />
      </div>
    </Widget>
  );
}
