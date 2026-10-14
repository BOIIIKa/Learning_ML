import { scaleLinear } from "d3-scale";
import { useState } from "react";
import { sigmoid } from "../../../engines/generative";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch05.css";

const TEXT = {
  title: { zh: "后验的形状：共享方差 ⇒ sigmoid", en: "The shape of the posterior: shared variance ⇒ a sigmoid" },
  d: { zh: "两类均值间距", en: "Distance between the means" },
  ratio: { zh: "方差比 σ₁ / σ₀", en: "Spread ratio σ₁ / σ₀" },
  prior: { zh: "先验 P(y = 1)", en: "Prior P(y = 1)" },
  reset: { zh: "让两类共享方差", en: "Make the variances equal" },
  a: { zh: "x² 的系数 a", en: "Coefficient a of x²" },
  regions: { zh: "判为类 1 的区间数", en: "Intervals predicted as class 1" },
  dens: { zh: "π_k · p(x | k)", en: "π_k · p(x | k)" },
  post: { zh: "P(y = 1 | x)", en: "P(y = 1 | x)" },
  caption: {
    zh: "上图是两类的“先验 × 类条件密度”，下图是由贝叶斯公式得到的后验 P(y = 1 | x)，中间的色带是判决区域。对数几率是 x 的二次函数 a x² + b x + c，其中 a = 1/(2σ₀²) − 1/(2σ₁²)。方差比为 1 时 a = 0，后验恰好是一条 sigmoid——这就是逻辑回归的形式；方差不等时 a ≠ 0，方差大的那一类会在两侧尾部“反超”，判决区域变成两段。",
    en: "Top: each class's prior × class-conditional density. Bottom: the posterior P(y = 1 | x) given by Bayes' rule; the band in between shows the decision regions. The log-odds is a quadratic a x² + b x + c with a = 1/(2σ₀²) − 1/(2σ₁²). With a spread ratio of 1, a = 0 and the posterior is exactly a sigmoid — the logistic-regression form. With unequal spreads a ≠ 0: the wider class wins again in both tails, and its decision region splits in two.",
  },
};

const W = 460;
const H = 310;
const XS = Array.from({ length: 241 }, (_, i) => -5 + (10 * i) / 240);
const npdf = (x: number, m: number, s: number) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
const f1 = (v: number) => v.toFixed(1);

export default function PosteriorShape({ lang }: { lang: Lang }) {
  const [d, setD] = useState(2);
  const [ratio, setRatio] = useState(1.8);
  const [p1, setP1] = useState(0.5);
  const m0 = -d / 2;
  const m1 = d / 2;
  const s0 = 1;
  const s1 = ratio;
  const a = 1 / (2 * s0 * s0) - 1 / (2 * s1 * s1);
  const b = m1 / (s1 * s1) - m0 / (s0 * s0);
  const c = (m0 * m0) / (2 * s0 * s0) - (m1 * m1) / (2 * s1 * s1) + Math.log(s0 / s1) + Math.log(p1 / (1 - p1));
  const logit = (x: number) => a * x * x + b * x + c;

  const sx = scaleLinear().domain([-5, 5]).range([10, W - 10]);
  const g0 = (x: number) => (1 - p1) * npdf(x, m0, s0);
  const g1 = (x: number) => p1 * npdf(x, m1, s1);
  const top = Math.max(...XS.map((x) => Math.max(g0(x), g1(x)))) * 1.1;
  const syD = scaleLinear().domain([0, top]).range([150, 12]);
  const syP = scaleLinear().domain([0, 1]).range([280, 190]);
  const path = (g: (x: number) => number, s: (v: number) => number) => "M" + XS.map((x) => `${f1(sx(x))},${f1(s(g(x)))}`).join("L");
  const area = (g: (x: number) => number) => path(g, syD) + `L${f1(sx(5))},${f1(syD(0))}L${f1(sx(-5))},${f1(syD(0))}Z`;

  // Decision regions: maximal runs of the grid where the log-odds keeps one sign.
  const runs: { from: number; to: number; pos: boolean }[] = [];
  XS.forEach((x, i) => {
    const pos = logit(x) > 0;
    if (i === 0 || runs[runs.length - 1].pos !== pos) runs.push({ from: x, to: x, pos });
    else runs[runs.length - 1].to = x;
  });
  const step = XS[1] - XS[0];
  const posRuns = runs.filter((r) => r.pos).length;

  const fmt = (v: number) => (Math.abs(v) < 5e-4 ? "0" : v.toFixed(2));
  const sign = (v: number) => (v < 0 ? "−" : "+");

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <svg className="g5-svg" viewBox={`0 0 ${W} ${H}`} role="img">
        <rect width={W} height={H} rx={12} className="g5-bg" />
        <path d={area(g0)} className="g5-fill neg" />
        <path d={area(g1)} className="g5-fill pos" />
        <path d={path(g0, syD)} className="g5-dens neg" />
        <path d={path(g1, syD)} className="g5-dens pos" />
        {runs.map((r, i) => (
          <rect key={i} x={f1(sx(Math.max(-5, r.from - step / 2)))} y={158} width={f1(sx(Math.min(5, r.to + step / 2)) - sx(Math.max(-5, r.from - step / 2)))} height={10}
            className={`g5-band ${r.pos ? "pos" : "neg"}`} />
        ))}
        {runs.slice(1).map((r, i) => (
          <line key={`t${i}`} x1={f1(sx(r.from - step / 2))} x2={f1(sx(r.from - step / 2))} y1={12} y2={280} className="g5-thr" />
        ))}
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={10} x2={W - 10} y1={syP(v)} y2={syP(v)} className="g5-grid" />
            <text x={W - 14} y={syP(v) - 3} textAnchor="end" className="g5-tick">{v}</text>
          </g>
        ))}
        <path d={path((x) => sigmoid(logit(x)), syP)} className="g5-post" />
        <text x={16} y={24} className="g5-lab">{TEXT.dens[lang]}</text>
        <text x={16} y={204} className="g5-lab">{TEXT.post[lang]}</text>
        {[-4, -2, 0, 2, 4].map((t) => <text key={t} x={sx(t)} y={300} textAnchor="middle" className="g5-tick">{t}</text>)}
      </svg>
      <div className="g5-formula">
        log-odds(x) = <b>{fmt(a)}</b>·x² {sign(b)} {fmt(Math.abs(b))}·x {sign(c)} {fmt(Math.abs(c))}
      </div>
      <div className="w-row">
        <Slider label={TEXT.ratio[lang]} value={ratio} min={0.4} max={2.5} step={0.01} onChange={setRatio} format={(v) => v.toFixed(2)} />
        <button className="btn" onClick={() => setRatio(1)}>{TEXT.reset[lang]}</button>
      </div>
      <div className="w-row">
        <Slider label={TEXT.d[lang]} value={d} min={0} max={4} step={0.05} onChange={setD} format={(v) => v.toFixed(2)} />
        <Slider label={TEXT.prior[lang]} value={p1} min={0.1} max={0.9} step={0.01} onChange={setP1} format={(v) => v.toFixed(2)} />
      </div>
      <div className="w-stats">
        <Stat label={TEXT.a[lang]} value={fmt(a)} tone={Math.abs(a) < 5e-4 ? "good" : "accent"} />
        <Stat label={TEXT.regions[lang]} value={String(posRuns)} />
      </div>
    </Widget>
  );
}
