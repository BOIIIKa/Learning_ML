import { scaleLinear } from "d3-scale";
import { useState } from "react";
import { MN, gdUnder, underLimit } from "../../../engines/dl-generalization";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch18.css";

const TEXT = {
  title: { zh: "无数个解，梯度下降挑哪一个？", en: "Infinitely many solutions: which one does gradient descent pick?" },
  x0: { zh: "起点 w₀ 的第 1 个分量", en: "Start w₀, first coordinate" },
  y0: { zh: "起点 w₀ 的第 2 个分量", en: "Start w₀, second coordinate" },
  steps: { zh: "梯度下降步数", en: "Gradient-descent steps" },
  origin: { zh: "从原点出发", en: "Start at the origin" },
  limit: { zh: "收敛到的解", en: "Converges to" },
  norm: { zh: "它的范数 / 最小范数", en: "Its norm / the minimum norm" },
  caption: {
    zh: "一个方程 w₁ + 2w₂ = 3、两个未知数：直线上每一点都是零训练误差的解。梯度 a(aᵀw − y) 永远平行于 a = (1, 2)，所以梯度下降沿着垂直于解直线的方向走直线，停在离起点最近的解上。拖动起点：从原点出发时，它停在最小范数解 (0.6, 1.2)，范数 1.342，正好是圆与直线的切点。没有人要求最小范数，是算法自己选了它：这就是隐式正则。",
    en: "One equation w₁ + 2w₂ = 3 in two unknowns: every point on the line is a solution with zero training error. The gradient a(aᵀw − y) is always parallel to a = (1, 2), so gradient descent walks straight across, perpendicular to the line of solutions, and stops at the solution nearest its start. Drag the start: from the origin it stops at the minimum-norm solution (0.6, 1.2) with norm 1.342, where the circle touches the line. Nobody asked for the minimum norm; the algorithm picked it. That is implicit regularisation.",
  },
};

const W = 340, H = 300;
const f1 = (v: number) => v.toFixed(1);

export default function MinNormGD({ lang }: { lang: Lang }) {
  const [x0, setX0] = useState(2.5);
  const [y0, setY0] = useState(2.5);
  const [steps, setSteps] = useState(40);
  const sx = scaleLinear().domain([-2.2, 3.2]).range([20, W - 10]);
  const sy = scaleLinear().domain([-1.6, 3.2]).range([H - 20, 10]);
  const path = gdUnder([x0, y0], steps);
  const lim = underLimit([x0, y0]);
  const star = underLimit([0, 0]);
  const rStar = Math.hypot(star[0], star[1]);
  const lineY = (x: number) => (MN.y - MN.a[0] * x) / MN.a[1];
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="chips">
        <button className="chip" onClick={() => { setX0(0); setY0(0); }}>{TEXT.origin[lang]}</button>
      </div>
      <svg className="lm-svg dg-mn" viewBox={`0 0 ${W} ${H}`} role="img">
        <line x1={f1(sx(0))} x2={f1(sx(0))} y1={10} y2={H - 20} className="lm-axis" />
        <line x1={20} x2={W - 10} y1={f1(sy(0))} y2={f1(sy(0))} className="lm-axis" />
        <text x={W - 14} y={f1(sy(0) - 6)} textAnchor="end" className="dg-label">w₁</text>
        <text x={f1(sx(0) + 6)} y={18} className="dg-label">w₂</text>
        <ellipse cx={f1(sx(0))} cy={f1(sy(0))} rx={f1(sx(rStar) - sx(0))} ry={f1(sy(0) - sy(rStar))} className="dg-circle" />
        <line x1={f1(sx(-2.2))} y1={f1(sy(lineY(-2.2)))} x2={f1(sx(3.2))} y2={f1(sy(lineY(3.2)))} className="dg-solline" />
        <line x1={f1(sx(x0))} y1={f1(sy(y0))} x2={f1(sx(lim[0]))} y2={f1(sy(lim[1]))} className="dg-proj" />
        {path.map(([a, b], i) => <circle key={i} cx={f1(sx(a))} cy={f1(sy(b))} r={i === 0 ? 5 : 2.6} className={i === 0 ? "dg-start" : "dg-step"} />)}
        <circle cx={f1(sx(star[0]))} cy={f1(sy(star[1]))} r={4.5} className="dg-star" />
        <text x={f1(sx(star[0]) - 8)} y={f1(sy(star[1]) + 16)} textAnchor="end" className="dg-label">w_min</text>
        <circle cx={f1(sx(lim[0]))} cy={f1(sy(lim[1]))} r={6} className="dg-pt-ridge" />
      </svg>
      <Slider label={TEXT.x0[lang]} value={x0} min={-2} max={3} step={0.1} onChange={setX0} format={(v) => v.toFixed(1)} />
      <Slider label={TEXT.y0[lang]} value={y0} min={-1.5} max={3} step={0.1} onChange={setY0} format={(v) => v.toFixed(1)} />
      <Slider label={TEXT.steps[lang]} value={steps} min={0} max={40} step={1} onChange={setSteps} />
      <div className="w-stats">
        <Stat label={TEXT.limit[lang]} value={`(${lim[0].toFixed(2)}, ${lim[1].toFixed(2)})`} tone="accent" />
        <Stat label={TEXT.norm[lang]} value={`${Math.hypot(lim[0], lim[1]).toFixed(3)} / ${rStar.toFixed(3)}`} tone={Math.hypot(lim[0], lim[1]) - rStar < 1e-9 ? "good" : undefined} />
      </div>
    </Widget>
  );
}
