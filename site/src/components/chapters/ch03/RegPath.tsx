import { scaleLinear } from "d3-scale";
import { useMemo, useState } from "react";
import { sample, truth } from "../../../engines/generalization";
import { logGrid, regPath, type Penalty } from "../../../engines/linreg";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import MetricChart from "../../viz/MetricChart";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch03.css";

const TEXT = {
  title: { zh: "正则化路径：一根滑块控制一切", en: "The regularization path: one slider controls everything" },
  ridge: { zh: "岭回归", en: "Ridge" },
  lasso: { zh: "Lasso", en: "Lasso" },
  lam: { zh: "λ（左强右弱）", en: "λ (strong ← → weak)" },
  coefs: { zh: "9 个系数随 λ 的变化", en: "The 9 coefficients as λ varies" },
  cv: { zh: "log₁₀ 5 折 CV 误差", en: "log₁₀ 5-fold CV error" },
  nnz: { zh: "非零系数个数", en: "Nonzero coefficients" },
  cvNow: { zh: "当前 CV 误差", en: "CV error now" },
  best: { zh: "CV 最优 λ", en: "Best λ by CV" },
  caption: {
    zh: "仍是第 02 章的 30 位学生，模型是 9 次多项式（特征已标准化）。中间的图是系数路径：从左到右惩罚变弱，系数从 0 逐渐长出来。岭回归的系数一起平滑地收缩，永远不会恰好为 0；Lasso 的系数一个接一个“进场”，右侧之前都恰好为 0——它在拟合的同时做了特征选择。下方是交叉验证误差，最低点给出推荐的 λ。",
    en: "The same 30 students as in Chapter 02, now with a degree-9 polynomial (standardized features). The middle chart is the coefficient path: from left to right the penalty weakens and coefficients grow out of 0. Ridge shrinks all coefficients smoothly together and never sets one exactly to 0; lasso coefficients enter one at a time and are exactly 0 until they do, so it selects features while it fits. The bottom chart is the cross-validated error; its minimum gives the recommended λ.",
  },
};

const LAMS = logGrid(1e-3, 30, 40);
const COLORS = ["var(--accent)", "var(--accent-2)", "var(--class-pos)", "var(--ink)", "var(--success)", "var(--danger)", "var(--text-muted)", "var(--class-neg)", "var(--warning)"];
const W = 460;
const H = 170;
const f1 = (v: number) => v.toFixed(1);

export default function RegPath({ lang }: { lang: Lang }) {
  const data = useMemo(() => sample(30, 2), []);
  const [pen, setPen] = useState<Penalty>("lasso");
  const paths = useMemo(() => ({
    ridge: regPath(data.xs, data.ys, 9, "ridge", LAMS),
    lasso: regPath(data.xs, data.ys, 9, "lasso", LAMS),
  }), [data]);
  const cur = paths[pen];
  const best = cur.cv.indexOf(Math.min(...cur.cv));
  const [i, setI] = useState(best);
  const w = cur.ws[i];
  const nnz = w.filter((v) => Math.abs(v) > 1e-9).length;

  const lim = Math.max(...cur.ws.flat().map(Math.abs), 1);
  const sx = scaleLinear().domain([0, LAMS.length - 1]).range([10, W - 10]);
  const sy = scaleLinear().domain([-lim, lim]).range([H - 10, 10]);

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="chips">
        {(["ridge", "lasso"] as Penalty[]).map((p) => (
          <button key={p} className={`chip${pen === p ? " on" : ""}`} onClick={() => setPen(p)}>{TEXT[p][lang]}</button>
        ))}
      </div>
      <CurvePlot xs={data.xs} ys={data.ys} xDomain={[0, 10]} yDomain={[0, 110]} height={260}
        curves={[{ f: truth, color: "var(--text-muted)", dashed: true, width: 1.6 }, { f: cur.fs[i], color: "var(--class-pos)" }]} />
      <Slider label={TEXT.lam[lang]} value={i} min={0} max={LAMS.length - 1} step={1} onChange={setI}
        format={(v) => `λ = ${LAMS[v] >= 1 ? LAMS[v].toFixed(1) : LAMS[v].toPrecision(2)}`} />
      <div className="lr-sub">{TEXT.coefs[lang]}</div>
      <svg className="lr-svg path" viewBox={`0 0 ${W} ${H}`} role="img">
        <rect width={W} height={H} className="plot-bg" />
        <line x1={10} x2={W - 10} y1={f1(sy(0))} y2={f1(sy(0))} className="lm-axis" />
        {w.map((_, j) => (
          <path key={`${pen}${j}`} d={"M" + cur.ws.map((ws, k) => `${f1(sx(k))},${f1(sy(ws[j]))}`).join("L")}
            stroke={COLORS[j]} pathLength={1} className="lr-coef draw" />
        ))}
        <line x1={f1(sx(i))} x2={f1(sx(i))} y1={6} y2={H - 6} className="mc-cursor" />
        {w.map((v, j) => <circle key={j} cx={f1(sx(i))} cy={f1(sy(v))} r={3} fill={COLORS[j]} />)}
      </svg>
      <MetricChart zero={false} cursor={i} height={110}
        series={[{ key: "cv", label: TEXT.cv[lang], values: cur.cv.map((v) => Math.log10(v)), color: "var(--accent)" }]} />
      <div className="w-stats">
        <Stat label={TEXT.nnz[lang]} value={`${nnz} / 9`} tone={pen === "lasso" ? "accent" : undefined} />
        <Stat label={TEXT.cvNow[lang]} value={cur.cv[i].toFixed(1)} />
        <Stat label={TEXT.best[lang]} value={LAMS[best] >= 1 ? LAMS[best].toFixed(1) : LAMS[best].toPrecision(2)} tone="good" />
      </div>
    </Widget>
  );
}
