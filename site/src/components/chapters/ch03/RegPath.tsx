import { scaleLinear } from "d3-scale";
import { useId, useMemo, useState } from "react";
import { fuelTruth, regPath, type Penalty } from "../../../engines/linreg";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch03.css";

const TEXT = {
  title: { zh: "正则化路径：一个 λ，三张图", en: "The regularization path: one λ, three views" },
  ridge: { zh: "岭回归（L2）", en: "Ridge (L2)" },
  lasso: { zh: "Lasso（L1）", en: "Lasso (L1)" },
  lam: { zh: "正则强度 λ", en: "Strength λ" },
  best: { zh: "跳到交叉验证最优 λ", en: "Jump to the CV-best λ" },
  x: { zh: "车速（km/h）", en: "Speed (km/h)" },
  y: { zh: "油耗（L/100km）", en: "Fuel (L/100 km)" },
  coef: { zh: "9 个系数随 λ 的变化", en: "The 9 coefficients versus λ" },
  err: { zh: "均方误差（对数刻度）", en: "Mean squared error (log scale)" },
  cv: { zh: "4 折交叉验证", en: "4-fold CV" },
  train: { zh: "训练集", en: "Training" },
  nnz: { zh: "非零系数", en: "Non-zero coefs" },
  cvNow: { zh: "交叉验证误差", en: "CV error" },
  trNow: { zh: "训练误差", en: "Training error" },
  norm: { zh: "‖w‖₂", en: "‖w‖₂" },
  caption: {
    zh: "12 次试车测量（车速 → 油耗），用 9 次多项式拟合，特征标准化后不惩罚截距。上图：拟合曲线（橙）与真实规律（虚线）；左下：9 个系数（u, u², …, u⁹ 的权重）随 λ 的变化，竖线是当前 λ；右下：训练误差只会随 λ 增大而上升，交叉验证误差却先降后升，紫点是它的最低处。切换岭回归与 Lasso：岭回归把所有系数一起往 0 拉，却几乎从不等于 0；Lasso 让系数一个接一个地变成恰好 0。目标函数：岭回归 (1/2n)‖y−Zw‖² + (λ/2)‖w‖²，Lasso (1/2n)‖y−Zw‖² + λ‖w‖₁。数据为合成示意。",
    en: "Twelve test-track measurements (speed → fuel use) fitted with a degree-9 polynomial; features are standardized and the intercept is not penalized. Top: the fitted curve (orange) and the true law (dashed). Bottom left: the 9 coefficients (weights on u, u², …, u⁹) versus λ, with the current λ as a vertical line. Bottom right: training error only rises with λ, while cross-validation error first falls and then rises; the purple dot is its minimum. Switch between ridge and lasso: ridge pulls all coefficients towards 0 together but almost never to exactly 0; lasso sets them to exactly 0 one after another. Objectives: ridge (1/2n)‖y−Zw‖² + (λ/2)‖w‖², lasso (1/2n)‖y−Zw‖² + λ‖w‖₁. Synthetic illustrative data.",
  },
};

export const COEF_COLORS = [
  "var(--accent)", "var(--accent-2)", "var(--accent-3)", "var(--success)", "var(--warning)",
  "var(--class-neg)", "var(--class-pos)", "var(--danger)", "var(--text-muted)",
];

const W = 300;
const H = 170;
const PAD = { l: 34, r: 8, t: 8, b: 20 };
const f1 = (v: number) => v.toFixed(1);

export default function RegPath({ lang, start = 4, initial = "ridge" }: { lang: Lang; start?: number; initial?: Penalty }) {
  const [pen, setPen] = useState<Penalty>(initial);
  const [k, setK] = useState(start);
  const clip = `rp-clip-${useId().replace(/:/g, "")}`;
  const paths = useMemo(() => ({ ridge: regPath("ridge"), lasso: regPath("lasso") }), []);
  const p = paths[pen];
  const n = p.lams.length;
  const w = p.ws[k];
  const f = (v: number) => p.predict(w, v);
  const best = p.cv.indexOf(Math.min(...p.cv));
  const nnz = w.filter((v) => v !== 0).length;

  const sx = scaleLinear().domain([0, n - 1]).range([PAD.l, W - PAD.r]);
  const cy = scaleLinear().domain([-3, 3]).range([H - PAD.b, PAD.t]);
  const clampC = (v: number) => cy(Math.max(-30, Math.min(30, v)));
  const all = [...p.cv, ...p.train];
  const ey = scaleLinear().domain([Math.log10(Math.min(...all)) - 0.1, Math.log10(Math.max(...all)) + 0.1]).range([H - PAD.b, PAD.t]);
  const lg = (v: number) => ey(Math.log10(v));
  const lamTicks = [0, 8, 16, 24, 32, 40].filter((i) => i < n);
  const lamLabel = (i: number) => { const e = Math.round(Math.log10(p.lams[i])); return e === 0 ? "1" : `1e${e}`; };
  const xAxis = lamTicks.map((i) => <text key={i} x={sx(i)} y={H - 6} textAnchor="middle" className="c3-tick">{lamLabel(i)}</text>);
  const line = (vals: number[], y: (v: number) => number) => "M" + vals.map((v, i) => `${f1(sx(i))},${f1(y(v))}`).join("L");
  const errTicks = [0.03, 0.1, 0.3, 1, 3, 10].filter((v) => lg(v) >= PAD.t && lg(v) <= H - PAD.b);

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="chips">
        {(["ridge", "lasso"] as Penalty[]).map((q) => (
          <button key={q} className={`chip${q === pen ? " on" : ""}`} onClick={() => setPen(q)}>{TEXT[q][lang]}</button>
        ))}
      </div>
      <CurvePlot xs={p.xs} ys={p.ys} xDomain={[20, 120]} yDomain={[2, 9]} xLabel={TEXT.x[lang]} yLabel={TEXT.y[lang]} height={280}
          curves={[{ f: fuelTruth, color: "var(--text-muted)", dashed: true, width: 1.6 }, { f, color: "var(--class-pos)", width: 2.8 }]} />
      <div className="c3-pair">
          <figure className="c3-sub">
            <svg className="c3-svg" viewBox={`0 0 ${W} ${H}`} role="img">
              <rect width={W} height={H} rx={10} className="c3-bg" />
              {[-3, -2, -1, 0, 1, 2, 3].map((t) => (
                <g key={t}>
                  <line x1={PAD.l} x2={W - PAD.r} y1={cy(t)} y2={cy(t)} className={t === 0 ? "c3-axis" : "c3-grid"} />
                  <text x={PAD.l - 5} y={cy(t)} textAnchor="end" dominantBaseline="middle" className="c3-tick">{t}</text>
                </g>
              ))}
              {xAxis}
              <clipPath id={clip}><rect x={PAD.l} y={PAD.t} width={W - PAD.l - PAD.r} height={H - PAD.t - PAD.b} /></clipPath>
              <g clipPath={`url(#${clip})`}>
                {w.map((_, j) => <path key={j} d={line(p.ws.map((ww) => ww[j]), clampC)} stroke={COEF_COLORS[j]} className="rp-path" />)}
              </g>
              <line x1={sx(k)} x2={sx(k)} y1={PAD.t} y2={H - PAD.b} className="rp-cursor" />
            </svg>
            <figcaption>{TEXT.coef[lang]}</figcaption>
          </figure>
          <figure className="c3-sub">
            <svg className="c3-svg" viewBox={`0 0 ${W} ${H}`} role="img">
              <rect width={W} height={H} rx={10} className="c3-bg" />
              {errTicks.map((t) => (
                <g key={t}>
                  <line x1={PAD.l} x2={W - PAD.r} y1={lg(t)} y2={lg(t)} className="c3-grid" />
                  <text x={PAD.l - 5} y={lg(t)} textAnchor="end" dominantBaseline="middle" className="c3-tick">{t}</text>
                </g>
              ))}
              {xAxis}
              <path d={line(p.train, lg)} className="rp-train" />
              <path d={line(p.cv, lg)} className="rp-cv" />
              <circle cx={sx(best)} cy={lg(p.cv[best])} r={5} className="rp-best" />
              <line x1={sx(k)} x2={sx(k)} y1={PAD.t} y2={H - PAD.b} className="rp-cursor" />
            </svg>
            <figcaption className="rp-legend">
              <span><i style={{ background: "var(--accent-3)" }} />{TEXT.cv[lang]}</span>
              <span><i style={{ background: "var(--accent)" }} />{TEXT.train[lang]}</span>
              <span>{TEXT.err[lang]}</span>
            </figcaption>
          </figure>
      </div>
      <div className="w-row">
        <Slider label={TEXT.lam[lang]} value={k} min={0} max={n - 1} step={1} onChange={setK} format={(i) => p.lams[i].toExponential(1)} />
        <button className="btn" onClick={() => setK(best)}>{TEXT.best[lang]}</button>
      </div>
      <div className="w-stats">
        <Stat label={TEXT.nnz[lang]} value={`${nnz} / ${w.length}`} tone={nnz < w.length ? "accent" : undefined} />
        <Stat label={TEXT.norm[lang]} value={Math.hypot(...w).toFixed(2)} />
        <Stat label={TEXT.trNow[lang]} value={p.train[k].toFixed(3)} />
        <Stat label={TEXT.cvNow[lang]} value={p.cv[k].toFixed(3)} tone={k === best ? "good" : undefined} />
      </div>
    </Widget>
  );
}
