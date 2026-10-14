import { useMemo, useState } from "react";
import { biasVariance, hoursDesign, linspace, truth, X_HI, X_LO } from "../../../engines/generalization";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import ErrChart from "./ErrChart";
import FitPlot from "./FitPlot";

const TEXT = {
  title: { zh: "偏差–方差模拟器", en: "Bias–variance simulator" },
  degree: { zh: "多项式阶数 d", en: "Polynomial degree d" },
  sigma: { zh: "噪声标准差 σ", en: "Noise level σ" },
  resample: { zh: "↻ 换一批训练集", en: "↻ New training sets" },
  bias: { zh: "偏差²", en: "Bias²" },
  variance: { zh: "方差", en: "Variance" },
  noise: { zh: "噪声 σ²", en: "Noise σ²" },
  total: { zh: "期望测试误差", en: "Expected test error" },
  x: { zh: "每周学习时长（小时）", en: "Study hours per week" },
  dx: { zh: "阶数 d", en: "degree d" },
  caption: {
    zh: "同样的 12 个学习时长（顶部的小点），换 200 批学生（每批成绩的噪声不同）各拟合一次。细线是其中 20 条拟合曲线，粗橙线是 200 条的平均，虚线是真实规律。平均曲线离虚线的距离是偏差，细线散开的程度是方差。下图把三部分随阶数的变化画在一起：偏差² 下降、方差上升，总误差呈 U 形。阶数 0 的偏差和阶数 10 的方差都超出了纵轴范围。",
    en: "Keep the same 12 study-hour values (the dots along the top), draw 200 cohorts whose scores carry fresh noise, and fit each one. Thin lines are 20 of those fits, the thick orange line is the average of all 200, and the dashed line is the true trend. The distance from the average to the truth is the bias; the spread of the thin lines is the variance. The chart below tracks all three parts against the degree: bias² falls, variance rises, and their sum is U-shaped. The bias at degree 0 and the variance at degree 10 run off the top of the chart.",
  },
};

const GRID = linspace(X_LO, X_HI, 91);
const DEGREES = Array.from({ length: 11 }, (_, d) => d);
const R = 200;
const SHOWN = 20;

export default function BiasVarianceSim({ lang }: { lang: Lang }) {
  const [d, setD] = useState(3);
  const [sigma, setSigma] = useState(8);
  const [seed, setSeed] = useState(1);
  const runs = useMemo(() => DEGREES.map((k) => biasVariance(k, 12, sigma, R, seed, GRID)), [sigma, seed]);
  const cur = runs[d];
  const design = useMemo(() => hoursDesign(12), []);
  const total = cur.bias2 + cur.variance + cur.noise;
  const fan = cur.preds.slice(0, SHOWN).map((ys, i) => ({ ys, className: "g2-fan", key: `f${i}` }));

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <FitPlot grid={GRID} xDomain={[0, 10]} yDomain={[10, 100]} xLabel={TEXT.x[lang]} height={260}
        curves={[...fan, { ys: GRID.map(truth), className: "g2-truth", key: "truth" }, { ys: cur.mean, className: "g2-mean", key: "mean" }]}
        points={[{ xs: design, ys: design.map(() => 96), className: "g2-train g2-dim", r: 3.5 }]} />
      <div className="w-row">
        <Slider label={TEXT.degree[lang]} value={d} min={0} max={10} step={1} onChange={setD} />
        <Slider label={TEXT.sigma[lang]} value={sigma} min={2} max={14} step={1} onChange={setSigma} />
        <button className="btn" onClick={() => setSeed(seed + 1)}>{TEXT.resample[lang]}</button>
      </div>
      <div className="w-stats">
        <Stat label={TEXT.bias[lang]} value={cur.bias2.toFixed(1)} />
        <Stat label={TEXT.variance[lang]} value={cur.variance.toFixed(1)} />
        <Stat label={TEXT.noise[lang]} value={cur.noise.toFixed(0)} />
        <Stat label={TEXT.total[lang]} value={total.toFixed(1)} tone="accent" />
      </div>
      <ErrChart xs={DEGREES} cursor={d} xLabel={TEXT.dx[lang]} height={180} yDomain={[0, 2 * sigma * sigma + 100]}
        series={[
          { key: "b", label: TEXT.bias[lang], values: runs.map((r) => r.bias2), color: "var(--accent-3)" },
          { key: "v", label: TEXT.variance[lang], values: runs.map((r) => r.variance), color: "var(--success)" },
          { key: "n", label: TEXT.noise[lang], values: runs.map((r) => r.noise), color: "var(--text-muted)", dashed: true, width: 1.6 },
          { key: "t", label: TEXT.total[lang], values: runs.map((r) => r.bias2 + r.variance + r.noise), color: "var(--class-pos)", width: 3 },
        ]} />
    </Widget>
  );
}
