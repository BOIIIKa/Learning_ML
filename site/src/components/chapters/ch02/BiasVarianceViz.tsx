import { useMemo, useState } from "react";
import { biasVariance, NOISE, truth } from "../../../engines/generalization";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import MetricChart from "../../viz/MetricChart";
import Widget, { Slider, Stat } from "../../ui/Widget";

const TEXT = {
  title: { zh: "偏差–方差分解：换 30 批数据看看", en: "Bias–variance: refit on 30 different datasets" },
  degree: { zh: "多项式次数", en: "Polynomial degree" },
  bias: { zh: "偏差²", en: "Bias²" },
  variance: { zh: "方差", en: "Variance" },
  noise: { zh: "噪声 σ²", en: "Noise σ²" },
  total: { zh: "期望测试误差", en: "Expected test error" },
  caption: {
    zh: "每条淡色细线是在一批新抽的 20 个学生上拟合的结果，粗线是 300 次拟合的平均（图中只画出 30 条），虚线是真实规律。次数低：细线挤在一起（方差小），但平均线偏离真实（偏差大）；次数高：平均线贴近真实（偏差小），细线却四散（方差大）。期望测试误差 = 偏差² + 方差 + 噪声。",
    en: "Each faint line is a fit on a freshly drawn set of 20 students; the bold line is the average of 300 fits (30 are drawn) and the dashed curve is the truth. Low degree: the faint lines huddle together (low variance) but their average misses the truth (high bias). High degree: the average hugs the truth (low bias) but the lines scatter (high variance). Expected test error = bias² + variance + noise.",
  },
};

const MAXD = 8;

export default function BiasVarianceViz({ lang }: { lang: Lang }) {
  const [d, setD] = useState(1);
  const all = useMemo(() => Array.from({ length: MAXD + 1 }, (_, k) => biasVariance(k, 20, 300)), []);
  const cur = all[d];
  const shown = cur.fits.slice(0, 30);
  const mean = (x: number) => cur.fits.reduce((s, f) => s + f(x), 0) / cur.fits.length;
  const log = (v: number) => Math.log10(v);
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <CurvePlot xs={[]} ys={[]} xDomain={[0, 10]} yDomain={[0, 110]} height={290}
        curves={[
          ...shown.map((f) => ({ f, color: "var(--accent)", width: 1, opacity: 0.35 })),
          { f: truth, color: "var(--text)", dashed: true, width: 2 },
          { f: mean, color: "var(--class-pos)", width: 3.2 },
        ]} />
      <Slider label={TEXT.degree[lang]} value={d} min={0} max={MAXD} step={1} onChange={setD} />
      <MetricChart zero={false} cursor={d} height={130} series={[
        { key: "b", label: `log₁₀ ${TEXT.bias[lang]}`, values: all.map((r) => log(r.bias2)), color: "var(--class-pos)" },
        { key: "v", label: `log₁₀ ${TEXT.variance[lang]}`, values: all.map((r) => log(r.variance)), color: "var(--accent)" },
        { key: "t", label: `log₁₀ ${TEXT.total[lang]}`, values: all.map((r) => log(r.bias2 + r.variance + NOISE ** 2)), color: "var(--accent-2)", dashed: true },
      ]} />
      <div className="w-stats">
        <Stat label={TEXT.bias[lang]} value={cur.bias2.toFixed(1)} />
        <Stat label={TEXT.variance[lang]} value={cur.variance.toFixed(1)} />
        <Stat label={TEXT.noise[lang]} value={(NOISE ** 2).toFixed(0)} />
        <Stat label={TEXT.total[lang]} value={(cur.bias2 + cur.variance + NOISE ** 2).toFixed(1)} tone="accent" />
      </div>
    </Widget>
  );
}
