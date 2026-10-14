import { useMemo, useState } from "react";
import { doubleDescentCurve } from "../../../engines/generalization";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import Widget, { Slider, Stat } from "../../ui/Widget";

const TEXT = {
  title: { zh: "双下降：越过插值阈值之后", en: "Double descent: beyond the interpolation threshold" },
  p: { zh: "使用的特征数 p", en: "Features used p" },
  risk: { zh: "测试风险", en: "Test risk" },
  regime: { zh: "所处区域", en: "Regime" },
  under: { zh: "欠参数化（p < n）", en: "Under-parametrized (p < n)" },
  at: { zh: "插值阈值（p ≈ n）", en: "Interpolation threshold (p ≈ n)" },
  over: { zh: "过参数化（p > n）", en: "Over-parametrized (p > n)" },
  xl: { zh: "特征数 p（n = 30）", en: "Features p (n = 30)" },
  caption: {
    zh: "n = 30 个样本，真实信号分散在 100 个特征上；模型只用前 p 个特征做最小范数最小二乘。经典的 U 形曲线在 p < n 时出现；p = n 时模型恰好能插值所有训练点，测试风险冲到峰值；继续增大 p，在所有插值解里选范数最小的那个，测试风险反而再次下降。纵轴为对数刻度。",
    en: "n = 30 samples, with the true signal spread over 100 features; the model uses only the first p features and fits them by minimum-norm least squares. The classical U shape appears for p < n; at p = n the model can just interpolate every training point and test risk spikes; as p grows further, picking the smallest-norm interpolant makes test risk fall again. Log scale on the vertical axis.",
  },
};

export default function DoubleDescent({ lang }: { lang: Lang }) {
  const { ps, errs, n } = useMemo(() => doubleDescentCurve(30, 100, 12), []);
  const [p, setP] = useState(15);
  const log = (v: number) => Math.log10(v);
  const f = (x: number) => { const i = Math.max(0, Math.min(ps.length - 1, Math.round(x) - 1)); return log(errs[i]); };
  const regime = Math.abs(p - n) <= 2 ? TEXT.at[lang] : p < n ? TEXT.under[lang] : TEXT.over[lang];
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="frontier">
      <CurvePlot xs={[p]} ys={[log(errs[p - 1])]} xDomain={[1, 100]} yDomain={[Math.min(...errs.map(log)) - 0.1, Math.max(...errs.map(log)) + 0.1]}
        xLabel={TEXT.xl[lang]} yLabel={`log₁₀ ${TEXT.risk[lang]}`}
        curves={[{ f, color: "var(--accent-2)", width: 2.6 }]} vlines={[{ x: n, color: "var(--warning)", label: "p = n" }]} />
      <Slider label={TEXT.p[lang]} value={p} min={1} max={100} step={1} onChange={setP} />
      <div className="w-stats">
        <Stat label={TEXT.risk[lang]} value={errs[p - 1].toFixed(2)} tone="accent" />
        <Stat label={TEXT.regime[lang]} value={regime} />
      </div>
    </Widget>
  );
}
