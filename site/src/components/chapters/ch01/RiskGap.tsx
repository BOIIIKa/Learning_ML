import { useMemo, useState } from "react";
import { empiricalAbsRisk, expAbsRisk, expSample, median } from "../../../engines/regression";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import Widget, { Slider, Stat } from "../../ui/Widget";

const TEXT = {
  title: { zh: "经验风险 vs 期望风险", en: "Empirical risk vs expected risk" },
  n: { zh: "样本量 n（对数刻度）", en: "Sample size n (log scale)" },
  resample: { zh: "↻ 重新抽样", en: "↻ Resample" },
  emp: { zh: "经验风险最小点（样本中位数）", en: "Empirical minimizer (sample median)" },
  true: { zh: "期望风险最小点（ln 2）", en: "Expected minimizer (ln 2)" },
  gap: { zh: "两者的期望风险之差", en: "Excess expected risk" },
  xl: { zh: "预测值 c", en: "Prediction c" },
  caption: {
    zh: "用常数 c 预测服从指数分布的 Y，损失取绝对值。紫色实线是真正关心的期望风险 R(c) = c − 1 + 2e^{−c}（只有知道分布才算得出），橙色折线是手上 n 个样本的经验风险。n 小时两条线差得远、最小点也差得远；n 增大时折线贴近实线，经验风险最小化（ERM）的解趋于最优解。",
    en: "Predict Y ~ Exponential(1) with a constant c under absolute loss. The purple curve is the expected risk R(c) = c − 1 + 2e^{−c} that we actually care about (computable only if we knew the distribution); the orange polyline is the empirical risk on the n samples in hand. For small n the two curves, and their minimizers, are far apart; as n grows the polyline hugs the curve and empirical risk minimization (ERM) approaches the optimum.",
  },
};

export default function RiskGap({ lang }: { lang: Lang }) {
  const [logN, setLogN] = useState(1);
  const [seed, setSeed] = useState(1);
  const n = Math.round(10 ** logN);
  const ys = useMemo(() => expSample(n, seed), [n, seed]);
  const m = median(ys);
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <CurvePlot xs={[]} ys={[]} xDomain={[0, 3]} yDomain={[0, 2.2]} xLabel={TEXT.xl[lang]}
        curves={[
          { f: expAbsRisk, color: "var(--accent-3)", width: 3 },
          { f: (c) => empiricalAbsRisk(ys, c), color: "var(--class-pos)", width: 2 },
        ]}
        vlines={[{ x: Math.LN2, color: "var(--accent-3)", label: "ln 2" }, { x: m, color: "var(--class-pos)", label: "median" }]} />
      <div className="w-row">
        <Slider label={TEXT.n[lang]} value={logN} min={0.5} max={3.3} step={0.05} onChange={setLogN} format={() => String(n)} />
        <button className="btn" onClick={() => setSeed(seed + 1)}>{TEXT.resample[lang]}</button>
      </div>
      <div className="w-stats">
        <Stat label={TEXT.emp[lang]} value={m.toFixed(3)} />
        <Stat label={TEXT.true[lang]} value={Math.LN2.toFixed(3)} />
        <Stat label={TEXT.gap[lang]} value={(expAbsRisk(m) - expAbsRisk(Math.LN2)).toFixed(4)} tone="accent" />
      </div>
    </Widget>
  );
}
