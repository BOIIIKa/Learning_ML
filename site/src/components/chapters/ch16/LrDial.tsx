import { useMemo, useState } from "react";
import { DIAL_STEPS, dial, students } from "../../../engines/optimization";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import MetricChart from "../../viz/MetricChart";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch16.css";

const TEXT = {
  title: { zh: "学习率旋钮：老师用梯度下降拟合成绩线", en: "The learning-rate dial: the teacher fits the score line by gradient descent" },
  raw: { zh: "原始小时数", en: "Raw hours" },
  std: { zh: "标准化小时数", en: "Standardised hours" },
  lr: { zh: "学习率 η", en: "Learning rate η" },
  loss: { zh: "log₁₀ 损失（100 步）", en: "log₁₀ loss (100 steps)" },
  best: { zh: "最小二乘", en: "Least squares" },
  kappa: { zh: "条件数 κ", en: "Condition number κ" },
  limit: { zh: "稳定上限 2/L", en: "Stability limit 2/L" },
  after: { zh: "100 步后的损失（最优 23.63）", en: "Loss after 100 steps (best 23.63)" },
  hit: { zh: "进入最优 1% 以内", en: "Within 1% of best" },
  never: { zh: "100 步内没有", en: "not in 100" },
  blown: { zh: "发散", en: "diverged" },
  steps: { zh: "步", en: "steps" },
  hours: { zh: "每周学习小时数", en: "Study hours per week" },
  score: { zh: "成绩", en: "Score" },
  caption: {
    zh: "老师用梯度下降拟合“成绩 = w₀ + w₁ × 小时数”，从 w = 0 出发走 100 步（实线是第 100 步的直线，淡线是第 1、3、10、30 步，虚线是最小二乘的答案）。用原始小时数时，学习率稍大于 0.059 就发散；不发散的最大学习率走完 100 步，截距仍然偏低。原因是损失曲面是一条又窄又长的山谷，条件数约 169。把小时数标准化（减均值、除以标准差）之后，曲面变成正圆，学习率可以放大到 1，一步就到最小二乘的答案。",
    en: "The teacher fits “score = w₀ + w₁ × hours” by gradient descent, starting from w = 0, for 100 steps (solid: the line at step 100; faint: steps 1, 3, 10, 30; dashed: the least-squares answer). With raw hours, a learning rate just above 0.059 diverges, and the largest rate that does not diverge still leaves the intercept too low after 100 steps. The loss surface is a long, narrow valley, with condition number about 169. Standardise the hours (subtract the mean, divide by the standard deviation) and the surface becomes round: the learning rate can go up to 1, and a single step lands on the least-squares answer.",
  },
};

const fmtLr = (s: number) => { const v = 10 ** s; return v >= 0.1 ? v.toFixed(2) : v.toFixed(v >= 0.01 ? 3 : 4); };

export default function LrDial({ lang }: { lang: Lang }) {
  const [std, setStd] = useState(false);
  const [s, setS] = useState(-1.5);
  const lr = 10 ** s;
  const r = useMemo(() => dial(std, lr), [std, lr]);
  const snap = [1, 3, 10, 30].filter((t) => t < r.path.length);
  const last = r.path[Math.min(DIAL_STEPS, r.path.length - 1)];
  const finalLoss = r.losses[DIAL_STEPS];
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="scenario">
      <div className="chips">
        <button className={`chip${!std ? " on" : ""}`} onClick={() => setStd(false)}>{TEXT.raw[lang]}</button>
        <button className={`chip${std ? " on" : ""}`} onClick={() => setStd(true)}>{TEXT.std[lang]}</button>
      </div>
      <div className="op-two">
        <CurvePlot xs={students.xs} ys={students.ys} xDomain={[0, 10]} yDomain={[0, 110]} xLabel={TEXT.hours[lang]} yLabel={TEXT.score[lang]} height={260}
          curves={[
            ...snap.map((t) => ({ f: r.line(r.path[t]), color: "var(--accent-2)", width: 1.5, opacity: 0.35 })),
            { f: r.line(r.wStar), color: "var(--text-muted)", dashed: true, width: 2 },
            ...(r.diverged ? [] : [{ f: r.line(last), color: "var(--class-pos)", width: 3 }]),
          ]} />
        <MetricChart zero={false} cursor={DIAL_STEPS} height={170}
          series={[
            { key: "l", label: TEXT.loss[lang], values: r.losses.map((l) => (Number.isFinite(l) && l < 1e8 ? Math.log10(l) : NaN)), color: "var(--class-pos)" },
            { key: "b", label: TEXT.best[lang], values: r.losses.map(() => Math.log10(r.best)), color: "var(--text-muted)", dashed: true },
          ]} />
      </div>
      <Slider label={TEXT.lr[lang]} value={s} min={-3} max={0.4} step={0.05} onChange={setS} format={fmtLr} />
      <div className="w-stats">
        <Stat label={TEXT.kappa[lang]} value={r.kappa.toFixed(0)} />
        <Stat label={TEXT.limit[lang]} value={(2 / r.L).toFixed(3)} tone={lr > 2 / r.L ? "bad" : undefined} />
        <Stat label={TEXT.after[lang]} value={Number.isFinite(finalLoss) && finalLoss < 1e6 ? finalLoss.toFixed(2) : TEXT.blown[lang]} tone="accent" />
        <Stat label={TEXT.hit[lang]} value={r.hit < 0 ? TEXT.never[lang] : `${r.hit} ${TEXT.steps[lang]}`} tone={r.hit >= 0 ? "good" : undefined} />
      </div>
    </Widget>
  );
}
