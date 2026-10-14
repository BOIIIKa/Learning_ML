import { useMemo, useState } from "react";
import { avgNll, coverage, fitHetero, fitHomo, heteroData } from "../../../engines/losses-as-likelihood";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import Widget, { Stat } from "../../ui/Widget";
import "./ch15.css";

const TEXT = {
  title: { zh: "预测不确定性：让网络同时输出均值和方差", en: "Predicting uncertainty: the network outputs a mean and a variance" },
  homo: { zh: "方差恒定", en: "Constant variance" },
  het: { zh: "方差随 x 变化", en: "Variance depends on x" },
  nll: { zh: "平均负对数似然", en: "Average NLL" },
  low: { zh: "±2σ 带的覆盖率（x < 5）", en: "±2σ coverage (x < 5)" },
  high: { zh: "±2σ 带的覆盖率（x ≥ 5）", en: "±2σ coverage (x ≥ 5)" },
  hours: { zh: "每周学习小时数", en: "Study hours per week" },
  score: { zh: "成绩", en: "Score" },
  caption: {
    zh: "这批学生学得越多，成绩分得越开：有人突飞猛进，有人原地踏步。平方损失等于假设噪声方差处处相同——拟合出的 ±2σ 带（虚线）宽度不变：学得少的那半边太宽（覆盖 100%），学得多的那半边太窄（只覆盖约 91%）。让网络多输出一个 log σ(x)，损失换成完整的高斯负对数似然 (y − μ)²/(2σ²) + log σ，带子就会跟着噪声变宽，两半的覆盖率都接近 95%，平均负对数似然也更低。",
    en: "In this group, the more the students study, the more their scores spread: some leap ahead, some stand still. The squared loss assumes the noise variance is the same everywhere, so the fitted ±2σ band (dashed) has a constant width: too wide where hours are low (covering 100%) and too narrow where they are high (covering only about 91%). Let the network output a second number, log σ(x), and use the full Gaussian negative log-likelihood (y − μ)²/(2σ²) + log σ: the band now widens with the noise, both halves are covered close to 95%, and the average NLL drops.",
  },
};

const pct = (v: number) => `${(100 * v).toFixed(0)}%`;

export default function HeteroBand({ lang }: { lang: Lang }) {
  const [het, setHet] = useState(false);
  const data = useMemo(() => heteroData(), []);
  const fits = useMemo(() => ({ homo: fitHomo(data.xs, data.ys), het: fitHetero(data.xs, data.ys) }), [data]);
  const m = het ? fits.het : fits.homo;
  const cov = coverage(m, data.xs, data.ys);
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="chips">
        <button className={`chip${!het ? " on" : ""}`} onClick={() => setHet(false)}>{TEXT.homo[lang]}</button>
        <button className={`chip${het ? " on" : ""}`} onClick={() => setHet(true)}>{TEXT.het[lang]}</button>
      </div>
      <CurvePlot xs={data.xs} ys={data.ys} xDomain={[0, 10]} yDomain={[20, 110]} xLabel={TEXT.hours[lang]} yLabel={TEXT.score[lang]}
        curves={[
          { f: (x) => m.mu(x) + 2 * m.sd(x), color: "var(--accent-2)", dashed: true, width: 2 },
          { f: (x) => m.mu(x) - 2 * m.sd(x), color: "var(--accent-2)", dashed: true, width: 2 },
          { f: m.mu, color: "var(--class-pos)", width: 3 },
        ]} vlines={[{ x: 5, color: "var(--border)" }]} />
      <div className="w-stats">
        <Stat label={TEXT.nll[lang]} value={avgNll(m, data.xs, data.ys).toFixed(3)} tone="accent" />
        <Stat label={TEXT.low[lang]} value={pct(cov.low)} />
        <Stat label={TEXT.high[lang]} value={pct(cov.high)} />
      </div>
    </Widget>
  );
}
