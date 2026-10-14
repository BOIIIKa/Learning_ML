import { useState } from "react";
import { LOSS_IDS, loss, pull, type LossId } from "../../../engines/linear-classification";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch04.css";

const TEXT = {
  title: { zh: "分类损失动物园", en: "A zoo of classification losses" },
  value: { zh: "损失 ℓ(m)", en: "Loss ℓ(m)" },
  pull: { zh: "拉力 −ℓ′(m)", en: "Pull −ℓ′(m)" },
  m: { zh: "间隔 m = y·f(x)", en: "Margin m = y·f(x)" },
  caption: {
    zh: "横轴是间隔 m = y·f(x)：m > 0 分对，m 越大越有把握。0-1 损失是我们真正关心的，但它是阶梯，梯度处处为 0（或不存在），没法直接优化。其余四条都是它的凸替身：感知机损失（本章）、交叉熵（本章，以 2 为底，过点 (0, 1)）、合页损失（第 07 章 SVM）、指数损失（第 08 章 AdaBoost）。切到“拉力”看 −ℓ′(m)，它就是这个点在梯度里的权重：合页损失对 m ≥ 1 的点完全放手（所以只有支持向量起作用），交叉熵对分对的点也一直轻轻拉着，指数损失对分错的点拉力指数爆炸（所以 AdaBoost 怕离群点）。",
    en: "The horizontal axis is the margin m = y·f(x): m > 0 is correct, and larger m is more confident. The 0-1 loss is what we really care about, but it is a staircase with zero (or no) gradient, so it cannot be optimized directly. The other four are convex stand-ins: the perceptron loss (this chapter), cross-entropy (this chapter, base 2 so it passes through (0, 1)), the hinge loss (Chapter 07, SVM) and the exponential loss (Chapter 08, AdaBoost). Switch to “pull” to see −ℓ′(m), the weight a point gets in the gradient: the hinge loss lets go completely once m ≥ 1 (so only support vectors matter), cross-entropy keeps pulling gently even on correct points, and the exponential loss pulls exponentially hard on mistakes (so AdaBoost fears outliers).",
  },
};

const NAMES: Record<LossId, { zh: string; en: string }> = {
  zeroOne: { zh: "0-1", en: "0-1" },
  perceptron: { zh: "感知机", en: "Perceptron" },
  logistic: { zh: "交叉熵", en: "Cross-entropy" },
  hinge: { zh: "合页", en: "Hinge" },
  exp: { zh: "指数", en: "Exponential" },
};
const COLORS: Record<LossId, string> = {
  zeroOne: "var(--text-muted)",
  perceptron: "var(--success)",
  logistic: "var(--accent)",
  hinge: "var(--accent-2)",
  exp: "var(--danger)",
};

export default function LossZoo({ lang }: { lang: Lang }) {
  const [mode, setMode] = useState(0);
  const [m, setM] = useState(-0.5);
  const fn = mode === 0 ? loss : pull;
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="frontier">
      <div className="chips">
        {[TEXT.value, TEXT.pull].map((t, i) => (
          <button key={i} className={`chip${mode === i ? " on" : ""}`} onClick={() => setMode(i)}>{t[lang]}</button>
        ))}
        <span className="lz-legend">
          {LOSS_IDS.map((id) => <span key={id}><i style={{ background: COLORS[id] }} />{NAMES[id][lang]}</span>)}
        </span>
      </div>
      <CurvePlot xs={[]} ys={[]} xDomain={[-2.5, 3]} yDomain={[-0.2, 4]} height={300} xLabel={TEXT.m[lang]}
        curves={LOSS_IDS.map((id) => ({ f: fn[id], color: COLORS[id], width: id === "zeroOne" ? 2 : 2.6, dashed: id === "zeroOne" }))}
        vlines={[{ x: m, color: "var(--ink)" }]} />
      <Slider label={TEXT.m[lang]} value={m} min={-2.5} max={3} step={0.05} onChange={setM} format={(v) => v.toFixed(2)} />
      <div className="w-stats lz-stats">
        {LOSS_IDS.map((id) => <Stat key={id} label={NAMES[id][lang]} value={fn[id](m).toFixed(2)} />)}
      </div>
    </Widget>
  );
}
