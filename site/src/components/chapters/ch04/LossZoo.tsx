import { useState } from "react";
import { LOSSES, type LossKey } from "../../../engines/linclass";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch04.css";

const ITEMS: { key: LossKey; color: string; zh: string; en: string; where: { zh: string; en: string } }[] = [
  { key: "zeroOne", color: "var(--text)", zh: "0-1 损失", en: "0-1 loss", where: { zh: "真正想要的", en: "what we really want" } },
  { key: "perceptron", color: "var(--warning)", zh: "感知机损失 max(0, −m)", en: "Perceptron max(0, −m)", where: { zh: "本章", en: "this chapter" } },
  { key: "logistic", color: "var(--accent)", zh: "交叉熵 log₂(1+e⁻ᵐ)", en: "Cross-entropy log₂(1+e⁻ᵐ)", where: { zh: "本章 · 逻辑回归", en: "this chapter · logistic" } },
  { key: "hinge", color: "var(--success)", zh: "合页 max(0, 1−m)", en: "Hinge max(0, 1−m)", where: { zh: "第 07 章 · SVM", en: "Ch. 07 · SVM" } },
  { key: "exponential", color: "var(--accent-2)", zh: "指数 e⁻ᵐ", en: "Exponential e⁻ᵐ", where: { zh: "第 08 章 · AdaBoost", en: "Ch. 08 · AdaBoost" } },
];

const TEXT = {
  title: { zh: "分类损失动物园：一切都是间隔 m = y·f(x) 的函数", en: "The zoo of classification losses: all functions of the margin m = y·f(x)" },
  m: { zh: "间隔 m = y·f(x)", en: "Margin m = y·f(x)" },
  caption: {
    zh: "横轴是间隔 m = y·f(x)：m > 0 表示分对，m 越大越“有把握”。0-1 损失才是我们真正关心的错误率，但它不连续、梯度处处为 0，无法优化，于是各家方法换上不同的“替身”。交叉熵、合页、指数损失都经过 (0, 1) 并压在 0-1 损失之上，所以把它们降下来，错误率也被压下来。区别在于：分对之后还罚不罚（合页在 m ≥ 1 后归零，交叉熵永远有一点点），分错时罚多重（指数损失对离群点极其敏感）。感知机损失在 m = 0 处为零，所以它只要求分对，不要求分得有把握。",
    en: "The horizontal axis is the margin m = y·f(x): m > 0 means correct, and a larger m means more confident. The 0-1 loss is the error rate we really care about, but it is discontinuous with zero gradient almost everywhere, so methods optimize a stand-in instead. Cross-entropy, hinge and exponential losses all pass through (0, 1) and lie above the 0-1 loss, so pushing them down pushes the error rate down too. They differ in whether correct points still pay (hinge drops to zero at m ≥ 1, cross-entropy never quite does) and how hard mistakes are punished (the exponential loss is very sensitive to outliers). The perceptron loss is already zero at m = 0: it asks only for the right side, not for confidence.",
  },
};

export default function LossZoo({ lang }: { lang: Lang }) {
  const [on, setOn] = useState<Record<LossKey, boolean>>({ zeroOne: true, perceptron: true, logistic: true, hinge: true, exponential: true });
  const [m, setM] = useState(-0.5);
  const shown = ITEMS.filter((it) => on[it.key]);

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <CurvePlot xs={[]} ys={[]} xDomain={[-2.5, 3]} yDomain={[-0.2, 4]} height={300} xLabel="m = y·f(x)"
        curves={shown.map((it) => ({ f: LOSSES[it.key], color: it.color, width: it.key === "zeroOne" ? 3 : 2.4 }))}
        vlines={[{ x: m, color: "var(--text-muted)" }, { x: 0, color: "var(--border)" }]} />
      <div className="chips">
        {ITEMS.map((it) => (
          <button key={it.key} className={`chip${on[it.key] ? " on" : ""}`} onClick={() => setOn({ ...on, [it.key]: !on[it.key] })}>
            <i style={{ background: it.color }} />{it[lang]}
          </button>
        ))}
      </div>
      <div className="w-row">
        <Slider label={TEXT.m[lang]} value={m} min={-2.5} max={3} step={0.01} onChange={setM} format={(v) => v.toFixed(2)} />
      </div>
      <div className="w-stats">
        {ITEMS.map((it) => (
          <Stat key={it.key} label={`${it[lang].split(" ")[0]} · ${it.where[lang]}`} value={LOSSES[it.key](m).toFixed(3)} />
        ))}
      </div>
    </Widget>
  );
}
