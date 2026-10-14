import { useMemo, useState } from "react";
import { entropy, maxEntDie, sameMeanAlternative } from "../../../engines/linclass";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import { f1 } from "./geom";
import "./ch04.css";

const TEXT = {
  title: { zh: "最大熵：只知道平均点数的骰子", en: "Maximum entropy: a die whose mean is all we know" },
  mean: { zh: "已知的平均点数 μ", en: "Known mean μ" },
  other: { zh: "换一个同均值的分布", en: "Another distribution, same mean" },
  hmax: { zh: "最大熵分布的熵", en: "Entropy of the max-ent distribution" },
  halt: { zh: "对比分布的熵", en: "Entropy of the alternative" },
  lambda: { zh: "乘子 λ", en: "Multiplier λ" },
  uniform: { zh: "上限 log 6", en: "Ceiling log 6" },
  legMax: { zh: "最大熵分布 pₖ ∝ exp(λk)", en: "Max-ent distribution pₖ ∝ exp(λk)" },
  legAlt: { zh: "另一个满足同样约束的分布", en: "Another distribution meeting the same constraint" },
  caption: {
    zh: "我们只知道一枚骰子掷出的平均点数是 μ。满足这个约束的分布有无穷多个（虚线柱是随手挑的一个），最大熵原则选“除了约束之外不做任何额外假设”的那个——实心柱。拖动 μ：最大熵分布永远是 pₖ ∝ exp(λk) 的指数形式，μ = 3.5 时 λ = 0，退化为均匀分布。不管换多少个对比分布，它们的熵都更低。把“点数”换成特征 f(x, y)，同样的推导就给出 softmax，也就是逻辑回归。",
    en: "All we know about a die is that its average roll is μ. Infinitely many distributions satisfy that (the dashed bars are one picked at random); the maximum-entropy principle picks the one that assumes nothing beyond the constraint: the solid bars. Drag μ: the max-ent answer is always exponential, pₖ ∝ exp(λk), and at μ = 3.5 we get λ = 0, the uniform die. However many alternatives you draw, their entropy is lower. Replace “the number of pips” with features f(x, y) and the same derivation gives softmax, i.e. logistic regression.",
  },
};

const W = 460;
const H = 250;
const PAD = { l: 20, r: 20, t: 22, b: 30 };

export default function MaxEntDice({ lang }: { lang: Lang }) {
  const [mu, setMu] = useState(4.5);
  const [seed, setSeed] = useState(1);
  const me = useMemo(() => maxEntDie(mu), [mu]);
  const alt = useMemo(() => sameMeanAlternative(me.p, seed), [me, seed]);

  const top = Math.max(0.5, ...me.p, ...alt) * 1.08;
  const slot = (W - PAD.l - PAD.r) / 6;
  const bw = slot * 0.36;
  const yOf = (p: number) => PAD.t + (H - PAD.t - PAD.b) * (1 - p / top);
  const meanX = PAD.l + slot * (mu - 0.5);

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <svg className="lc-svg" viewBox={`0 0 ${W} ${H}`} role="img">
        <rect width={W} height={H} rx={12} className="plot-bg" />
        <line x1={PAD.l} x2={W - PAD.r} y1={yOf(0)} y2={yOf(0)} className="lc-axis" />
        {me.p.map((p, k) => {
          const cx = PAD.l + slot * (k + 0.5);
          return (
            <g key={k}>
              <rect x={f1(cx - bw - 2)} y={f1(yOf(p))} width={f1(bw)} height={f1(yOf(0) - yOf(p))} rx={3} className="lc-bar" />
              <rect x={f1(cx + 2)} y={f1(yOf(alt[k]))} width={f1(bw)} height={f1(yOf(0) - yOf(alt[k]))} rx={3} className="lc-bar-alt" />
              <text x={f1(cx - bw / 2 - 2)} y={f1(yOf(p) - 5)} textAnchor="middle" className="lc-bar-lab">{p.toFixed(2)}</text>
              <text x={f1(cx + bw / 2 + 2)} y={f1(yOf(alt[k]) - 5)} textAnchor="middle" className="lc-bar-lab alt">{alt[k].toFixed(2)}</text>
              <text x={f1(cx)} y={H - 10} textAnchor="middle" className="lc-face">{k + 1}</text>
            </g>
          );
        })}
        <line x1={f1(meanX)} x2={f1(meanX)} y1={PAD.t - 8} y2={yOf(0)} className="lc-mean" />
        <text x={f1(Math.min(W - 60, meanX + 5))} y={PAD.t - 2} className="lc-mean-lab">μ = {mu.toFixed(2)}</text>
      </svg>
      <div className="lc-legend">
        <span><i className="max" />{TEXT.legMax[lang]}</span>
        <span><i className="alt" />{TEXT.legAlt[lang]}</span>
      </div>
      <div className="w-row">
        <Slider label={TEXT.mean[lang]} value={mu} min={1.5} max={5.5} step={0.05} onChange={setMu} format={(v) => v.toFixed(2)} />
        <button className="btn" onClick={() => setSeed(seed + 1)}>{TEXT.other[lang]}</button>
      </div>
      <div className="w-stats">
        <Stat label={TEXT.hmax[lang]} value={me.entropy.toFixed(4)} tone="good" />
        <Stat label={TEXT.halt[lang]} value={entropy(alt).toFixed(4)} />
        <Stat label={TEXT.lambda[lang]} value={me.lambda.toFixed(3)} tone="accent" />
        <Stat label={TEXT.uniform[lang]} value={Math.log(6).toFixed(4)} />
      </div>
    </Widget>
  );
}
