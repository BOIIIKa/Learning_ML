import { useState } from "react";
import { entropy, maxentDie, twoFaceDie } from "../../../engines/linear-classification";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch04.css";

const TEXT = {
  title: { zh: "最大熵：只知道均值时，最不武断的猜测", en: "Maximum entropy: the least presumptuous guess given a mean" },
  mu: { zh: "已知的平均点数 μ", en: "Known mean μ" },
  maxent: { zh: "最大熵分布", en: "Max-entropy" },
  other: { zh: "另一个同均值的分布", en: "Another with the same mean" },
  H: { zh: "熵 H（奈特）", en: "Entropy H (nats)" },
  Hmax: { zh: "上限 log 6", en: "Ceiling log 6" },
  lam: { zh: "乘子 λ", en: "Multiplier λ" },
  caption: {
    zh: "一颗骰子，只知道平均点数 μ。满足这个约束的分布有无穷多个；最大熵原则挑熵最大的那个，也就是除了 μ 之外不额外假设任何东西的那个。解总是 p_k ∝ exp(λk)——点数的 softmax，λ 是约束的拉格朗日乘子。μ = 3.5 时 λ = 0，正好是均匀分布。切到“另一个分布”：均值一样，熵却低得多。",
    en: "A die whose only known property is its mean μ. Infinitely many distributions satisfy that constraint; the maximum-entropy principle picks the one with the largest entropy, the one that assumes nothing beyond μ. The answer is always p_k ∝ exp(λk), a softmax of the face values, where λ is the Lagrange multiplier of the constraint. At μ = 3.5, λ = 0 and the die is uniform. Switch to “another distribution”: same mean, much lower entropy.",
  },
};

const W = 460;
const H = 230;
const f1 = (v: number) => v.toFixed(1);

export default function MaxEntDice({ lang }: { lang: Lang }) {
  const [mu, setMu] = useState(4.5);
  const [mode, setMode] = useState(0);
  const me = maxentDie(mu);
  const p = mode === 0 ? me.p : twoFaceDie(mu);
  const Hp = entropy(p);
  const bw = (W - 60) / 6;
  const top = 0.75;
  const sy = (v: number) => H - 30 - (Math.min(v, top) / top) * (H - 50);
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="chips">
        {[TEXT.maxent, TEXT.other].map((t, i) => (
          <button key={i} className={`chip${mode === i ? " on" : ""}`} onClick={() => setMode(i)}>{t[lang]}</button>
        ))}
      </div>
      <svg className="lm-svg me-svg" viewBox={`0 0 ${W} ${H}`} role="img">
        {[0.25, 0.5].map((g) => (
          <g key={g}>
            <line x1={36} x2={W - 12} y1={f1(sy(g))} y2={f1(sy(g))} className="cp-grid" />
            <text x={30} y={f1(sy(g))} textAnchor="end" dominantBaseline="middle" className="cp-tick">{g}</text>
          </g>
        ))}
        <line x1={36} x2={W - 12} y1={f1(sy(1 / 6))} y2={f1(sy(1 / 6))} className="me-uniform" />
        {p.map((q, k) => {
          const x = 44 + k * bw;
          return (
            <g key={k}>
              <rect x={f1(x)} y={f1(sy(q))} width={f1(bw - 10)} height={f1(H - 30 - sy(q))} className={`me-bar${mode === 1 ? " alt" : ""}`} />
              <text x={f1(x + (bw - 10) / 2)} y={H - 12} textAnchor="middle" className="cp-tick">{k + 1}</text>
              <text x={f1(x + (bw - 10) / 2)} y={f1(sy(q) - 6)} textAnchor="middle" className="me-val">{q.toFixed(2)}</text>
            </g>
          );
        })}
        <line x1={f1(44 + (mu - 1) * bw + (bw - 10) / 2)} x2={f1(44 + (mu - 1) * bw + (bw - 10) / 2)} y1={14} y2={H - 30} className="me-mean" />
        <text x={f1(44 + (mu - 1) * bw + (bw - 10) / 2 + 5)} y={22} className="me-val">μ</text>
      </svg>
      <Slider label={TEXT.mu[lang]} value={mu} min={1.5} max={5.5} step={0.05} onChange={setMu} format={(v) => v.toFixed(2)} />
      <div className="w-stats">
        <Stat label={TEXT.H[lang]} value={Hp.toFixed(3)} tone={mode === 0 ? "accent" : "bad"} />
        <Stat label={TEXT.Hmax[lang]} value={Math.log(6).toFixed(3)} />
        <Stat label={TEXT.lam[lang]} value={me.lambda.toFixed(3)} />
      </div>
    </Widget>
  );
}
