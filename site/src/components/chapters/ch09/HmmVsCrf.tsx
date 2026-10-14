import { useState } from "react";
import type { Lang } from "../../../i18n/ui";
import Widget from "../../ui/Widget";
import "./ch09.css";

const TEXT = {
  title: { zh: "HMM 与 CRF：同一条链，两种建模", en: "HMM vs CRF: one chain, two ways to model it" },
  hmm: { zh: "HMM（生成式）", en: "HMM (generative)" },
  crf: { zh: "线性链 CRF（判别式）", en: "Linear-chain CRF (discriminative)" },
  caption: {
    zh: "上排是隐藏状态 y，下排是观测 x。HMM 是有向图：先按转移概率生成状态链，每个状态再生成自己的观测，建模的是联合分布 P(x, y)，每个因子都是归一化的条件概率。线性链 CRF 是无向图：只建模 P(y | x)，因子是任意实数打分 exp(w·f)，可以看整个观测序列（虚线），最后用全局的规范化因子 Z(x) 归一化。这正是第 04、05 章“逻辑回归对朴素贝叶斯”在序列上的翻版。",
    en: "The top row are the hidden states y, the bottom row the observations x. The HMM is a directed graph: it generates the state chain with transition probabilities and lets each state emit its observation, modelling the joint P(x, y); every factor is a normalized conditional probability. The linear-chain CRF is undirected: it models only P(y | x), its factors are arbitrary real-valued scores exp(w·f) that may look at the whole observation sequence (dashed), and one global normalizer Z(x) makes it a distribution. It is Chapters 04–05's \"logistic regression vs naive Bayes\" again, on sequences.",
  },
  fHmm: "P(x, y) = P(y₁) ∏ P(yₜ | yₜ₋₁) ∏ P(xₜ | yₜ)",
  fCrf: "P(y | x) = exp(Σₜ w·f(yₜ₋₁, yₜ, x, t)) / Z(x)",
};

const f1 = (v: number) => v.toFixed(1);

export default function HmmVsCrf({ lang }: { lang: Lang }) {
  const [crf, setCrf] = useState(false);
  const W = 460, H = 200, n = 4;
  const x = (k: number) => 70 + k * 105;
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="chips">
        <button className={`chip${!crf ? " on" : ""}`} onClick={() => setCrf(false)}>{TEXT.hmm[lang]}</button>
        <button className={`chip${crf ? " on" : ""}`} onClick={() => setCrf(true)}>{TEXT.crf[lang]}</button>
      </div>
      <svg className="lm-svg hc-svg" viewBox={`0 0 ${W} ${H}`} role="img">
        <defs>
          <marker id="hc-arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" className="hc-head" /></marker>
        </defs>
        {Array.from({ length: n - 1 }, (_, k) => (
          <line key={`t${k}`} x1={f1(x(k) + 18)} y1={60} x2={f1(x(k + 1) - 18)} y2={60} className="hc-edge" markerEnd={crf ? undefined : "url(#hc-arrow)"} />
        ))}
        {Array.from({ length: n }, (_, k) => (
          crf
            ? Array.from({ length: n }, (_, j) => <line key={`c${k}${j}`} x1={f1(x(k))} y1={78} x2={f1(x(j))} y2={142} className={`hc-edge${j === k ? "" : " global"}`} />)
            : <line key={`e${k}`} x1={f1(x(k))} y1={78} x2={f1(x(k))} y2={140} className="hc-edge" markerEnd="url(#hc-arrow)" />
        ))}
        {Array.from({ length: n }, (_, k) => (
          <g key={k}>
            <circle cx={f1(x(k))} cy={60} r={18} className="hc-y" />
            <text x={f1(x(k))} y={65} textAnchor="middle" className="hc-lab">y{k + 1}</text>
            <rect x={f1(x(k) - 18)} y={142} width={36} height={32} className="hc-x" />
            <text x={f1(x(k))} y={163} textAnchor="middle" className="hc-lab">x{k + 1}</text>
          </g>
        ))}
      </svg>
      <p className="hc-formula">{crf ? TEXT.fCrf : TEXT.fHmm}</p>
    </Widget>
  );
}
