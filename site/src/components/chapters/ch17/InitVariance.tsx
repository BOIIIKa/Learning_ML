import { useMemo, useState } from "react";
import { reluTheory, varianceProfile, type Act } from "../../../engines/backprop-init";
import type { Lang } from "../../../i18n/ui";
import MetricChart from "../../viz/MetricChart";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch17.css";

const TEXT = {
  title: { zh: "初始化与各层的方差", en: "Initialisation and per-layer variance" },
  gain: { zh: "初始化尺度 g（Var w = g/n）", en: "Init scale g (Var w = g/n)" },
  depth: { zh: "层数", en: "Depth" },
  fwd: { zh: "前向：log₁₀ 激活 RMS", en: "Forward: log₁₀ activation RMS" },
  bwd: { zh: "反向：log₁₀ 梯度 RMS", en: "Backward: log₁₀ gradient RMS" },
  theory: { zh: "理论（ReLU）", en: "Theory (ReLU)" },
  last: { zh: "最后一层的激活", en: "Last layer's activation" },
  first: { zh: "第一层收到的梯度", en: "Gradient reaching layer 1" },
  he: { zh: "He：g = 2", en: "He: g = 2" },
  xavier: { zh: "Xavier / LeCun：g = 1", en: "Xavier / LeCun: g = 1" },
  caption: {
    zh: "宽 96 的全连接网络，权重方差 g/n，喂入标准正态的输入，再从顶端送回随机梯度，画出每层激活和梯度的均方根（对数刻度，横轴是层号）。ReLU 每层把二阶矩乘以 g/2：g = 2（He 初始化）时前向和反向都大致不变；g = 1 时 30 层后缩到 10⁻⁵ 量级，g = 4 时涨到 10⁴ 量级——梯度在反方向上做同样的事。tanh 在零点附近斜率为 1，适合 g = 1（Xavier / LeCun）；g 太小同样会消失。",
    en: "A fully connected network of width 96 with weight variance g/n is fed standard normal inputs, and random gradients are sent back from the top; the plots show each layer's RMS activation and gradient (log scale, layer index across). ReLU multiplies the second moment by g/2 per layer: with g = 2 (He initialisation) both directions stay roughly level; with g = 1 they shrink to the order of 10⁻⁵ after 30 layers, with g = 4 they grow to the order of 10⁴, and the gradients do the same in the other direction. tanh has slope 1 at zero and suits g = 1 (Xavier / LeCun); too small a g makes it vanish as well.",
  },
};

export default function InitVariance({ lang }: { lang: Lang }) {
  const [a, setA] = useState<Act>("relu");
  const [gain, setGain] = useState(2);
  const [depth, setDepth] = useState(30);
  const p = useMemo(() => varianceProfile(a, gain, depth), [a, gain, depth]);
  const lg = (v: number) => Math.log10(Math.max(v, 1e-30));
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="chips">
        <button className={`chip${a === "relu" ? " on" : ""}`} onClick={() => setA("relu")}>ReLU</button>
        <button className={`chip${a === "tanh" ? " on" : ""}`} onClick={() => setA("tanh")}>tanh</button>
        <button className="chip" onClick={() => setGain(2)}>{TEXT.he[lang]}</button>
        <button className="chip" onClick={() => setGain(1)}>{TEXT.xavier[lang]}</button>
      </div>
      <div className="bp-two">
        <MetricChart zero={false} span={[-1, 1]} cursor={depth - 1} height={160}
          series={[
            { key: "f", label: TEXT.fwd[lang], values: p.forward.map(lg), color: "var(--accent)" },
            ...(a === "relu" ? [{ key: "t", label: TEXT.theory[lang], values: p.forward.map((_, l) => lg(reluTheory(gain, l))), color: "var(--text-muted)", dashed: true }] : []),
          ]} />
        <MetricChart zero={false} span={[-1, 1]} cursor={0} height={160}
          series={[{ key: "b", label: TEXT.bwd[lang], values: p.backward.map(lg), color: "var(--accent-2)" }]} />
      </div>
      <Slider label={TEXT.gain[lang]} value={gain} min={0.5} max={4} step={0.1} onChange={setGain} format={(v) => v.toFixed(1)} />
      <Slider label={TEXT.depth[lang]} value={depth} min={5} max={40} step={1} onChange={setDepth} />
      <div className="w-stats">
        <Stat label={TEXT.last[lang]} value={p.forward[depth - 1].toExponential(1)} tone="accent" />
        <Stat label={TEXT.first[lang]} value={p.backward[0].toExponential(1)} />
      </div>
    </Widget>
  );
}
