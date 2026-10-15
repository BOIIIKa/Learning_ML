import { useMemo, useState } from "react";
import { STALL_EPOCHS, trainDeep } from "../../../engines/backprop-init";
import type { Lang } from "../../../i18n/ui";
import MetricChart from "../../viz/MetricChart";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch17.css";

const TEXT = {
  title: { zh: "网络一加深，就学不动了", en: "Make the network deeper and it stops learning" },
  depth: { zh: "隐藏层数", en: "Hidden layers" },
  small: { zh: "小随机数 N(0, 0.1²)", en: "Small random N(0, 0.1²)" },
  he: { zh: "He 初始化 N(0, 2/n)", en: "He init N(0, 2/n)" },
  loss: { zh: "训练损失", en: "Training loss" },
  grads: { zh: "初始时各层权重梯度的大小（对数刻度）", en: "Gradient size per layer at initialisation (log scale)" },
  layer: { zh: "层", en: "layer" },
  finalS: { zh: "小随机数：最终损失", en: "Small: final loss" },
  finalH: { zh: "He：最终损失", en: "He: final loss" },
  g1: { zh: "第 1 层梯度（小随机数 / He）", en: "Layer-1 gradient (small / He)" },
  caption: {
    zh: "老师把双月分类的网络加深：每个隐藏层 16 个 ReLU 单元，小批量 SGD 训练 24 轮。权重初始化成 N(0, 0.1²) 的小随机数时，1–3 层还能学；从 5 层起，损失一直停在 ln 2 ≈ 0.693，等于瞎猜。下面的柱子说明了原因：权重太小，每过一层信号就缩小约 3.5 倍（0.1 × √(16/2) ≈ 0.28），梯度也一样；8 层时第 1 层的梯度只有 10⁻⁶ 量级，而 He 初始化下是 3 × 10⁻²。改用 He 初始化（方差 2/n），每层的梯度都在同一个量级，任何深度都能训练。",
    en: "The teacher deepens the two-moons classifier: 16 ReLU units per hidden layer, 24 epochs of minibatch SGD. With weights initialised as small random numbers N(0, 0.1²), 1–3 layers still learn; from 5 layers on the loss sits at ln 2 ≈ 0.693, no better than guessing. The bars show why: with weights this small, every layer shrinks the signal about 3.5× (0.1 × √(16/2) ≈ 0.28), and the gradient with it; at 8 layers the first layer's gradient is of order 10⁻⁶, against 3 × 10⁻² under He initialisation. With He initialisation (variance 2/n), every layer's gradient has the same order of magnitude, and every depth trains.",
  },
};

const W = 460, H = 200;
const f1 = (v: number) => v.toFixed(1);
const lg = (v: number) => Math.log10(Math.max(v, 1e-12));

export default function DeepStall({ lang }: { lang: Lang }) {
  const [depth, setDepth] = useState(2);
  const small = useMemo(() => trainDeep(depth, "small"), [depth]);
  const he = useMemo(() => trainDeep(depth, "he"), [depth]);
  const L = small.gradRms.length;
  const bw = (W - 40) / L;
  const y = (v: number) => H - 18 - ((lg(v) + 9) / 9) * (H - 28); // 1e-9 … 1
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="scenario">
      <Slider label={TEXT.depth[lang]} value={depth} min={1} max={10} step={1} onChange={setDepth} />
      <div className="bp-two">
      <div>
      <div className="bp-sub">{TEXT.loss[lang]}</div>
      <MetricChart cursor={STALL_EPOCHS} height={150}
        series={[
          { key: "s", label: TEXT.small[lang], values: small.losses, color: "var(--danger)" },
          { key: "h", label: TEXT.he[lang], values: he.losses, color: "var(--success)" },
        ]} />
      </div>
      <div>
      <div className="bp-sub">{TEXT.grads[lang]}</div>
      <svg className="lm-svg" viewBox={`0 0 ${W} ${H}`} role="img">
        {[0, -3, -6, -9].map((e) => (
          <g key={e}>
            <line x1={34} x2={W - 4} y1={f1(y(10 ** e))} y2={f1(y(10 ** e))} className="bp-grid" />
            <text x={30} y={f1(y(10 ** e))} textAnchor="end" dominantBaseline="middle" className="bp-tick">{`1e${e}`}</text>
          </g>
        ))}
        {small.gradRms.map((v, l) => (
          <g key={l}>
            <rect x={f1(38 + l * bw)} y={f1(y(v))} width={f1(bw * 0.4)} height={f1(Math.max(0, H - 18 - y(v)))} className="bp-bar small" />
            <rect x={f1(38 + l * bw + bw * 0.42)} y={f1(y(he.gradRms[l]))} width={f1(bw * 0.4)} height={f1(Math.max(0, H - 18 - y(he.gradRms[l])))} className="bp-bar he" />
            <text x={f1(38 + l * bw + bw * 0.41)} y={H - 4} textAnchor="middle" className="bp-tick">{l + 1}</text>
          </g>
        ))}
      </svg>
      </div>
      </div>
      <div className="w-stats">
        <Stat label={TEXT.finalS[lang]} value={small.losses.at(-1)!.toFixed(3)} tone={small.losses.at(-1)! > 0.69 ? "bad" : undefined} />
        <Stat label={TEXT.finalH[lang]} value={he.losses.at(-1)!.toFixed(3)} tone="good" />
        <Stat label={TEXT.g1[lang]} value={`${small.gradRms[0].toExponential(0)} / ${he.gradRms[0].toExponential(0)}`} tone="accent" />
      </div>
    </Widget>
  );
}
