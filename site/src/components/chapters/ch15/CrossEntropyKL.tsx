import { useState } from "react";
import { crossEntropy, entropy, kl } from "../../../engines/losses-as-likelihood";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch15.css";

const TEXT = {
  title: { zh: "交叉熵 = 熵 + KL 散度", en: "Cross-entropy = entropy + KL divergence" },
  p: { zh: "真实的及格率 p", en: "True pass rate p" },
  q: { zh: "模型预测的及格率 q", en: "Model's pass rate q" },
  H: { zh: "熵 H(p)（降不下去的部分）", en: "Entropy H(p) (the floor)" },
  KL: { zh: "KL(p‖q)（模型的代价）", en: "KL(p‖q) (the model's cost)" },
  CE: { zh: "交叉熵 H(p, q)", en: "Cross-entropy H(p, q)" },
  caption: {
    zh: "平均交叉熵损失（以 nats 计）分成两部分：数据本身的随机性 H(p)——任何模型都消除不了；以及模型分布 q 偏离真实分布 p 的代价 KL(p‖q) ≥ 0。曲线是交叉熵随 q 的变化，水平线是 H(p)，两者之间的竖直距离就是 KL。最小化交叉熵（也就是最大化似然）只能压缩 KL，当且仅当 q = p 时 KL = 0，交叉熵降到熵。",
    en: "The average cross-entropy loss (in nats) splits into two parts: the randomness of the data itself, H(p), which no model can remove, and the cost KL(p‖q) ≥ 0 of the model's distribution q differing from the true p. The curve is the cross-entropy as q varies, the horizontal line is H(p), and the vertical gap between them is the KL. Minimising cross-entropy (that is, maximising likelihood) can only shrink the KL, which is zero exactly when q = p, where the cross-entropy drops to the entropy.",
  },
};

export default function CrossEntropyKL({ lang }: { lang: Lang }) {
  const [p, setP] = useState(0.7);
  const [q, setQ] = useState(0.5);
  const H = entropy(p), K = kl(p, q), C = crossEntropy(p, q);
  const W = 420, scale = W / 2.4;
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <CurvePlot xs={[]} ys={[]} xDomain={[0.01, 0.99]} yDomain={[0, 2.4]} height={220} xLabel="q" yLabel="nats"
        curves={[{ f: (x) => crossEntropy(p, x), color: "var(--class-pos)", width: 3 }, { f: () => H, color: "var(--accent-2)", dashed: true, width: 2 }]}
        vlines={[{ x: q, color: "var(--accent)" }, { x: p, color: "var(--border)" }]} />
      <svg className="lm-svg ck-bar" viewBox={`0 0 ${W} 34`} role="img">
        <rect x={0} y={6} width={(H * scale).toFixed(1)} height={22} className="ck-h" />
        <rect x={(H * scale).toFixed(1)} y={6} width={Math.max(0, K * scale).toFixed(1)} height={22} className="ck-kl" />
      </svg>
      <Slider label={TEXT.p[lang]} value={p} min={0.05} max={0.95} step={0.01} onChange={setP} format={(v) => v.toFixed(2)} />
      <Slider label={TEXT.q[lang]} value={q} min={0.02} max={0.98} step={0.01} onChange={setQ} format={(v) => v.toFixed(2)} />
      <div className="w-stats">
        <Stat label={TEXT.H[lang]} value={H.toFixed(3)} />
        <Stat label={TEXT.KL[lang]} value={K.toFixed(3)} tone="accent" />
        <Stat label={TEXT.CE[lang]} value={C.toFixed(3)} />
      </div>
    </Widget>
  );
}
