import { scaleLinear } from "d3-scale";
import { useEffect, useMemo, useState } from "react";
import { gaussian, r6, rng } from "../../../engines/data";
import { gradientBoost1D, stumpValue } from "../../../engines/tree";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";

const TEXT = {
  title: { zh: "梯度提升：一轮一轮拟合残差", en: "Gradient boosting: fit the residuals, round after round" },
  rounds: { zh: "轮数 m", en: "Round m" },
  mse: { zh: "训练均方误差", en: "Training MSE" },
  play: { zh: "▶ 播放", en: "▶ Play" },
  top: { zh: "当前模型 Fₘ(x)", en: "Current model Fₘ(x)" },
  bottom: { zh: "残差 y − Fₘ₋₁(x) 与本轮拟合的树桩", en: "Residuals y − Fₘ₋₁(x) and this round's stump" },
  caption: {
    zh: "上图：橙色阶梯是当前模型。下图：竖线是上一轮留下的残差，紫色阶梯是本轮去拟合残差的树桩；它乘以学习率 ν 加到模型上。平方损失的负梯度正是残差，所以这就是函数空间里的梯度下降。",
    en: "Top: the orange staircase is the current model. Bottom: sticks are the residuals left by the previous round; the purple step is the stump fitted to them, which is added to the model after scaling by the learning rate ν. For squared loss the negative gradient is exactly the residual, so this is gradient descent in function space.",
  },
};

export default function ResidualBoost({ lang }: { lang: Lang }) {
  const { xs, ys } = useMemo(() => {
    const rand = rng(3);
    const xs = Array.from({ length: 50 }, (_, i) => (i / 49) * 6);
    return { xs, ys: xs.map((x) => r6(Math.sin(x) + 0.3 * Math.sin(3 * x) + 0.12 * gaussian(rand))) };
  }, []);
  const ROUNDS = 40;
  const gb = useMemo(() => gradientBoost1D(xs, ys, ROUNDS, 0.3), [xs, ys]);
  const [m, setM] = useState(1);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing) return;
    if (m >= ROUNDS) { setPlaying(false); return; }
    const id = setTimeout(() => setM(m + 1), 380);
    return () => clearTimeout(id);
  }, [playing, m]);

  const W = 460;
  const H = 170;
  const sx = scaleLinear().domain([0, 6]).range([14, W - 14]);
  const syTop = scaleLinear().domain([-1.8, 1.8]).range([H - 10, 10]);
  const stage = gb.stages[m];
  const grid = Array.from({ length: 241 }, (_, i) => (i / 240) * 6);
  const path = (f: (x: number) => number, sy: (v: number) => number) =>
    "M" + grid.map((x) => `${sx(x).toFixed(1)},${sy(f(x)).toFixed(1)}`).join("L");
  const resMax = Math.max(0.3, ...stage.residual.map(Math.abs));
  const syBot = scaleLinear().domain([-resMax, resMax]).range([H - 10, 10]);

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="replay">
      <figure className="gb-fig">
        <figcaption>{TEXT.top[lang]}</figcaption>
        <svg viewBox={`0 0 ${W} ${H}`} role="img">
          <rect width={W} height={H} className="plot-bg" rx={10} />
          {xs.map((x, i) => <circle key={i} cx={sx(x)} cy={syTop(ys[i])} r={3.2} className="gb-pt" />)}
          <path d={path((x) => gb.predict(x, m), syTop)} className="gb-model" />
        </svg>
        <figcaption>{TEXT.bottom[lang]}</figcaption>
        <svg viewBox={`0 0 ${W} ${H}`} role="img">
          <rect width={W} height={H} className="plot-bg" rx={10} />
          <line x1={0} x2={W} y1={syBot(0)} y2={syBot(0)} className="gb-axis" />
          {xs.map((x, i) => (
            <line key={i} x1={sx(x)} x2={sx(x)} y1={syBot(0)} y2={syBot(stage.residual[i])} className="gb-res" />
          ))}
          {stage.stump && <path d={path((x) => stumpValue(stage.stump!, x), syBot)} className="gb-stump" />}
        </svg>
      </figure>
      <div className="w-row">
        <button className="btn btn-primary" onClick={() => { if (m >= ROUNDS) setM(1); setPlaying(!playing); }}>{playing ? "❚❚" : TEXT.play[lang]}</button>
        <Slider label={TEXT.rounds[lang]} value={m} min={1} max={ROUNDS} step={1} onChange={(v) => { setPlaying(false); setM(v); }} />
      </div>
      <div className="w-stats">
        <Stat label={TEXT.mse[lang]} value={stage.mse.toFixed(4)} tone="accent" />
      </div>
    </Widget>
  );
}
