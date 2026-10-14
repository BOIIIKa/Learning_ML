import { useState } from "react";
import { EXAMPLE_OBS, HMM, forward, viterbi } from "../../../engines/structured-prediction";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch09.css";

const TEXT = {
  title: { zh: "网格图上的动态规划", en: "Dynamic programming on the trellis" },
  fwd: { zh: "前向：求和", en: "Forward: sum" },
  vit: { zh: "维特比：取最大", en: "Viterbi: max" },
  short: { zh: "例题：3 天", en: "Example: 3 days" },
  long: { zh: "6 天", en: "6 days" },
  t: { zh: "点亮到第几天", en: "Light up to day" },
  states: { zh: ["稳定", "恶化"], en: ["stable", "worse"] },
  obs: { zh: ["正常", "异常"], en: ["normal", "abnormal"] },
  p: { zh: "P(观测序列)", en: "P(observations)" },
  best: { zh: "最优路径概率", en: "Best path probability" },
  path: { zh: "最可能的病情", en: "Most likely states" },
  caption: {
    zh: "住院病人每天的病情（稳定 / 恶化）看不见，只看得到当天化验正常还是异常。每一列是一天，每个圆是“这一天处于某状态”。前向算法逐列计算 α：到达该结点的所有路径的概率之和；维特比算法把求和换成取最大，并记下最大来自哪里（粗线是回溯指针）。点亮到最后一列，前向给出整个观测序列的概率，维特比沿回溯指针给出最可能的病情序列（黄色）。两者的代价都是 O(TN²)，而不是穷举 Nᵀ 条路径。",
    en: "A hospitalized patient's daily condition (stable / worse) is hidden; we only see whether the day's test is normal or abnormal. Each column is a day, each circle \"in this state on this day\". The forward algorithm fills α column by column: the summed probability of all paths reaching that node. Viterbi replaces the sum by a max and remembers where the max came from (bold edges are back-pointers). At the last column, forward gives the probability of the whole observation sequence and Viterbi, following the back-pointers, the most likely sequence of states (yellow). Both cost O(TN²) instead of enumerating Nᵀ paths.",
  },
};

const SEQS = [EXAMPLE_OBS, [0, 1, 1, 0, 0, 1]];
const f1 = (v: number) => v.toFixed(1);
const fmt = (v: number) => (v >= 0.001 ? v.toFixed(4) : v.toExponential(2));

export default function Trellis({ lang }: { lang: Lang }) {
  const [mode, setMode] = useState(0);
  const [si, setSi] = useState(0);
  const O = SEQS[si];
  const T = O.length;
  const [tRaw, setT] = useState(3);
  const t = Math.min(tRaw, T);
  const al = forward(HMM, O);
  const vit = viterbi(HMM, O);
  const V = mode === 0 ? al : vit.delta;
  const W = 460, H = 210;
  const cx = (k: number) => 50 + (k * (W - 100)) / Math.max(1, T - 1);
  const cy = (i: number) => 80 + i * 90;
  const done = t === T;
  const onPath = (k: number, i: number) => mode === 1 && done && vit.path[k] === i;
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="scenario">
      <div className="chips">
        {[TEXT.fwd, TEXT.vit].map((x, i) => <button key={i} className={`chip${mode === i ? " on" : ""}`} onClick={() => setMode(i)}>{x[lang]}</button>)}
        {[TEXT.short, TEXT.long].map((x, i) => <button key={`s${i}`} className={`chip${si === i ? " on" : ""}`} onClick={() => { setSi(i); setT(SEQS[i].length); }}>{x[lang]}</button>)}
      </div>
      <svg className="lm-svg tr-svg" viewBox={`0 0 ${W} ${H}`} role="img">
        {O.map((o, k) => <text key={`o${k}`} x={f1(cx(k))} y={22} textAnchor="middle" className={`tr-obs${o ? " ab" : ""}${k < t ? " on" : ""}`}>{TEXT.obs[lang][o]}</text>)}
        {[0, 1].map((i) => <text key={`s${i}`} x={6} y={f1(cy(i) + 4)} className="tr-state">{TEXT.states[lang][i]}</text>)}
        {O.slice(1).map((_, k1) => [0, 1].flatMap((i) => [0, 1].map((j) => {
          const k = k1 + 1;
          if (k >= t) return null;
          const isBack = mode === 1 && vit.back[k - 1][j] === i;
          const path = mode === 1 && done && vit.path[k - 1] === i && vit.path[k] === j;
          return <line key={`${k}${i}${j}`} x1={f1(cx(k - 1))} y1={f1(cy(i))} x2={f1(cx(k))} y2={f1(cy(j))} className={`tr-edge${isBack ? " back" : ""}${path ? " path" : ""}`} />;
        })))}
        {O.map((_, k) => [0, 1].map((i) => (
          <g key={`n${k}${i}`} className={k < t ? "tr-on" : "tr-off"}>
            <circle cx={f1(cx(k))} cy={f1(cy(i))} r={15} className={`tr-node${onPath(k, i) ? " path" : ""}`} />
            {k < t && <text x={f1(cx(k))} y={f1(cy(i) + 30)} textAnchor="middle" className="tr-val">{fmt(V[k][i])}</text>}
          </g>
        )))}
      </svg>
      <Slider label={TEXT.t[lang]} value={t} min={1} max={T} step={1} onChange={setT} format={(v) => `${v} / ${T}`} />
      <div className="w-stats">
        <Stat label={TEXT.p[lang]} value={done ? fmt(al[T - 1][0] + al[T - 1][1]) : "…"} tone={mode === 0 ? "accent" : undefined} />
        <Stat label={TEXT.best[lang]} value={done ? fmt(vit.prob) : "…"} tone={mode === 1 ? "accent" : undefined} />
        <Stat label={TEXT.path[lang]} value={done ? vit.path.map((s) => TEXT.states[lang][s]).join(" → ") : "…"} />
      </div>
    </Widget>
  );
}
