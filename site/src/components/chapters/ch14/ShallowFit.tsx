import { useMemo, useState } from "react";
import { sample } from "../../../engines/generalization";
import { countPieces, knotsFor, shallowFit } from "../../../engines/neural-networks";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch14.css";

const TEXT = {
  title: { zh: "浅层网络 = 折线：隐单元就是“铰链”", en: "A shallow network is a broken line: hidden units are hinges" },
  D: { zh: "隐单元个数 D", en: "Hidden units D" },
  units: { zh: "显示每个隐单元的贡献", en: "Show each unit's contribution" },
  pieces: { zh: "线性段数（≤ D + 1）", en: "Linear pieces (≤ D + 1)" },
  mse: { zh: "训练均方误差", en: "Training MSE" },
  hours: { zh: "每周学习小时数", en: "Study hours per week" },
  score: { zh: "考试成绩", en: "Exam score" },
  caption: {
    zh: "老师的 40 位学生：成绩随学习时间先快速上升，后来趋于饱和。一个隐层的 ReLU 网络 f(x) = φ₀ + Σ φ_d · max(0, x − k_d) 中，每个隐单元在 k_d 处贡献一个“铰链”（竖线），输出层把它们加权求和——结果是一条折线，线性段最多 D + 1 段。D = 1 只能画一条斜线；D 增大，折线越来越贴近弯曲的趋势。这里铰链位置固定均匀分布，只用最小二乘求输出层；真正训练时铰链位置也会被学出来（第 16 章）。",
    en: "The teacher's 40 students: scores rise quickly with study time, then level off. In a one-hidden-layer ReLU network f(x) = φ₀ + Σ φ_d · max(0, x − k_d), each hidden unit contributes a hinge at k_d (the vertical lines) and the output layer adds them up with weights. The result is a broken line with at most D + 1 linear pieces. With D = 1 it is a single slope; as D grows, the broken line hugs the curved trend. Here the hinges are fixed and evenly spread and only the output layer is fitted by least squares; in real training the hinge positions are learned too (chapter 16).",
  },
};

export default function ShallowFit({ lang }: { lang: Lang }) {
  const [D, setD] = useState(3);
  const [show, setShow] = useState(true);
  const data = useMemo(() => sample(40, 11), []);
  const m = useMemo(() => shallowFit(data.xs, data.ys, knotsFor(D)), [data, D]);
  const pieces = useMemo(() => countPieces(m.f, -1, 11), [m]);
  const mse = data.xs.reduce((s, x, i) => s + (m.f(x) - data.ys[i]) ** 2, 0) / data.xs.length;
  const base = m.phi[0];
  const curves = [
    ...(show ? m.knots.map((_, d) => ({ f: (x: number) => base + m.unit(d, x), color: "var(--accent-2)", width: 1.2, opacity: 0.45, dashed: true })) : []),
    { f: m.f, color: "var(--class-pos)", width: 3 },
  ];
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="scenario">
      <CurvePlot xs={data.xs} ys={data.ys} xDomain={[0, 10]} yDomain={[10, 110]} curves={curves}
        vlines={m.knots.filter((k) => k > 0.05).map((k) => ({ x: k, color: "var(--accent-2)" }))} xLabel={TEXT.hours[lang]} yLabel={TEXT.score[lang]} />
      <Slider label={TEXT.D[lang]} value={D} min={1} max={12} step={1} onChange={setD} />
      <div className="chips">
        <button className={`chip${show ? " on" : ""}`} onClick={() => setShow(!show)}>{TEXT.units[lang]}</button>
      </div>
      <div className="w-stats">
        <Stat label={TEXT.pieces[lang]} value={pieces} tone="accent" />
        <Stat label={TEXT.mse[lang]} value={mse.toFixed(1)} />
      </div>
    </Widget>
  );
}
