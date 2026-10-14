import { useMemo, useState } from "react";
import { HMM, baumWelch, sampleHmm, type Hmm } from "../../../engines/structured-prediction";
import type { Lang } from "../../../i18n/ui";
import MetricChart from "../../viz/MetricChart";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch09.css";

const TEXT = {
  title: { zh: "Baum–Welch：没有标签也能学", en: "Baum–Welch: learning without labels" },
  it: { zh: "EM 迭代次数", en: "EM iteration" },
  ll: { zh: "对数似然", en: "Log-likelihood" },
  A: { zh: "转移矩阵 A（估计 / 真实）", en: "Transitions A (estimate / truth)" },
  B: { zh: "发射矩阵 B（估计 / 真实）", en: "Emissions B (estimate / truth)" },
  caption: {
    zh: "只给 400 天的化验结果，不给任何一天的真实病情，从一个粗糙的初值出发学 HMM 的参数。每次迭代：E 步用前向–后向算出每天处于各状态的后验 γ 和相邻两天的联合后验 ξ；M 步把它们当作“软计数”，像有标签时那样数数、归一化。对数似然单调上升（EM 的保证，第 11 章证明），参数逐步接近真实值。数据为合成示意。",
    en: "Given only 400 days of test results and not a single true state, learn the HMM's parameters from a rough start. Each iteration: the E-step runs forward–backward to get the posterior γ of each state on each day and the joint posterior ξ of consecutive days; the M-step treats them as soft counts and counts and normalizes as if labels were available. The log-likelihood never decreases (EM's guarantee, proved in Chapter 11) and the parameters approach the truth. Synthetic illustrative data.",
  },
};

const INIT: Hmm = { pi: [0.5, 0.5], A: [[0.6, 0.4], [0.5, 0.5]], B: [[0.7, 0.3], [0.4, 0.6]] };

// Plain divs, not <table>: the notes page decorates tables after load, which would break hydration.
function Mat({ M, T }: { M: number[][]; T: number[][] }) {
  return (
    <div className="bw-mat">
      {M.flatMap((row, i) => row.map((v, j) => (
        <div key={`${i}${j}`} className="bw-cell"><span className="bw-bar" style={{ width: `${Math.round(100 * v)}%` }} /><b>{v.toFixed(2)}</b> / {T[i][j].toFixed(2)}</div>
      )))}
    </div>
  );
}

export default function BaumWelchFit({ lang }: { lang: Lang }) {
  const { models, loglik } = useMemo(() => baumWelch(sampleHmm(HMM, 400, 7).O, INIT, 40), []);
  const [k, setK] = useState(0);
  const m = models[k];
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <MetricChart series={[{ key: "ll", label: TEXT.ll[lang], values: loglik, color: "var(--accent)" }]} cursor={k} zero={false} />
      <Slider label={TEXT.it[lang]} value={k} min={0} max={models.length - 1} step={1} onChange={setK} />
      <div className="bw-grid">
        <div><div className="bw-cap">{TEXT.A[lang]}</div><Mat M={m.A} T={HMM.A} /></div>
        <div><div className="bw-cap">{TEXT.B[lang]}</div><Mat M={m.B} T={HMM.B} /></div>
      </div>
      <div className="w-stats"><Stat label={TEXT.ll[lang]} value={loglik[k].toFixed(2)} tone="accent" /></div>
    </Widget>
  );
}
