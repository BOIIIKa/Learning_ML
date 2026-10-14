import { useMemo, useState } from "react";
import { lloyd, seedPlusPlus } from "../../../engines/clustering";
import { emRun, shapedGroups, type Gmm } from "../../../engines/em-mixtures";
import { eigSym } from "../../../engines/linalg";
import type { Lang } from "../../../i18n/ui";
import MetricChart from "../../viz/MetricChart";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "../ch10/ch10.css";
import "./ch11.css";

const TEXT = {
  title: { zh: "高斯混合的 EM：软分配，椭圆形的簇", en: "EM for a Gaussian mixture: soft assignments, elliptical clusters" },
  rnd: { zh: "粗糙的初值", en: "Rough start" },
  km: { zh: "用 k 均值初始化", en: "Start from k-means" },
  it: { zh: "EM 迭代", en: "EM iteration" },
  ll: { zh: "对数似然", en: "Log-likelihood" },
  caption: {
    zh: "150 位病人来自三个形状不同的亚型：一个细长、一个圆、一个倾斜。每个亚型是一个高斯，椭圆是 2σ 等高线。E 步：算每个病人属于各亚型的后验概率（责任度，颜色深浅表示把握）；M 步：用责任度加权，重新估计每个高斯的权重、均值和协方差。对数似然每一步都上升。与第 10 章的 k 均值不同，簇可以是椭圆，分配可以是“七成像 A、三成像 B”。实践中常先跑 k 均值，用它的结果初始化 EM。数据为合成示意。",
    en: "150 patients from three subtypes of different shapes: one elongated, one round, one tilted. Each subtype is a Gaussian; the ellipses are 2σ contours. The E-step computes each patient's posterior probability of belonging to each subtype (the responsibility; colour strength shows confidence); the M-step re-estimates each Gaussian's weight, mean and covariance from the responsibility-weighted data. The log-likelihood rises at every step. Unlike Chapter 10's k-means, clusters can be ellipses and assignments can be \"70% like A, 30% like B\". In practice EM is often started from a k-means solution. Synthetic illustrative data.",
  },
};

const f1 = (v: number) => v.toFixed(1);
const sx = (x: number) => 230 + x * 70;
const sy = (y: number) => 150 - y * 58;
const I: [[number, number], [number, number]] = [[1, 0], [0, 1]];

function ellipse(m: [number, number], S: [[number, number], [number, number]]) {
  const { values, vectors } = eigSym(S);
  return "M" + Array.from({ length: 41 }, (_, i) => {
    const t = (2 * Math.PI * i) / 40;
    const a = 2 * Math.sqrt(values[0]) * Math.cos(t), b = 2 * Math.sqrt(Math.max(0, values[1])) * Math.sin(t);
    return `${f1(sx(m[0] + a * vectors[0][0] + b * vectors[1][0]))},${f1(sy(m[1] + a * vectors[0][1] + b * vectors[1][1]))}`;
  }).join("L") + "Z";
}

export default function GmmEm({ lang }: { lang: Lang }) {
  const X = useMemo(() => shapedGroups(), []);
  const [km, setKm] = useState(false);
  const runs = useMemo(() => {
    const rough: Gmm = { w: [1 / 3, 1 / 3, 1 / 3], mu: [[-0.5, 0.5], [0.5, 0.5], [0, -0.5]], S: [I, I, I] };
    const k = lloyd(X, seedPlusPlus(X, 3, 1)).at(-1)!;
    const fromKm: Gmm = { w: [1 / 3, 1 / 3, 1 / 3], mu: k.C.map((c) => [c[0], c[1]] as [number, number]), S: [[[0.3, 0], [0, 0.3]], [[0.3, 0], [0, 0.3]], [[0.3, 0], [0, 0.3]]] };
    return [emRun(X, rough, 40), emRun(X, fromKm, 40)];
  }, [X]);
  const run = runs[km ? 1 : 0];
  const [t, setT] = useState(0);
  const g = run.models[t];
  const r = run.r[t];
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="scenario">
      <div className="chips">
        <button className={`chip${!km ? " on" : ""}`} onClick={() => setKm(false)}>{TEXT.rnd[lang]}</button>
        <button className={`chip${km ? " on" : ""}`} onClick={() => setKm(true)}>{TEXT.km[lang]}</button>
      </div>
      <svg className="lm-svg km-svg" viewBox="0 0 460 300" role="img">
        {X.map((p, i) => {
          const k = r[i].indexOf(Math.max(...r[i]));
          return <circle key={i} cx={f1(sx(p[0]))} cy={f1(sy(p[1]))} r={4} className={`km-pt c${k}`} opacity={(0.25 + 0.75 * Math.max(...r[i])).toFixed(2)} />;
        })}
        {g.mu.map((m, k) => <path key={k} d={ellipse(m, g.S[k])} className={`gm-ell c${k}`} />)}
      </svg>
      <MetricChart series={[{ key: "ll", label: TEXT.ll[lang], values: run.ll, color: "var(--accent)" }]} cursor={t} height={110} zero={false} />
      <Slider label={TEXT.it[lang]} value={t} min={0} max={run.models.length - 1} step={1} onChange={setT} />
      <div className="w-stats">
        <Stat label={TEXT.ll[lang]} value={run.ll[t].toFixed(1)} tone="accent" />
        <Stat label="π" value={g.w.map((w) => w.toFixed(2)).join(" / ")} />
      </div>
    </Widget>
  );
}
