import { useMemo, useState } from "react";
import { lloyd, seedPlusPlus, seedRandom, subgroups } from "../../../engines/clustering";
import type { Lang } from "../../../i18n/ui";
import MetricChart from "../../viz/MetricChart";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch10.css";

const TEXT = {
  title: { zh: "k 均值：分配，再更新", en: "k-means: assign, then update" },
  rnd: { zh: "随机初始化", en: "Random seeding" },
  pp: { zh: "k-means++ 初始化", en: "k-means++ seeding" },
  reseed: { zh: "换个随机种子", en: "New random seed" },
  step: { zh: "第几个半步", en: "Half-step" },
  k: { zh: "簇数 k", en: "Clusters k" },
  obj: { zh: "簇内平方和", en: "Within-cluster SS" },
  phase: { zh: { init: "初始化", assign: "分配", update: "更新" }, en: { init: "init", assign: "assign", update: "update" } },
  caption: {
    zh: "90 位病人，两项标准化指标，没有任何标签。k 均值交替两步：分配——每个病人归到最近的中心；更新——每个中心移到它那一簇的平均位置（叉号，细线是中心走过的路）。下方曲线是簇内平方和，每半步都不增加。换几个随机种子：有时两个中心挤进同一群病人，算法停在一个更差的局部最优；k-means++ 让后续中心以距离平方为概率挑选，很少这样。数据为合成示意。",
    en: "90 patients, two standardized markers, no labels at all. k-means alternates two steps: assign each patient to the nearest centre, then update each centre to the mean of its cluster (crosses; thin lines trace where the centres went). The curve below is the within-cluster sum of squares, which never rises from one half-step to the next. Try a few random seeds: sometimes two centres crowd into one group and the algorithm stops at a worse local optimum; k-means++ picks each further centre with probability proportional to squared distance and rarely does. Synthetic illustrative data.",
  },
};

const W = 460, H = 300;
const f1 = (v: number) => v.toFixed(1);
const sx = (x: number) => 230 + x * 75;
const sy = (y: number) => 150 - y * 62;

export default function KMeansSteps({ lang }: { lang: Lang }) {
  const { X } = useMemo(() => subgroups(), []);
  const [pp, setPp] = useState(false);
  const [seed, setSeed] = useState(3);
  const [k, setK] = useState(3);
  const run = useMemo(() => lloyd(X, pp ? seedPlusPlus(X, k, seed) : seedRandom(X, k, seed)), [X, pp, k, seed]);
  const [sRaw, setS] = useState(0);
  const s = Math.min(sRaw, run.length - 1);
  const cur = run[s];
  const trail = (j: number) => run.slice(0, s + 1).map((r) => `${f1(sx(r.C[j][0]))},${f1(sy(r.C[j][1]))}`).join("L");
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="scenario">
      <div className="chips">
        <button className={`chip${!pp ? " on" : ""}`} onClick={() => { setPp(false); setS(0); }}>{TEXT.rnd[lang]}</button>
        <button className={`chip${pp ? " on" : ""}`} onClick={() => { setPp(true); setS(0); }}>{TEXT.pp[lang]}</button>
        <button className="btn" onClick={() => { setSeed((v) => v + 1); setS(0); }}>↻ {TEXT.reseed[lang]}</button>
      </div>
      <svg className="lm-svg km-svg" viewBox={`0 0 ${W} ${H}`} role="img">
        {X.map((p, i) => <circle key={i} cx={f1(sx(p[0]))} cy={f1(sy(p[1]))} r={4} className={`km-pt c${cur.lab[i] % 6}`} />)}
        {cur.C.map((_, j) => <path key={`t${j}`} d={"M" + trail(j)} className="km-trail" />)}
        {cur.C.map((c, j) => (
          <g key={`c${j}`} transform={`translate(${f1(sx(c[0]))},${f1(sy(c[1]))})`} className={`km-c c${j % 6}`}>
            <path d="M-8,-8L8,8M-8,8L8,-8" />
          </g>
        ))}
      </svg>
      <MetricChart series={[{ key: "obj", label: TEXT.obj[lang], values: run.map((r) => r.obj), color: "var(--accent)" }]} cursor={s} height={110} />
      <Slider label={TEXT.step[lang]} value={s} min={0} max={run.length - 1} step={1} onChange={setS} format={(v) => `${v} / ${run.length - 1}`} />
      <Slider label={TEXT.k[lang]} value={k} min={2} max={6} step={1} onChange={(v) => { setK(v); setS(0); }} />
      <div className="w-stats">
        <Stat label={lang === "zh" ? "这一步" : "Phase"} value={TEXT.phase[lang][cur.phase]} />
        <Stat label={TEXT.obj[lang]} value={cur.obj.toFixed(1)} tone="accent" />
      </div>
    </Widget>
  );
}
