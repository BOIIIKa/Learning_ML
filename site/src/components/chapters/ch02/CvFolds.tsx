import { scaleLinear } from "d3-scale";
import { useEffect, useMemo, useState } from "react";
import { cohort, crossValidate, cvSimulation, linspace, meanSd, polyPredict, type CvSimulation } from "../../../engines/generalization";
import type { Lang } from "../../../i18n/ui";
import Widget, { Stat } from "../../ui/Widget";
import FitPlot from "./FitPlot";
import "./ch02.css";

const TEXT = {
  title: { zh: "k 折交叉验证：一折一折地看", en: "k-fold cross-validation, one fold at a time" },
  k: { zh: "折数 k", en: "Folds k" },
  degree: { zh: "阶数 d", en: "Degree d" },
  play: { zh: "▶ 播放", en: "▶ Play" },
  pause: { zh: "❚❚ 暂停", en: "❚❚ Pause" },
  fold: { zh: "当前这一折的 MSE", en: "MSE on this fold" },
  cv: { zh: "CV 估计（这批数据）", en: "CV estimate (this dataset)" },
  spread: { zh: "150 批数据：均值 ± 标准差", en: "Over 150 datasets: mean ± sd" },
  trueErr: { zh: "真实测试误差", en: "True test error" },
  folds: { zh: "各折误差（橙 = 当前折，横线 = 平均，即 CV 估计）", en: "Per-fold error (orange = current fold; horizontal line = average, the CV estimate)" },
  strip: { zh: "150 批数据上的 CV 估计（点）与均值 ± 标准差（短横与竖线）；绿色虚线为真实测试误差，超过 300 的点画在顶端", en: "CV estimates on 150 datasets (dots) with mean ± sd (tick and whisker); dashed green = true test error; values above 300 sit on the top edge" },
  busy: { zh: "正在模拟 150 批数据…", en: "Simulating 150 datasets…" },
  x: { zh: "每周学习时长（小时）", en: "Study hours per week" },
  y: { zh: "成绩", en: "Score" },
  loo: { zh: "留一", en: "LOO" },
  caption: {
    zh: "24 位学生被随机分成 k 折。每一步把一折（橙色、闪烁）留作验证，用其余各折（蓝色）拟合，再在留出的这一折上算误差；k 步之后取平均，就是 CV 估计。最下面把同样的流程在 150 批独立数据上重复：k 越小，每次只用较少的数据训练，估计偏高（悲观）且更分散；k 增大到留一法（LOO）时，均值贴近真实测试误差，但要训练 n 次。",
    en: "The 24 students are shuffled into k folds. Each step holds one fold out (orange, pulsing), fits on the remaining folds (blue), and scores the held-out fold; the average over the k steps is the CV estimate. The bottom panel repeats the procedure on 150 independent datasets: with small k each model trains on less data, so the estimate runs high (pessimistic) and is more spread out; as k grows to leave-one-out (LOO) its mean approaches the true test error, at the cost of n fits.",
  },
};

const KS = [2, 3, 4, 6, 8, 12, 24];
const DEGS = [1, 3, 5];
const GRID = linspace(0.5, 9.5, 91);
const W = 460;
const f1 = (v: number) => v.toFixed(1);

export default function CvFolds({ lang }: { lang: Lang }) {
  const data = useMemo(() => cohort(24, 8, 21), []);
  const [k, setK] = useState(4);
  const [d, setD] = useState(3);
  const [j, setJ] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [sim, setSim] = useState<{ d: number; s: CvSimulation } | null>(null);
  const cv = useMemo(() => crossValidate(data.xs, data.ys, d, k, 7), [data, d, k]);
  const jj = Math.min(j, k - 1);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setJ((x) => (x + 1) % k), k > 12 ? 450 : 1100);
    return () => clearInterval(id);
  }, [playing, k]);

  // The Monte Carlo study runs in the browser only, after first paint.
  useEffect(() => {
    const id = setTimeout(() => setSim({ d, s: cvSimulation(24, 8, d, KS, 150, 5) }), 30);
    return () => clearTimeout(id);
  }, [d]);

  const val = new Set(cv.folds[jj]);
  const tr = data.xs.map((_, i) => i).filter((i) => !val.has(i));
  const pick = (idx: number[]) => ({ xs: idx.map((i) => data.xs[i]), ys: idx.map((i) => data.ys[i]) });
  const w = cv.weights[jj];

  // per-fold bars
  const barH = 64;
  const maxErr = Math.max(...cv.foldMse, cv.estimate) * 1.1;
  const bw = (W - 50) / k;
  const by = scaleLinear().domain([0, maxErr]).range([barH - 6, 6]);

  // strip plot
  const sH = 200;
  const sx = scaleLinear().domain([-0.5, KS.length - 0.5]).range([40, W - 10]);
  const sy = scaleLinear().domain([0, 300]).range([sH - 26, 10]);
  const cy = (v: number) => f1(sy(Math.min(v, 300)));
  const ready = sim && sim.d === d ? sim.s : null;
  const kIdx = KS.indexOf(k);
  const stats = ready ? meanSd(ready.estimates[kIdx]) : null;

  const chip = (v: number, cur: number, set: (v: number) => void, label: string) => (
    <button key={v} className={`chip${v === cur ? " on" : ""}`} onClick={() => { set(v); setJ(0); }}>{label}</button>
  );

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="triad-pickers">
        <div><span className="tp-k">{TEXT.k[lang]}</span><div className="chips">{KS.map((v) => chip(v, k, setK, v === 24 ? `24 (${TEXT.loo[lang]})` : String(v)))}</div></div>
        <div><span className="tp-k">{TEXT.degree[lang]}</span><div className="chips">{DEGS.map((v) => chip(v, d, setD, String(v)))}</div></div>
      </div>
      <FitPlot grid={GRID} xDomain={[0, 10]} yDomain={[10, 100]} xLabel={TEXT.x[lang]} yLabel={TEXT.y[lang]} height={240}
        curves={[{ ys: GRID.map((x) => polyPredict(w, x)), className: "g2-fit", key: "fit" }]}
        points={[{ ...pick(tr), className: "g2-train" }, { ...pick(cv.folds[jj]), className: "g2-val", ring: true, r: 5 }]} />
      <figure className="g2-chart">
        <svg className="g2-svg" viewBox={`0 0 ${W} ${barH}`} role="img">
          <rect width={W} height={barH} rx={10} className="plot-bg" />
          {cv.foldMse.map((e, i) => (
            <rect key={i} x={f1(40 + i * bw + bw * 0.15)} y={f1(by(e))} width={f1(bw * 0.7)} height={f1(by(0) - by(e))} className={`g2-bar${i === jj ? " on" : ""}`} />
          ))}
          <line x1={40} x2={W - 10} y1={f1(by(cv.estimate))} y2={f1(by(cv.estimate))} className="g2-bar-mean" />
          <text x={34} y={f1(by(cv.estimate))} textAnchor="end" dominantBaseline="middle" className="g2-tick">{cv.estimate.toFixed(0)}</text>
        </svg>
        <figcaption className="g2-legend">{TEXT.folds[lang]}</figcaption>
      </figure>
      <div className="w-row">
        <button className="btn" onClick={() => setPlaying(!playing)}>{playing ? TEXT.pause[lang] : TEXT.play[lang]}</button>
        <span className="muted">{lang === "zh" ? `第 ${jj + 1} / ${k} 折` : `fold ${jj + 1} / ${k}`}</span>
      </div>
      <div className="w-stats">
        <Stat label={TEXT.fold[lang]} value={cv.foldMse[jj].toFixed(1)} />
        <Stat label={TEXT.cv[lang]} value={cv.estimate.toFixed(1)} tone="accent" />
        <Stat label={TEXT.spread[lang]} value={stats ? `${stats.mean.toFixed(0)} ± ${stats.sd.toFixed(0)}` : "…"} />
        <Stat label={TEXT.trueErr[lang]} value={ready ? ready.trueError.toFixed(1) : "…"} tone="good" />
      </div>
      <figure className="g2-chart">
        <svg className="g2-svg" viewBox={`0 0 ${W} ${sH}`} role="img">
          <rect width={W} height={sH} rx={12} className="plot-bg" />
          {[0, 100, 200, 300].map((t) => (
            <g key={t}>
              <line x1={40} x2={W - 10} y1={sy(t)} y2={sy(t)} className="g2-grid" />
              <text x={34} y={sy(t)} textAnchor="end" dominantBaseline="middle" className="g2-tick">{t}</text>
            </g>
          ))}
          <rect x={f1(sx(kIdx - 0.5))} y={10} width={f1(sx(1) - sx(0))} height={sH - 36} className="g2-kcol" />
          {KS.map((v, i) => <text key={v} x={f1(sx(i))} y={sH - 9} textAnchor="middle" className="g2-tick">{v === 24 ? `k=24` : `k=${v}`}</text>)}
          {ready ? (
            <>
              {ready.estimates.map((est, i) => est.map((e, r) => (
                <circle key={`${i}-${r}`} cx={f1(sx(i) + 26 * (((r * 0.618034) % 1) - 0.5))} cy={cy(e)} r={2.2} className={`g2-jit${i === kIdx ? " on" : ""}`} />
              )))}
              {ready.estimates.map((est, i) => {
                const s = meanSd(est);
                return (
                  <g key={`m${i}`}>
                    <line x1={f1(sx(i))} x2={f1(sx(i))} y1={cy(s.mean - s.sd)} y2={cy(s.mean + s.sd)} className="g2-sd" />
                    <line x1={f1(sx(i) - 14)} x2={f1(sx(i) + 14)} y1={cy(s.mean)} y2={cy(s.mean)} className="g2-meanbar" />
                  </g>
                );
              })}
              <line x1={40} x2={W - 10} y1={cy(ready.trueError)} y2={cy(ready.trueError)} className="g2-true" />
            </>
          ) : (
            <text x={W / 2} y={sH / 2} textAnchor="middle" className="g2-placeholder">{TEXT.busy[lang]}</text>
          )}
        </svg>
        <figcaption className="g2-legend">{TEXT.strip[lang]}</figcaption>
      </figure>
    </Widget>
  );
}
