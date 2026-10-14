import { scaleLinear } from "d3-scale";
import { useEffect, useMemo, useState } from "react";
import { rng } from "../../../engines/data";
import { TRUE_SEPARATOR, novikoffBound, perceptron, separablePatients } from "../../../engines/linclass";
import type { Lang } from "../../../i18n/ui";
import Widget, { Stat } from "../../ui/Widget";
import { Arrow, f1, halfPlane, lineInBox, type Box } from "./geom";
import "./ch04.css";

const TEXT = {
  title: { zh: "感知机逐步更新：错一个，改一次", en: "The perceptron, one mistake at a time" },
  primal: { zh: "原始形式", en: "Primal" },
  dual: { zh: "对偶形式", en: "Dual" },
  shuffle: { zh: "换一种遍历顺序", en: "Another visiting order" },
  reset: { zh: "回到开头", en: "Restart" },
  update: { zh: "第几次更新", en: "Update" },
  wrong: { zh: "当前误分类点", en: "Misclassified now" },
  bound: { zh: "Novikoff 上界 (R/γ)²", en: "Novikoff bound (R/γ)²" },
  sumAlpha: { zh: "Σ αᵢ（= 更新次数 × η）", en: "Σ αᵢ (= updates × η)" },
  nonzero: { zh: "αᵢ > 0 的点", en: "Points with αᵢ > 0" },
  start: {
    zh: "从 w = 0、b = 0 出发：所有点都在“边界上”，都算误分类。按 › 或 ▶ 开始。",
    en: "Start from w = 0, b = 0: every point sits on the “boundary” and counts as a mistake. Press › or ▶ to begin.",
  },
  done: { zh: "一整轮没有误分类——算法停止。", en: "A full pass without a mistake: the algorithm stops." },
  caption: {
    zh: "黄色脉冲圈是这一步被选中的误分类点 xᵢ。紫色箭头是更新量 Δw = η yᵢ xᵢ：正类点把 w 拉向自己，负类点把 w 推离自己；灰色虚线是更新前的边界。切到“对偶形式”，同样的过程改用“每个点被用了几次”（αᵢ）来记录，点的大小与 αᵢ 成正比——最终 w = Σ αᵢ yᵢ xᵢ 只由少数犯过错的点决定。换一种遍历顺序，得到的是另一条同样能分开数据的线：感知机的解不唯一。",
    en: "The pulsing yellow ring marks this step's misclassified point xᵢ. The purple arrow is the update Δw = η yᵢ xᵢ: a positive point pulls w towards itself, a negative point pushes it away; the grey dashed line is the boundary before the update. Switch to the dual form and the same run is recorded as “how many times each point was used” (αᵢ), drawn as point size; the final w = Σ αᵢ yᵢ xᵢ depends only on the few points that ever caused a mistake. Try another visiting order: you get a different line that also separates the data. The perceptron's solution is not unique.",
  },
};

const W = 460;
const H = 400;
const BOX: Box = { x: [-3.2, 3.2], y: [-2.78, 2.78] };
const ETA = 0.5;

function permutation(n: number, seed: number) {
  const idx = Array.from({ length: n }, (_, i) => i);
  if (seed === 0) return idx;
  const rand = rng(seed);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx;
}

export default function PerceptronSteps({ lang }: { lang: Lang }) {
  const { X, y } = useMemo(() => separablePatients(16, 9, 0.2), []);
  const [seed, setSeed] = useState(0);
  const steps = useMemo(() => perceptron(X, y, ETA, permutation(X.length, seed)), [X, y, seed]);
  const [k, setK] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [mode, setMode] = useState<"primal" | "dual">("primal");
  const bound = useMemo(() => novikoffBound(X, y, TRUE_SEPARATOR.w, TRUE_SEPARATOR.b).bound, [X, y]);

  useEffect(() => {
    if (!playing) return;
    const id = window.setTimeout(() => (k >= steps.length - 1 ? setPlaying(false) : setK(k + 1)), 900);
    return () => clearTimeout(id);
  }, [playing, k, steps.length]);

  const sx = scaleLinear().domain(BOX.x).range([0, W]);
  const sy = scaleLinear().domain(BOX.y).range([H, 0]);
  const S = (p: number[]): [number, number] => [sx(p[0]), sy(p[1])];
  const poly = (pts: [number, number][]) => pts.map((p) => `${f1(sx(p[0]))},${f1(sy(p[1]))}`).join(" ");

  const cur = steps[k];
  const prev = k > 0 ? steps[k - 1] : null;
  const hasLine = cur.w[0] !== 0 || cur.w[1] !== 0;
  const seg = hasLine ? lineInBox(cur.w, cur.b, BOX) : null;
  const prevSeg = prev && (prev.w[0] !== 0 || prev.w[1] !== 0) ? lineInBox(prev.w, prev.b, BOX) : null;
  const wrong = new Set(X.flatMap((x, i) => (y[i] * (cur.w[0] * x[0] + cur.w[1] * x[1] + cur.b) <= 0 ? [i] : [])));
  const i = cur.i;
  const sumAlpha = cur.alpha.reduce((a, v) => a + v, 0) * ETA;
  const nonzero = cur.alpha.filter((a) => a > 0).length;

  const fmt = (v: number) => (Math.abs(v) < 5e-4 ? "0" : v.toFixed(2));
  let note: React.ReactNode = TEXT.start[lang];
  if (i >= 0) {
    const sign = y[i] > 0 ? "+" : "−";
    note = lang === "zh"
      ? <>点 <b>#{i + 1}</b>（{y[i] > 0 ? "正类" : "负类"}）被分错：<code>w ← w {sign} η·x<sub>{i + 1}</sub></code>，<code>b ← b {sign} η</code>，得到 w = ({fmt(cur.w[0])}, {fmt(cur.w[1])})，b = {fmt(cur.b)}。{k === steps.length - 1 && TEXT.done.zh}</>
      : <>Point <b>#{i + 1}</b> ({y[i] > 0 ? "positive" : "negative"}) is misclassified: <code>w ← w {sign} η·x<sub>{i + 1}</sub></code>, <code>b ← b {sign} η</code>, giving w = ({fmt(cur.w[0])}, {fmt(cur.w[1])}), b = {fmt(cur.b)}. {k === steps.length - 1 && TEXT.done.en}</>;
  }

  const go = (v: number) => { setPlaying(false); setK(Math.max(0, Math.min(steps.length - 1, v))); };

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="replay">
      <svg className="lc-svg" viewBox={`0 0 ${W} ${H}`} role="img">
        <rect width={W} height={H} rx={12} className="plot-bg" />
        {hasLine && (
          <>
            <polygon points={poly(halfPlane(cur.w, cur.b, BOX))} className="lc-half pos" />
            <polygon points={poly(halfPlane([-cur.w[0], -cur.w[1]], -cur.b, BOX))} className="lc-half neg" />
          </>
        )}
        <line x1={sx(BOX.x[0])} x2={sx(BOX.x[1])} y1={sy(0)} y2={sy(0)} className="lc-axis" />
        <line x1={sx(0)} x2={sx(0)} y1={sy(BOX.y[0])} y2={sy(BOX.y[1])} className="lc-axis" />
        {prevSeg && <line x1={f1(sx(prevSeg[0][0]))} y1={f1(sy(prevSeg[0][1]))} x2={f1(sx(prevSeg[1][0]))} y2={f1(sy(prevSeg[1][1]))} className="lc-sep-prev" />}
        {seg && <line x1={f1(sx(seg[0][0]))} y1={f1(sy(seg[0][1]))} x2={f1(sx(seg[1][0]))} y2={f1(sy(seg[1][1]))} className="lc-sep" />}

        {i >= 0 && <line x1={sx(0)} y1={sy(0)} x2={f1(sx(X[i][0]))} y2={f1(sy(X[i][1]))} className="lc-xvec" />}

        {X.map((p, j) => {
          const r = mode === "dual" ? 5 + 2.4 * Math.sqrt(cur.alpha[j]) : 6;
          return (
            <g key={j} transform={`translate(${f1(sx(p[0]))},${f1(sy(p[1]))})`}>
              {j === i && <circle r={r + 4} className="lc-hl" />}
              {wrong.has(j) && hasLine && <circle r={r + 3.5} className="lc-wrong" />}
              <circle r={r} className={`lc-pt ${y[j] > 0 ? "pos" : "neg"}`} />
              {mode === "dual" && cur.alpha[j] > 0 && <text x={r + 3} y={-r - 1} className="lc-alpha">α={(cur.alpha[j] * ETA).toFixed(1)}</text>}
            </g>
          );
        })}

        {mode === "primal" && prev && <Arrow from={S([0, 0])} to={S(prev.w)} cls="wprev" />}
        {mode === "primal" && prev && i >= 0 && <Arrow from={S(prev.w)} to={S(cur.w)} cls="dw" />}
        {mode === "primal" && hasLine && <Arrow from={S([0, 0])} to={S(cur.w)} cls="w" />}
        {mode === "primal" && hasLine && <text x={f1(sx(cur.w[0]) + 8)} y={f1(sy(cur.w[1]) - 6)} className="lc-vlab w">w</text>}
        {mode === "primal" && prev && i >= 0 && (
          <text x={f1((sx(prev.w[0]) + sx(cur.w[0])) / 2 + 8)} y={f1((sy(prev.w[1]) + sy(cur.w[1])) / 2 + 14)} className="lc-vlab dw">Δw</text>
        )}
      </svg>

      <div className="lc-note">{note}</div>

      <div className="lc-controls">
        <button className="btn tp-icon" onClick={() => go(0)} aria-label={TEXT.reset[lang]}>⏮</button>
        <button className="btn tp-icon" onClick={() => go(k - 1)} aria-label="-1">‹</button>
        <button className="btn tp-play" onClick={() => { if (k >= steps.length - 1) setK(0); setPlaying(!playing); }} aria-label="play">{playing ? "❚❚" : "▶"}</button>
        <button className="btn tp-icon" onClick={() => go(k + 1)} aria-label="+1">›</button>
        <button className="btn" onClick={() => { setPlaying(false); setK(0); setSeed(seed + 1); }}>{TEXT.shuffle[lang]}</button>
        <div className="chips">
          {(["primal", "dual"] as const).map((m) => (
            <button key={m} className={`chip${mode === m ? " on" : ""}`} onClick={() => setMode(m)}>{TEXT[m][lang]}</button>
          ))}
        </div>
      </div>

      <div className="w-stats">
        <Stat label={TEXT.update[lang]} value={`${k} / ${steps.length - 1}`} tone="accent" />
        <Stat label={TEXT.wrong[lang]} value={wrong.size} tone={wrong.size ? "bad" : "good"} />
        {mode === "primal" ? (
          <>
            <Stat label="w" value={`(${fmt(cur.w[0])}, ${fmt(cur.w[1])})`} />
            <Stat label="b" value={fmt(cur.b)} />
          </>
        ) : (
          <>
            <Stat label={TEXT.sumAlpha[lang]} value={sumAlpha.toFixed(1)} />
            <Stat label={TEXT.nonzero[lang]} value={`${nonzero} / ${X.length}`} />
          </>
        )}
        <Stat label={TEXT.bound[lang]} value={bound.toFixed(0)} />
      </div>
    </Widget>
  );
}
