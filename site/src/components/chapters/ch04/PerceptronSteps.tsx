import { useId, useMemo, useState } from "react";
import { mistakes, novikoff, perceptron, screening, shuffled } from "../../../engines/linear-classification";
import type { Lang } from "../../../i18n/ui";
import Plot2D from "../../viz/Plot2D";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch04.css";

const TEXT = {
  title: { zh: "感知机：一次只改一个错误", en: "The perceptron: fix one mistake at a time" },
  k: { zh: "第几次更新 k", en: "Update k" },
  primal: { zh: "原始形式：w", en: "Primal: w" },
  dual: { zh: "对偶形式：α", en: "Dual: α" },
  reorder: { zh: "换一种访问顺序", en: "Visit in another order" },
  wspace: { zh: "w 空间：w ← w + y·x", en: "w-space: w ← w + y·x" },
  alphas: { zh: "α_i：每个点被用了几次", en: "α_i: how often each point was used" },
  errors: { zh: "当前误分类", en: "Mistakes now" },
  bound: { zh: "Novikoff 上界 (R/γ)²", en: "Novikoff bound (R/γ)²" },
  total: { zh: "本顺序共更新", en: "Updates, this order" },
  caption: {
    zh: "医院的 24 位病人，两项标准化后的血液指标；红色需要复查，蓝色不需要。从 w = 0 出发，每次找到一个分错的病人（黄圈），把它“加进” w：w ← w + y·x，b ← b + y。虚线是更新前的直线。右下角的小图是同一步在 w 空间里的向量加法。切到“对偶形式”，点的大小变成 α_i：最终的 w 只是这些病人的加权和。换一种访问顺序，会停在另一条同样零错误的直线上——解不唯一。数据为合成示意。",
    en: "24 patients with two standardized blood markers; red need a follow-up, blue do not. Starting from w = 0, each step takes one misclassified patient (yellow ring) and adds it into w: w ← w + y·x, b ← b + y. The dashed line is the line before the update. The small panel shows the same step as vector addition in w-space. Switch to the dual form and point size becomes α_i: the final w is just a weighted sum of these patients. Visit them in another order and you stop at a different, equally error-free line: the solution is not unique. Synthetic illustrative data.",
  },
};

const f1 = (v: number) => v.toFixed(1);
const IW = 170;
const IH = 130;

export default function PerceptronSteps({ lang }: { lang: Lang }) {
  const uid = useId().replace(/:/g, "");
  const { X, y } = useMemo(() => screening(), []);
  const { bound } = useMemo(() => novikoff(X, y), [X, y]);
  const [seed, setSeed] = useState(0);
  const run = useMemo(() => perceptron(X, y, 1, seed === 0 ? undefined : shuffled(X.length, seed)), [X, y, seed]);
  const K = run.steps.length - 1;
  const [kRaw, setK] = useState(K);
  const k = Math.min(kRaw, K);
  const [dual, setDual] = useState(false);

  const cur = run.steps[k];
  const prev = run.steps[Math.max(0, k - 1)];
  const err = mistakes(X, y, cur.w, cur.b);
  const amax = Math.max(1, ...run.steps[K].alpha);
  const weights = dual ? cur.alpha.map((a) => (a > 0 ? 0.5 + (1.6 * a) / amax : 0.25)) : undefined;

  // w-space inset: scale so this step's vectors (old w, new w) fill the panel.
  const wmax = Math.max(0.5, Math.hypot(...prev.w), Math.hypot(...cur.w));
  const sc = (0.42 * IH) / wmax;
  const P = (v: [number, number]) => [IW / 2 + sc * v[0], IH / 2 - sc * v[1]];
  const arrow = (from: [number, number], to: [number, number], cls: string) => {
    const [x1, y1] = P(from);
    const [x2, y2] = P(to);
    return <line x1={f1(x1)} y1={f1(y1)} x2={f1(x2)} y2={f1(y2)} className={cls} markerEnd={`url(#ah-${uid})`} />;
  };

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="scenario">
      <Plot2D
        X={X}
        y={y}
        domain={{ x: [-2.9, 2.9], y: [-2.1, 2.5] }}
        height={340}
        showMargin={false}
        step={k > 0 ? { w: cur.w, b: cur.b, highlight: cur.i >= 0 ? [cur.i] : [], weights } : { highlight: [], weights }}
        extraLines={k > 1 ? [{ w: prev.w, b: prev.b, color: "var(--text-muted)", dashed: true }] : []}
        overlay={(sx, sy) =>
          dual ? (
            <g className="pc-alpha">
              {cur.alpha.map((a, i) => (a > 0 ? (
                <text key={i} x={f1(sx(X[i][0]) + 9)} y={f1(sy(X[i][1]) - 8)}>{a}</text>
              ) : null))}
            </g>
          ) : null
        }
      />
      <div className="pc-row">
        <svg className="pc-inset" viewBox={`0 0 ${IW} ${IH}`} role="img">
          <defs>
            <marker id={`ah-${uid}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0L10,5L0,10z" className="pc-head" />
            </marker>
          </defs>
          {dual ? (
            <>
              <text x={4} y={12} className="pc-cap">{TEXT.alphas[lang]}</text>
              {cur.alpha.map((a, i) => {
                const bw = (IW - 8) / X.length;
                const h = (a / amax) * (IH - 34);
                return <rect key={i} x={f1(4 + i * bw)} y={f1(IH - 12 - h)} width={f1(bw - 1.5)} height={f1(h)} className={`pc-bar ${y[i] > 0 ? "pos" : "neg"}${i === cur.i ? " hl" : ""}`} />;
              })}
              <line x1={4} x2={IW - 4} y1={IH - 12} y2={IH - 12} className="pc-axis" />
            </>
          ) : (
            <>
              <text x={4} y={12} className="pc-cap">{TEXT.wspace[lang]}</text>
              <line x1={0} x2={IW} y1={IH / 2} y2={IH / 2} className="pc-axis" />
              <line x1={IW / 2} x2={IW / 2} y1={16} y2={IH} className="pc-axis" />
              {k > 0 && arrow([0, 0], prev.w, "pc-old")}
              {k > 0 && cur.i >= 0 && arrow(prev.w, cur.w, `pc-step ${y[cur.i] > 0 ? "pos" : "neg"}`)}
              {k > 0 && arrow([0, 0], cur.w, "pc-new")}
            </>
          )}
        </svg>
        <div className="pc-side">
          <div className="chips">
            <button className={`chip${!dual ? " on" : ""}`} onClick={() => setDual(false)}>{TEXT.primal[lang]}</button>
            <button className={`chip${dual ? " on" : ""}`} onClick={() => setDual(true)}>{TEXT.dual[lang]}</button>
          </div>
          <button className="btn" onClick={() => { setSeed((s) => s + 1); setK(0); }}>↻ {TEXT.reorder[lang]}</button>
          <div className="w-stats">
            <Stat label={TEXT.errors[lang]} value={err} tone={err === 0 ? "good" : "bad"} />
            <Stat label={TEXT.total[lang]} value={K} tone="accent" />
            <Stat label={TEXT.bound[lang]} value={bound.toFixed(0)} />
          </div>
        </div>
      </div>
      <Slider label={TEXT.k[lang]} value={k} min={0} max={K} step={1} onChange={setK} format={(v) => `${v} / ${K}`} />
    </Widget>
  );
}
