import { scaleLinear } from "d3-scale";
import { useEffect, useMemo, useRef, useState } from "react";
import { moons } from "../../../engines/data";
import { bestSplit, candidateSplits, gini } from "../../../engines/tree";
import type { Lang } from "../../../i18n/ui";
import Plot2D from "../../viz/Plot2D";
import Widget, { Slider, Stat } from "../../ui/Widget";

const TEXT = {
  title: { zh: "你来切第一刀", en: "Make the first cut yourself" },
  axis: { zh: "按哪个特征切", en: "Split on" },
  x1: { zh: "特征 1（半径）", en: "Feature 1 (radius)" },
  x2: { zh: "特征 2（纹理）", en: "Feature 2 (texture)" },
  thr: { zh: "阈值", en: "Threshold" },
  parent: { zh: "切前基尼", en: "Gini before" },
  child: { zh: "切后加权基尼", en: "Weighted Gini after" },
  gain: { zh: "不纯度下降", en: "Impurity decrease" },
  best: { zh: "让算法找最佳一刀", en: "Let the algorithm find the best cut" },
  curve: { zh: "每个阈值的不纯度下降（虚线 = 你的位置）", en: "Impurity decrease for every threshold (dashed = your choice)" },
  caption: {
    zh: "医生想要的是能说清楚的规则：“如果半径 ≤ t，则……”。左右拖动阈值，看一刀把数据分得有多“纯”。下方曲线扫描了所有阈值，算法就是取它的最高点。数据为合成示意。",
    en: "Doctors want rules they can explain: “if radius ≤ t, then …”. Drag the threshold and see how pure one cut makes the two sides. The curve below scans every threshold; the algorithm simply takes its peak. Synthetic illustrative data.",
  },
};

export default function ManualSplit({ lang }: { lang: Lang }) {
  const { X, y } = useMemo(() => moons(70, 5, 0.25), []);
  const all = useMemo(() => X.map((_, i) => i), [X]);
  const [feature, setFeature] = useState(0);
  const lo = Math.min(...X.map((p) => p[feature]));
  const hi = Math.max(...X.map((p) => p[feature]));
  const [t, setT] = useState(0.2);
  const anim = useRef(0);

  useEffect(() => setT((lo + hi) / 2), [feature, lo, hi]);

  const left = all.filter((i) => X[i][feature] <= t);
  const right = all.filter((i) => X[i][feature] > t);
  const rate = (idx: number[]) => (idx.length ? idx.filter((i) => y[i] > 0).length / idx.length : 0);
  const parent = gini(rate(all));
  const child = (left.length * gini(rate(left)) + right.length * gini(rate(right))) / all.length;
  const curve = useMemo(() => candidateSplits(X, y, all, [feature]), [X, y, all, feature]);

  const goBest = () => {
    const s = bestSplit(X, y, all)!;
    setFeature(s.feature);
    cancelAnimationFrame(anim.current);
    const from = s.feature === feature ? t : (lo + hi) / 2;
    const start = performance.now();
    const tick = (now: number) => {
      const k = Math.min(1, (now - start) / 700);
      setT(from + (s.threshold - from) * (1 - (1 - k) ** 3));
      if (k < 1) anim.current = requestAnimationFrame(tick);
    };
    // Wait one frame so the feature switch's reset does not overwrite the animation.
    requestAnimationFrame(() => (anim.current = requestAnimationFrame(tick)));
  };

  const line = feature === 0 ? { w: [1, 0] as [number, number], b: -t } : { w: [0, 1] as [number, number], b: -t };
  const W = 460;
  const H = 120;
  const sx = scaleLinear().domain([lo, hi]).range([10, W - 10]);
  const maxGain = Math.max(1e-9, ...curve.map((c) => c.gain));
  const sy = scaleLinear().domain([0, maxGain]).range([H - 14, 8]);

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="scenario">
      <div className="chips">
        {[0, 1].map((f) => (
          <button key={f} className={`chip${f === feature ? " on" : ""}`} onClick={() => setFeature(f)}>{f === 0 ? TEXT.x1[lang] : TEXT.x2[lang]}</button>
        ))}
      </div>
      <Plot2D X={X} y={y} extraLines={[{ ...line, color: "var(--warning)" }]} showMargin={false} highlightSide={{ feature, threshold: t }} />
      <Slider label={TEXT.thr[lang]} value={t} min={lo} max={hi} step={(hi - lo) / 200} onChange={setT} format={(v) => v.toFixed(2)} />
      <figure className="gain-curve">
        <svg viewBox={`0 0 ${W} ${H}`} role="img">
          <path className="gc-line" d={"M" + curve.map((c) => `${sx(c.threshold).toFixed(1)},${sy(c.gain).toFixed(1)}`).join("L")} />
          <line className="gc-cursor" x1={sx(t)} x2={sx(t)} y1={4} y2={H - 10} />
        </svg>
        <figcaption>{TEXT.curve[lang]}</figcaption>
      </figure>
      <div className="w-stats">
        <Stat label={TEXT.parent[lang]} value={parent.toFixed(3)} />
        <Stat label={TEXT.child[lang]} value={child.toFixed(3)} />
        <Stat label={TEXT.gain[lang]} value={(parent - child).toFixed(3)} tone="accent" />
      </div>
      <div className="w-row">
        <button className="btn btn-primary" onClick={goBest}>{TEXT.best[lang]}</button>
      </div>
    </Widget>
  );
}
