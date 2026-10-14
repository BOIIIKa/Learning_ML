import { useMemo, useState } from "react";
import { buildKd, kdNodes, searchKd } from "../../../engines/nonparametric";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch06.css";

const TEXT = {
  title: { zh: "kd 树：先切分，再回溯", en: "The kd-tree: split first, then backtrack" },
  depth: { zh: "构造：切分到第几层", en: "Build: split down to depth" },
  step: { zh: "搜索：第几步", en: "Search: step" },
  visited: { zh: "访问的结点", en: "Nodes visited" },
  pruned: { zh: "剪掉的子树", en: "Subtrees pruned" },
  best: { zh: "当前最近距离", en: "Best distance so far" },
  query: { zh: "查询点", en: "Query" },
  caption: {
    zh: "六个点（讲义例题）。构造：根结点按 x 坐标的中位数把平面切成两半，下一层按 y 切，再按 x……每个结点存一个点、管一个矩形。搜索：先像二叉树一样往下走到查询点（黄色）所在的叶子，得到一个候选；回溯时，以查询点为圆心、当前最近距离为半径画圆——只有圆越过某条切分线，那条线另一侧的子树才可能有更近的点，否则整棵子树被剪掉（灰色）。",
    en: "Six points (the worked example of the notes). Building: the root splits the plane at the median x, the next level at the median y, then x again… Each node stores one point and owns one rectangle. Searching: walk down like a binary tree to the leaf containing the query (yellow) for a first candidate; while backtracking, draw a circle around the query with the best distance so far as radius. Only if the circle crosses a splitting line can the subtree on the other side hold a closer point; otherwise the whole subtree is pruned (grey).",
  },
};

const P = [[1, 6], [3, 2], [4, 7], [6, 3], [7, 8], [9, 5]];
const QUERIES = [[2, 6.5], [8, 6], [5, 1]];
const S = 26;
const f1 = (v: number) => v.toFixed(1);
const sx = (x: number) => 20 + x * S;
const sy = (y: number) => 280 - y * S;

export default function KdTreeViz({ lang }: { lang: Lang }) {
  const root = useMemo(() => buildKd(P, [0, 10, 0, 10]), []);
  const nodes = useMemo(() => kdNodes(root), [root]);
  const [depth, setDepth] = useState(3);
  const [qi, setQi] = useState(0);
  const q = QUERIES[qi];
  const { events } = useMemo(() => searchKd(root, P, q), [root, q]);
  const [stepRaw, setStep] = useState(events.length);
  const step = Math.min(stepRaw, events.length);
  const shown = events.slice(0, step);
  const last = shown.at(-1);
  const visited = new Set(shown.filter((e) => e.kind === "visit").map((e) => e.i));
  const prunedRoots = shown.filter((e) => e.kind === "prune").map((e) => e.i);
  const prunedBoxes = nodes.filter((n) => prunedRoots.includes(n.i)).map((n) => n.box);
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="chips">
        <span className="kd-lab">{TEXT.query[lang]}:</span>
        {QUERIES.map((p, i) => (
          <button key={i} className={`chip${qi === i ? " on" : ""}`} onClick={() => { setQi(i); setStep(0); }}>({p[0]}, {p[1]})</button>
        ))}
      </div>
      <svg className="lm-svg kd-svg" viewBox="0 0 300 300" role="img">
        <rect x={sx(0)} y={sy(10)} width={10 * S} height={10 * S} className="kd-frame" />
        {prunedBoxes.map((b, i) => <rect key={i} x={f1(sx(b[0]))} y={f1(sy(b[3]))} width={f1((b[1] - b[0]) * S)} height={f1((b[3] - b[2]) * S)} className="kd-pruned" />)}
        {nodes.filter((n) => n.depth < depth).map((n) => {
          const v = P[n.i][n.axis];
          const [x0, x1, y0, y1] = n.box;
          return n.axis === 0
            ? <line key={n.i} x1={f1(sx(v))} x2={f1(sx(v))} y1={f1(sy(y0))} y2={f1(sy(y1))} className={`kd-split d${n.depth}`} />
            : <line key={n.i} x1={f1(sx(x0))} x2={f1(sx(x1))} y1={f1(sy(v))} y2={f1(sy(v))} className={`kd-split d${n.depth}`} />;
        })}
        {last && <circle cx={f1(sx(q[0]))} cy={f1(sy(q[1]))} r={f1(last.bestDist * S)} className="kd-ball" />}
        {P.map((p, i) => (
          <g key={i}>
            <circle cx={f1(sx(p[0]))} cy={f1(sy(p[1]))} r={5} className={`kd-pt${visited.has(i) ? " visited" : ""}${last?.best === i ? " best" : ""}`} />
            <text x={f1(sx(p[0]) + 7)} y={f1(sy(p[1]) - 6)} className="kd-txt">({p[0]},{p[1]})</text>
          </g>
        ))}
        <circle cx={f1(sx(q[0]))} cy={f1(sy(q[1]))} r={5} className="kd-q" />
      </svg>
      <Slider label={TEXT.depth[lang]} value={depth} min={0} max={3} step={1} onChange={setDepth} />
      <Slider label={TEXT.step[lang]} value={step} min={0} max={events.length} step={1} onChange={setStep} format={(v) => `${v} / ${events.length}`} />
      <div className="w-stats">
        <Stat label={TEXT.visited[lang]} value={`${visited.size} / ${P.length}`} tone="accent" />
        <Stat label={TEXT.pruned[lang]} value={prunedRoots.length} />
        <Stat label={TEXT.best[lang]} value={last ? last.bestDist.toFixed(3) : "—"} />
      </div>
    </Widget>
  );
}
