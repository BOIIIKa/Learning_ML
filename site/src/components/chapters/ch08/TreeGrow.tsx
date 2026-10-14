import { useMemo, useState } from "react";
import { gridFor, moons } from "../../../engines/data";
import { accuracy, growTree, leavesOf, predictProba, type Node } from "../../../engines/tree";
import type { Lang } from "../../../i18n/ui";
import MetricChart from "../../viz/MetricChart";
import Plot2D from "../../viz/Plot2D";
import Widget, { Slider, Stat } from "../../ui/Widget";

const TEXT = {
  title: { zh: "长一棵树：深度与过拟合", en: "Grow a tree: depth and overfitting" },
  depth: { zh: "最大深度", en: "Max depth" },
  leaves: { zh: "叶结点", en: "Leaves" },
  train: { zh: "训练准确率", en: "Training accuracy" },
  test: { zh: "测试准确率", en: "Test accuracy" },
  caption: {
    zh: "每加一层，平面就被继续切成更小的矩形（左），树也多长一层（右，只画前四层）。训练准确率一路上升，测试准确率却先升后降——这就是过拟合。",
    en: "Each extra level cuts the plane into smaller rectangles (left) and grows the tree by one level (right, first four levels drawn). Training accuracy keeps rising, but test accuracy rises then falls: overfitting.",
  },
};

const MAX = 10;

function TreeDiagram({ root, featureNames }: { root: Node; featureNames: string[] }) {
  // Lay out leaves left-to-right; internal nodes sit above the middle of their children.
  const DEPTH = 4;
  const nodes: { x: number; y: number; node: Node; parent?: { x: number; y: number }; cut: boolean }[] = [];
  let leafX = 0;
  const walk = (n: Node, d: number, parent?: { x: number; y: number }): number => {
    const cut = d === DEPTH && !n.leaf;
    let x: number;
    if (n.leaf || cut) x = leafX++;
    else {
      const l = walk(n.left, d + 1, undefined);
      const r = walk(n.right, d + 1, undefined);
      x = (l + r) / 2;
    }
    const entry = { x, y: d, node: n, parent, cut };
    nodes.push(entry);
    return x;
  };
  walk(root, 0);
  // Attach parents in a second pass (positions are known now).
  const pos = new Map(nodes.map((e) => [e.node, e]));
  for (const e of nodes) {
    const n = e.node;
    if (!n.leaf && !e.cut) for (const c of [n.left, n.right]) { const ce = pos.get(c); if (ce) ce.parent = { x: e.x, y: e.y }; }
  }
  const W = 460;
  const H = 300;
  const n = Math.max(1, leafX);
  const px = (x: number) => 20 + ((W - 40) * (x + 0.5)) / n;
  const py = (d: number) => 24 + (d * (H - 50)) / DEPTH;
  return (
    <svg className="tree-diagram" viewBox={`0 0 ${W} ${H}`} role="img">
      {nodes.map((e, i) => e.parent && (
        <line key={`l${i}`} x1={px(e.parent.x)} y1={py(e.parent.y)} x2={px(e.x)} y2={py(e.y)} className="td-edge" />
      ))}
      {nodes.map((e, i) => {
        const leafLike = e.node.leaf || e.cut;
        const cls = e.node.p >= 0.5 ? "pos" : "neg";
        return (
          <g key={`n${i}`} transform={`translate(${px(e.x).toFixed(1)},${py(e.y).toFixed(1)})`} className="td-node" style={{ animationDelay: `${e.y * 90}ms` }}>
            {leafLike ? (
              <rect x={-9} y={-9} width={18} height={18} rx={4} className={`td-leaf ${cls}`} style={{ opacity: 0.35 + 0.65 * Math.abs(e.node.p - 0.5) * 2 }} />
            ) : (
              <>
                <circle r={11} className="td-split" />
                {nodes.length < 24 && !e.node.leaf && (
                  <text y={-15} textAnchor="middle" className="td-label">{`${featureNames[e.node.feature]}≤${e.node.threshold.toFixed(2)}`}</text>
                )}
              </>
            )}
            {e.cut && <text y={22} textAnchor="middle" className="td-more">…</text>}
          </g>
        );
      })}
    </svg>
  );
}

export default function TreeGrow({ lang }: { lang: Lang }) {
  const train = useMemo(() => moons(100, 2, 0.4), []);
  const test = useMemo(() => moons(600, 99, 0.4), []);
  const grid = useMemo(() => gridFor(train.X, 40), [train]);
  const [depth, setDepth] = useState(2);

  const trees = useMemo(() => Array.from({ length: MAX }, (_, d) => growTree(train.X, train.y, undefined, { maxDepth: d + 1 })), [train]);
  const tree = trees[depth - 1];
  const field = useMemo(() => grid.points.map((p) => predictProba(tree, p) - 0.5), [tree, grid]);
  const trainAcc = trees.map((t) => accuracy(t, train.X, train.y));
  const testAcc = trees.map((t) => accuracy(t, test.X, test.y));

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="split-2">
        <Plot2D X={train.X} y={train.y} step={{ field }} grid={{ x: grid.x, y: grid.y }} tweenMs={350} />
        <TreeDiagram root={tree} featureNames={["x₁", "x₂"]} />
      </div>
      <Slider label={TEXT.depth[lang]} value={depth} min={1} max={MAX} step={1} onChange={setDepth} />
      <MetricChart
        zero={false}
        cursor={depth - 1}
        series={[
          { key: "train", label: TEXT.train[lang], values: trainAcc, color: "var(--accent)" },
          { key: "test", label: TEXT.test[lang], values: testAcc, color: "var(--accent-2)", dashed: true },
        ]}
      />
      <div className="w-stats">
        <Stat label={TEXT.leaves[lang]} value={leavesOf(tree)} />
        <Stat label={TEXT.train[lang]} value={`${(trainAcc[depth - 1] * 100).toFixed(1)}%`} />
        <Stat label={TEXT.test[lang]} value={`${(testAcc[depth - 1] * 100).toFixed(1)}%`} tone="accent" />
      </div>
    </Widget>
  );
}
