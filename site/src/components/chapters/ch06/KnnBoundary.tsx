import { useMemo, useState } from "react";
import { moons } from "../../../engines/data";
import { dist, type Metric } from "../../../engines/nonparametric";
import type { Lang } from "../../../i18n/ui";
import Plot2D from "../../viz/Plot2D";
import Widget, { Slider, Stat } from "../../ui/Widget";

const TEXT = {
  title: { zh: "k 近邻：问问最像的 k 位老病人", en: "k nearest neighbours: ask the k most similar past patients" },
  k: { zh: "邻居数 k", en: "Neighbours k" },
  train: { zh: "训练误差", en: "Training error" },
  test: { zh: "测试误差", en: "Test error" },
  caption: {
    zh: "150 位老病人（两类交错，呈月牙形）。新病人的类别由离他最近的 k 位老病人投票决定；色带是投票中“复查”所占比例。k = 1 时边界锯齿状、训练误差为 0，却记住了每一个噪声点；k 增大，边界变平滑；k 接近 n 时，所有人都按多数类处理。测试误差（600 位新病人）在中间最低：k 就是 kNN 的复杂度旋钮，k 越小模型越复杂。切换距离：L1 的“球”是菱形，L∞ 是正方形，边界会随之改变。数据为合成示意。",
    en: "150 past patients (two interleaving half-moons). A new patient's class is the vote of the k nearest past patients; the shading is the share of follow-up votes. With k = 1 the boundary is jagged and the training error is 0, but every noisy point has been memorized; as k grows the boundary smooths out; near k = n everyone gets the majority class. The test error (600 new patients) is lowest in between: k is kNN's complexity knob, and smaller k means a more complex model. Switch the distance: the L1 ball is a diamond and the L∞ ball a square, and the boundary changes with them. Synthetic illustrative data.",
  },
};

const METRICS: { id: Metric; label: string }[] = [{ id: "l2", label: "L2" }, { id: "l1", label: "L1" }, { id: "linf", label: "L∞" }];

/** For each query: labels of the training points sorted by distance, as prefix sums (so any k is O(1)). */
function prefixVotes(X: number[][], y: number[], Q: number[][], m: Metric): Float64Array[] {
  return Q.map((q) => {
    const order = X.map((x, i) => [dist(x, q, m), i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const c = new Float64Array(order.length + 1);
    order.forEach(([, i], j) => { c[j + 1] = c[j] + y[i]; });
    return c;
  });
}

export default function KnnBoundary({ lang }: { lang: Lang }) {
  const { X, y } = useMemo(() => moons(150, 1, 0.3), []);
  const T = useMemo(() => moons(600, 2, 0.3), []);
  const grid = useMemo(() => {
    const n = 40;
    const lin = (lo: number, hi: number) => Array.from({ length: n }, (_, i) => +(lo + ((hi - lo) * i) / (n - 1)).toFixed(4));
    return { x: lin(-1.9, 2.9), y: lin(-1.4, 1.9) };
  }, []);
  const gpts = useMemo(() => grid.y.flatMap((gy) => grid.x.map((gx) => [gx, gy])), [grid]);
  const [mi, setMi] = useState(0);
  const [k, setK] = useState(1);
  const m = METRICS[mi].id;
  const pre = useMemo(() => ({ g: prefixVotes(X, y, gpts, m), tr: prefixVotes(X, y, X, m), te: prefixVotes(X, y, T.X, m) }), [X, y, gpts, T, m]);
  const field = pre.g.map((c) => +(c[k] / k).toFixed(3));
  const err = (P: Float64Array[], labels: number[]) => P.reduce((s, c, i) => s + ((c[k] / k) * labels[i] > 0 ? 0 : 1), 0) / P.length;
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="scenario">
      <div className="chips">
        {METRICS.map((t, i) => <button key={t.id} className={`chip${mi === i ? " on" : ""}`} onClick={() => setMi(i)}>{t.label}</button>)}
      </div>
      <Plot2D X={X} y={y} grid={grid} step={{ field }} levels={[1e-9, 0.34, 0.67]} showMargin={false} height={320} tweenMs={0} />
      <Slider label={TEXT.k[lang]} value={k} min={1} max={149} step={2} onChange={setK} />
      <div className="w-stats">
        <Stat label="k" value={k} tone="accent" />
        <Stat label={TEXT.train[lang]} value={`${(100 * err(pre.tr, y)).toFixed(1)}%`} />
        <Stat label={TEXT.test[lang]} value={`${(100 * err(pre.te, T.y)).toFixed(1)}%`} tone="good" />
      </div>
    </Widget>
  );
}
