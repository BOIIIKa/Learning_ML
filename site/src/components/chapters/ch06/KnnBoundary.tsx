import { scaleLinear } from "d3-scale";
import { useId, useMemo, useRef, useState } from "react";
import { gridFor, moons, r6 } from "../../../engines/data";
import { kNearest, knnError, knnScore, type Metric } from "../../../engines/nonparametric";
import type { Lang } from "../../../i18n/ui";
import Plot2D from "../../viz/Plot2D";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch06.css";

const TEXT = {
  title: { zh: "k 值、距离与决策边界", en: "k, the distance and the decision boundary" },
  k: { zh: "邻居数 k", en: "Neighbours k" },
  train: { zh: "训练误差", en: "Training error" },
  test: { zh: "测试误差", en: "Test error" },
  vote: { zh: "查询点的投票（迟到 : 准时）", en: "Query's vote (late : on time)" },
  df: { zh: "有效参数个数 ≈ n/k", en: "Effective parameters ≈ n/k" },
  caption: {
    zh: "地图上是 120 个历史订单（蓝 = 准时，橙 = 迟到），背景是 kNN 对每个位置的判断。点击图中任意位置放一个新地址：虚线把它连到 k 个邻居，外框是包住这些邻居的“球”——L1 下是菱形，L2 下是圆，L∞ 下是正方形。k = 1 时边界贴着每个点打转，训练误差为 0、测试误差却高；k 增大，边界变平滑；k 太大，边界又被抹平到欠拟合。",
    en: "The map shows 120 past orders (blue = on time, orange = late); the background is kNN's verdict at every location. Click anywhere to drop a new address: dashed lines join it to its k neighbours, and the outline is the “ball” that holds them — a diamond under L1, a circle under L2, a square under L∞. With k = 1 the boundary wraps around every point: zero training error but high test error. Larger k smooths it; too large a k flattens it into underfitting.",
  },
};

const METRICS: { m: Metric; label: string }[] = [
  { m: "l1", label: "L1" },
  { m: "l2", label: "L2" },
  { m: "linf", label: "L∞" },
];
const W = 460;
const H = 380;

export default function KnnBoundary({ lang }: { lang: Lang }) {
  const [ki, setKi] = useState(3);
  const [metric, setMetric] = useState<Metric>("l2");
  const [q, setQ] = useState<[number, number]>([0.45, 0.2]);
  const overlay = useRef<SVGSVGElement>(null);
  const clip = `knn-clip-${useId().replace(/:/g, "")}`;
  const k = 2 * ki + 1;

  const train = useMemo(() => moons(120, 0, 0.3), []);
  const test = useMemo(() => moons(300, 99, 0.3), []);
  const grid = useMemo(() => gridFor(train.X, 40, 0.4), [train]);
  const field = useMemo(() => grid.points.map((p) => r6(knnScore(train.X, train.y, p, k, metric))), [grid, train, k, metric]);
  const trainErr = useMemo(() => knnError(train.X, train.y, train.X, train.y, k, metric), [train, k, metric]);
  const testErr = useMemo(() => knnError(train.X, train.y, test.X, test.y, k, metric), [train, test, k, metric]);

  const sx = scaleLinear().domain([grid.x[0], grid.x[grid.x.length - 1]]).range([0, W]);
  const sy = scaleLinear().domain([grid.y[0], grid.y[grid.y.length - 1]]).range([H, 0]);
  const nb = kNearest(train.X, q, k, metric);
  const r = nb.d[nb.d.length - 1];
  const late = nb.idx.filter((i) => train.y[i] > 0).length;
  const f1 = (v: number) => v.toFixed(1);
  const [qx, qy] = [sx(q[0]), sy(q[1])];
  const rx = Math.abs(sx(r) - sx(0));
  const ry = Math.abs(sy(r) - sy(0));
  const ball =
    metric === "l2" ? <ellipse cx={f1(qx)} cy={f1(qy)} rx={f1(rx)} ry={f1(ry)} className="knn-ball" />
    : metric === "linf" ? <rect x={f1(qx - rx)} y={f1(qy - ry)} width={f1(2 * rx)} height={f1(2 * ry)} className="knn-ball" />
    : <polygon points={`${f1(qx)},${f1(qy - ry)} ${f1(qx + rx)},${f1(qy)} ${f1(qx)},${f1(qy + ry)} ${f1(qx - rx)},${f1(qy)}`} className="knn-ball" />;

  const place = (e: React.PointerEvent) => {
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(overlay.current!.getScreenCTM()!.inverse());
    setQ([r6(sx.invert(p.x)), r6(sy.invert(p.y))]);
  };

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="chips">
        {METRICS.map(({ m, label }) => (
          <button key={m} className={`chip${m === metric ? " on" : ""}`} onClick={() => setMetric(m)}>{label}</button>
        ))}
      </div>
      <div className="ch06-stage">
        <Plot2D X={train.X} y={train.y} step={{ field }} grid={{ x: grid.x, y: grid.y }} height={H} tweenMs={300} />
        <svg ref={overlay} className="ch06-overlay" viewBox={`0 0 ${W} ${H}`} onPointerDown={place} role="img">
          <defs><clipPath id={clip}><rect width={W} height={H} /></clipPath></defs>
          <g clipPath={`url(#${clip})`}>
            {ball}
            {nb.idx.map((i) => (
              <line key={i} x1={f1(qx)} y1={f1(qy)} x2={f1(sx(train.X[i][0]))} y2={f1(sy(train.X[i][1]))} className={`knn-link ${train.y[i] > 0 ? "pos" : "neg"}`} />
            ))}
            <circle cx={f1(qx)} cy={f1(qy)} r={7} className="knn-query" />
          </g>
        </svg>
      </div>
      <Slider label={TEXT.k[lang]} value={ki} min={0} max={30} step={1} onChange={setKi} format={() => `k = ${k}`} />
      <div className="w-stats">
        <Stat label={TEXT.train[lang]} value={`${(100 * trainErr).toFixed(1)}%`} />
        <Stat label={TEXT.test[lang]} value={`${(100 * testErr).toFixed(1)}%`} tone="accent" />
        <Stat label={TEXT.vote[lang]} value={`${late} : ${k - late}`} />
        <Stat label={TEXT.df[lang]} value={(120 / k).toFixed(1)} />
      </div>
    </Widget>
  );
}
