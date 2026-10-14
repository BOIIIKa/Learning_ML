import { useMemo, useState } from "react";
import { r6 } from "../../../engines/data";
import { distancesInCube, edgeFor, minMaxRatio, shellFraction } from "../../../engines/nonparametric";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch06.css";

const TEXT = {
  title: { zh: "维数灾难：高维里“近”与“远”分不开", en: "The curse of dimensionality: near and far merge in high dimensions" },
  d: { zh: "维度 d", en: "Dimension d" },
  ratio: { zh: "最近 / 最远距离", en: "Nearest / farthest distance" },
  shell: { zh: "单位球外层 10% 壳的体积占比", en: "Volume share of the ball's outer 10% shell" },
  edge: { zh: "装下 1% 数据的子立方体边长", en: "Edge of a sub-cube holding 1% of the data" },
  hist: { zh: "200 个点到查询点的距离 ÷ 最远距离", en: "Distances from 200 points to the query ÷ the farthest one" },
  xl: { zh: "log₁₀ d", en: "log₁₀ d" },
  caption: {
    zh: "在 [0,1]ᵈ 里均匀撒 200 个点，再随机放一个查询点。上图是所有距离除以最远距离后的直方图：d = 1 时有的点很近、有的很远；d 增大，所有距离都挤到最远距离附近。下图紫线是“最近 / 最远”之比（四个查询点平均），它随 d 爬向 1——“最近邻”不再比随便一个点近多少；橙色虚线是外层 10% 球壳所占的体积比例 1 − 0.9ᵈ，高维球的体积几乎全在表皮上。",
    en: "Scatter 200 points uniformly in [0,1]ᵈ and add a random query. Top: the histogram of all distances divided by the farthest one. At d = 1 some points are close and some far; as d grows every distance crowds next to the farthest. Bottom: the purple curve is the nearest/farthest ratio (averaged over four queries), climbing towards 1 — the “nearest” neighbour is hardly nearer than any other point. The dashed orange curve is the share of volume in the outer 10% shell, 1 − 0.9ᵈ: a high-dimensional ball is nearly all skin.",
  },
};

const DS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000];
const HW = 460;
const HH = 150;
const BINS = 25;

export default function CurseOfDim({ lang }: { lang: Lang }) {
  const [i, setI] = useState(4);
  const d = DS[i];
  const curve = useMemo(() => DS.map((dd) => [Math.log10(dd), minMaxRatio(dd)] as [number, number]), []);
  const ds = useMemo(() => distancesInCube(d, 200, 1), [d]);
  const max = Math.max(...ds);
  const min = Math.min(...ds);
  const counts = new Array(BINS).fill(0);
  for (const v of ds) counts[Math.min(BINS - 1, Math.floor((v / max) * BINS))]++;
  const top = Math.max(...counts);
  const bw = (HW - 20) / BINS;
  const interp = (lx: number) => {
    if (lx <= curve[0][0]) return curve[0][1];
    for (let j = 1; j < curve.length; j++) {
      if (lx <= curve[j][0]) {
        const [x0, y0] = curve[j - 1];
        const [x1, y1] = curve[j];
        return y0 + ((y1 - y0) * (lx - x0)) / (x1 - x0);
      }
    }
    return curve[curve.length - 1][1];
  };

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <figure className="cod-hist">
        <svg viewBox={`0 0 ${HW} ${HH}`} role="img">
          <rect width={HW} height={HH} rx={10} className="plot-bg" />
          {counts.map((c, j) => {
            const h = r6(((HH - 34) * c) / top);
            return <rect key={j} x={r6(10 + j * bw + 1)} y={r6(HH - 22 - h)} width={r6(bw - 2)} height={h} rx={2} className="cod-bar" />;
          })}
          <line x1={r6(10 + (min / max) * (HW - 20))} x2={r6(10 + (min / max) * (HW - 20))} y1={8} y2={HH - 22} className="cod-min" />
          <text x={r6(Math.min(HW - 70, 14 + (min / max) * (HW - 20)))} y={20} className="cod-lab">min/max</text>
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <text key={t} x={r6(10 + t * (HW - 20))} y={HH - 7} textAnchor={t === 0 ? "start" : t === 1 ? "end" : "middle"} className="cp-tick">{t}</text>
          ))}
        </svg>
        <figcaption>{TEXT.hist[lang]}</figcaption>
      </figure>
      <CurvePlot xs={[]} ys={[]} xDomain={[0, 3]} yDomain={[0, 1]} xLabel={TEXT.xl[lang]} height={220}
        curves={[
          { f: interp, color: "var(--accent-3)", width: 3 },
          { f: (lx) => shellFraction(10 ** lx, 0.1), color: "var(--class-pos)", width: 2, dashed: true },
        ]}
        vlines={[{ x: Math.log10(d), color: "var(--accent)", label: `d = ${d}` }]} />
      <Slider label={TEXT.d[lang]} value={i} min={0} max={DS.length - 1} step={1} onChange={setI} format={() => String(d)} />
      <div className="w-stats">
        <Stat label={TEXT.ratio[lang]} value={(min / max).toFixed(3)} tone="accent" />
        <Stat label={TEXT.shell[lang]} value={`${(100 * shellFraction(d, 0.1)).toFixed(1)}%`} />
        <Stat label={TEXT.edge[lang]} value={edgeFor(0.01, d).toFixed(3)} />
      </div>
    </Widget>
  );
}
