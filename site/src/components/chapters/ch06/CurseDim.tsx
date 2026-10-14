import { useMemo, useState } from "react";
import { contrast, edgeFor, shellFraction } from "../../../engines/nonparametric";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import Widget, { Slider, Stat } from "../../ui/Widget";

const TEXT = {
  title: { zh: "维数灾难：高维空间里，人人都一样远", en: "The curse of dimensionality: in high dimensions everyone is equally far" },
  d: { zh: "维数 d", en: "Dimension d" },
  ratio: { zh: "最近 / 最远距离", en: "Nearest / farthest distance" },
  shell: { zh: "外层 5% 球壳的体积占比", en: "Volume in the outer 5% shell" },
  edge: { zh: "装下 10% 数据的小立方体边长", en: "Edge of a cube holding 10% of the data" },
  x: { zh: "维数 d", en: "dimension d" },
  caption: {
    zh: "单位立方体里均匀撒 400 个点，随机取查询点，算最近邻与最远邻距离之比（曲线）。d = 2 时最近的点近得多；d 上百时比值接近 1——“最近邻”和“最远邻”几乎一样远，kNN 的“局部”失去意义。另外两个数字说的是同一件事：高维球的体积几乎全在薄薄的外壳里；要装下 10% 的数据，小立方体的边长要接近整条边——“局部邻域”其实一点也不局部。",
    en: "400 points scattered uniformly in the unit cube; for random queries, the ratio of nearest to farthest distance (the curve). In d = 2 the nearest point is much closer; with d in the hundreds the ratio approaches 1. The nearest and farthest neighbours are almost equally far, and kNN's notion of local loses its meaning. The other two numbers say the same thing: almost all of a high-dimensional ball's volume sits in a thin outer shell, and a cube holding 10% of the data needs an edge almost as long as the whole cube. A local neighbourhood is not local at all.",
  },
};

const DS = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200];

export default function CurseDim({ lang }: { lang: Lang }) {
  const ratios = useMemo(() => DS.map((d) => contrast(d)), []);
  const [k, setK] = useState(1);
  const d = DS[k];
  const L = DS.map((v) => Math.log10(v));
  const f = (x: number) => {
    if (x <= L[0]) return ratios[0];
    for (let i = 1; i < L.length; i++) if (x <= L[i]) return ratios[i - 1] + ((ratios[i] - ratios[i - 1]) * (x - L[i - 1])) / (L[i] - L[i - 1]);
    return ratios.at(-1)!;
  };
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="frontier">
      <CurvePlot xs={L} ys={ratios} xDomain={[-0.05, 2.35]} yDomain={[0, 1]} height={250} xLabel={`log₁₀ ${TEXT.x[lang]}`}
        curves={[{ f, color: "var(--accent)" }]} vlines={[{ x: Math.log10(d), color: "var(--ink)" }]} />
      <Slider label={TEXT.d[lang]} value={k} min={0} max={DS.length - 1} step={1} onChange={setK} format={(v) => `${DS[v]}`} />
      <div className="w-stats">
        <Stat label={TEXT.ratio[lang]} value={ratios[k].toFixed(2)} tone="accent" />
        <Stat label={TEXT.shell[lang]} value={`${(100 * shellFraction(d, 0.05)).toFixed(1)}%`} />
        <Stat label={TEXT.edge[lang]} value={edgeFor(0.1, d).toFixed(2)} />
      </div>
    </Widget>
  );
}
