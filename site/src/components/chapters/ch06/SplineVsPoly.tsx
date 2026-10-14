import { useMemo, useState } from "react";
import { mse, polyFit, sample, truth } from "../../../engines/generalization";
import { fitSpline } from "../../../engines/nonparametric";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import Widget, { Slider, Stat } from "../../ui/Widget";

const TEXT = {
  title: { zh: "样条：分段的多项式，接缝处光滑", en: "Splines: piecewise polynomials, smooth at the seams" },
  K: { zh: "节点数 K", en: "Knots K" },
  df: { zh: "自由度 K + 4", en: "Degrees of freedom K + 4" },
  sp: { zh: "样条测试 MSE", en: "Spline test MSE" },
  po: { zh: "同自由度多项式测试 MSE", en: "Same-df polynomial test MSE" },
  caption: {
    zh: "第 02 章的 30 位学生：学习时长与成绩。黄线是三次回归样条：在 K 个节点（竖虚线）之间各是一段三次多项式，节点处函数值、一阶和二阶导数都连续。蓝色虚线是自由度相同的单个多项式（次数 K + 3）。增加自由度时，多项式在两端剧烈摆动（龙格现象），样条只在局部调整——每个基函数 (x − ξ)³₊ 只影响节点右边。灰色虚线是真实曲线。数据为合成示意。",
    en: "Chapter 02's 30 students: study hours and grades. The yellow curve is a cubic regression spline: a cubic between consecutive knots (dashed verticals), with the value and the first and second derivatives continuous at every knot. The blue dashed curve is a single polynomial with the same degrees of freedom (degree K + 3). As the degrees of freedom grow, the polynomial swings wildly at the ends (Runge's phenomenon) while the spline adjusts locally: each basis function (x − ξ)³₊ only acts to the right of its knot. The grey dashed curve is the truth. Synthetic illustrative data.",
  },
};

export default function SplineVsPoly({ lang }: { lang: Lang }) {
  const data = useMemo(() => sample(30, 2), []);
  const test = useMemo(() => sample(400, 9), []);
  const [K, setK] = useState(4);
  const sp = useMemo(() => fitSpline(data.xs, data.ys, K), [data, K]);
  const po = useMemo(() => polyFit(data.xs, data.ys, K + 3, 0), [data, K]);
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <CurvePlot xs={data.xs} ys={data.ys} xDomain={[0, 10]} yDomain={[0, 110]} height={300}
        curves={[
          { f: truth, color: "var(--text-muted)", dashed: true, width: 1.5 },
          { f: po, color: "var(--accent-2)", dashed: true },
          { f: sp.f, color: "var(--accent)" },
        ]}
        vlines={sp.knots.map((x) => ({ x, color: "var(--border)" }))} />
      <Slider label={TEXT.K[lang]} value={K} min={0} max={12} step={1} onChange={setK} />
      <div className="w-stats">
        <Stat label={TEXT.df[lang]} value={sp.df} tone="accent" />
        <Stat label={TEXT.sp[lang]} value={mse(sp.f, test.xs, test.ys).toFixed(0)} tone="good" />
        <Stat label={TEXT.po[lang]} value={Math.min(9999, mse(po, test.xs, test.ys)).toFixed(0)} />
      </div>
    </Widget>
  );
}
