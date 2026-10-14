import { useState } from "react";
import { softAssign } from "../../../engines/em-mixtures";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import Widget, { Slider, Stat } from "../../ui/Widget";

const TEXT = {
  title: { zh: "从软分配到硬分配：k 均值是 σ → 0 的极限", en: "From soft to hard assignments: k-means is the σ → 0 limit" },
  s: { zh: "共享方差 σ²（log₁₀）", en: "Shared variance σ² (log₁₀)" },
  x: { zh: "指标 x", en: "marker x" },
  caption: {
    zh: "两个等权、等方差的高斯，中心在 0 和 2（讲义例题）。曲线是点 x 属于右边那个高斯的责任度 r(x)，四个点是例题的数据 −1、0、3、4。方差大时 r 是一条平缓的 S 形，离中点 1 较远的点也拿不到很高的把握（σ² = 10 时 r(4) 只有 0.65）；方差缩小，曲线变陡；σ² → 0 时 r 变成阶跃函数：每个点 100% 属于最近的中心——这正是 k 均值的分配步。",
    en: "Two Gaussians with equal weights and variances, centred at 0 and 2 (the worked example). The curve is the responsibility r(x) of the right-hand Gaussian for a point x; the four dots are the example's data −1, 0, 3, 4. With a large variance r is a gentle S, and even points well away from the midpoint 1 are far from certain (at σ² = 10, r(4) is only 0.65); as the variance shrinks the curve steepens, and as σ² → 0 r becomes a step: every point belongs 100% to its nearest centre, which is exactly k-means' assignment step.",
  },
};

const XS = [-1, 0, 3, 4];

export default function HardLimit({ lang }: { lang: Lang }) {
  const [ls, setLs] = useState(0);
  const s = 10 ** ls;
  const r = XS.map((x) => softAssign(x, 0, 2, s));
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <CurvePlot xs={XS} ys={r} xDomain={[-2, 5]} yDomain={[-0.05, 1.05]} height={240} xLabel={TEXT.x[lang]}
        curves={[{ f: (x) => softAssign(x, 0, 2, s), color: "var(--accent)" }]}
        vlines={[{ x: 0, color: "var(--accent-2)" }, { x: 2, color: "var(--class-pos)" }]} />
      <Slider label={TEXT.s[lang]} value={ls} min={-2} max={1} step={0.05} onChange={setLs} format={(v) => `σ² = ${(10 ** v).toPrecision(2)}`} />
      <div className="w-stats">
        {XS.map((x, i) => <Stat key={x} label={`r(${x})`} value={r[i].toFixed(3)} tone={i === 1 ? "accent" : undefined} />)}
      </div>
    </Widget>
  );
}
