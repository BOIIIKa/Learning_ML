import { useMemo, useState } from "react";
import { collinearDraws } from "../../../engines/linreg";
import type { Lang } from "../../../i18n/ui";
import Plot2D from "../../viz/Plot2D";
import Widget, { Slider, Stat } from "../../ui/Widget";

const TEXT = {
  title: { zh: "两个几乎一样的特征：系数在乱跳", en: "Two near-identical features: the coefficients jump around" },
  rho: { zh: "两特征的相关系数 ρ", en: "Correlation between the features ρ" },
  lam: { zh: "岭惩罚 λ", en: "Ridge penalty λ" },
  showRidge: { zh: "显示岭回归", en: "Show ridge" },
  sdOls: { zh: "OLS 的 w₁ 标准差", en: "SD of w₁, OLS" },
  sdRidge: { zh: "岭回归 w₁ 标准差", en: "SD of w₁, ridge" },
  caption: {
    zh: "横轴是“学习时长”的系数 w₁，纵轴是“图书馆时长”的系数 w₂，真实值都是 1（带圈的点）。每个点是用一批新的 25 位学生拟合的结果。两个特征越相关，最小二乘的估计越沿着 w₁ + w₂ = 2 的虚线（两者之和）散开：数据只能确定“和”，分不清“谁的功劳”。加上岭惩罚，点云立刻收拢，代价是向原点略微收缩。",
    en: "The horizontal axis is the coefficient w₁ of “hours studied”, the vertical axis the coefficient w₂ of “hours in the library”; both are truly 1 (ringed point). Each dot is a fit on a fresh set of 25 students. The more correlated the features, the more the least-squares estimates spread along the dashed line w₁ + w₂ = 2: the data pin down the sum but cannot tell whose credit it is. A ridge penalty pulls the cloud together, at the price of shrinking it slightly towards the origin.",
  },
};

const sd = (v: number[]) => { const m = v.reduce((a, b) => a + b, 0) / v.length; return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length); };

export default function CollinearCoefs({ lang }: { lang: Lang }) {
  const [rho, setRho] = useState(0.95);
  const [lam, setLam] = useState(0.3);
  const [show, setShow] = useState(true);
  const draws = useMemo(() => collinearDraws(rho, lam), [rho, lam]);
  const pts = show ? [...draws.ols, ...draws.ridge] : draws.ols;
  const y = pts.map((_, i) => (i < draws.ols.length ? -1 : 1));
  // The true coefficient (1, 1) goes last so it is drawn on top, highlighted with a ring.
  const X = [...pts, [1, 1]];
  const yAll = [...y, 1];
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="scenario">
      <Plot2D X={X} y={yAll} step={{ highlight: [X.length - 1] }} domain={{ x: [-2.5, 4.5], y: [-2.5, 4.5] }} height={400}
        extraLines={[
          { w: [1, 0], b: 0, color: "var(--border)" },
          { w: [0, 1], b: 0, color: "var(--border)" },
          { w: [1, 1], b: -2, color: "var(--ink)", dashed: true },
        ]} />
      <div className="w-row">
        <Slider label={TEXT.rho[lang]} value={rho} min={0} max={0.99} step={0.01} onChange={setRho} />
        <Slider label={TEXT.lam[lang]} value={lam} min={0} max={2} step={0.05} onChange={setLam} />
        <label className="chip"><input type="checkbox" checked={show} onChange={(e) => setShow(e.target.checked)} /> {TEXT.showRidge[lang]}</label>
      </div>
      <div className="w-stats">
        <Stat label={TEXT.sdOls[lang]} value={sd(draws.ols.map((w) => w[0])).toFixed(2)} />
        <Stat label={TEXT.sdRidge[lang]} value={sd(draws.ridge.map((w) => w[0])).toFixed(2)} tone="accent" />
      </div>
    </Widget>
  );
}
