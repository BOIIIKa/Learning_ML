import { scaleLinear } from "d3-scale";
import { useMemo, useState } from "react";
import { ES, alphaFor, gdAt, gdFilter, ridgeAt, ridgeFilter } from "../../../engines/dl-generalization";
import { detCos } from "../../../engines/optimization";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch18.css";

const TEXT = {
  title: { zh: "早停 ≈ 岭回归", en: "Early stopping ≈ ridge" },
  t: { zh: "梯度下降步数 t（对数刻度）", en: "Gradient-descent steps t (log scale)" },
  plane: { zh: "参数平面：梯度下降路径（实线）与岭回归路径（虚线）", en: "Parameter plane: the GD path (solid) and the ridge path (dashed)" },
  filt: { zh: "沿特征值 λ 方向恢复了多少：横轴 log₁₀ λ", en: "How much is recovered along eigenvalue λ: log₁₀ λ across" },
  alpha: { zh: "对应的岭惩罚 α = 1/(ηt)", en: "Matching ridge penalty α = 1/(ηt)" },
  small: { zh: "小特征值方向（λ = 0.04）：GD / 岭", en: "Small direction (λ = 0.04): GD / ridge" },
  gap: { zh: "实心点与空心点的距离 / ‖w*‖", en: "Distance between the two points / ‖w*‖" },
  caption: {
    zh: "二维最小二乘：损失的等高线是椭圆，两个特征值 1 和 0.04，学习率 η = 0.5，从原点出发。梯度下降先沿陡的方向走到底，再沿平的方向慢慢爬（实线）；岭回归的解随 α 从大到小也画出一条路径（虚线）。两条路径几乎重合（最远处相距不到 ‖w*‖ 的 1/10）。滑动 t：第 t 步的梯度下降点（实心）对应 α = 1/(ηt) 的岭解（空心），这个对应是近似的，两点最多相差 ‖w*‖ 的四分之一。右图说明原因：沿特征值 λ 的方向，梯度下降恢复了 1 − (1 − ηλ)ᵗ，岭回归恢复了 λ/(λ + α)。λ 远大于 α 时两者都接近 1，远小于 α 时都约等于 λ/α，两种做法都是把小特征值方向压掉。t = 50（α = 0.04）时，小特征值方向上分别是 0.636 和 0.5。",
    en: "Two-dimensional least squares: the loss contours are ellipses with eigenvalues 1 and 0.04, learning rate η = 0.5, starting at the origin. Gradient descent first runs down the steep direction, then creeps along the flat one (solid); ridge regression's solution traces its own path as α goes from large to small (dashed). The two paths nearly coincide (never more than a tenth of ‖w*‖ apart). Slide t: the gradient-descent point at step t (filled) is matched with the ridge solution for α = 1/(ηt) (hollow); the match is approximate, and the two points are at most a quarter of ‖w*‖ apart. The right plot says why: along an eigenvalue λ, gradient descent has recovered 1 − (1 − ηλ)ᵗ and ridge λ/(λ + α). For λ far above α both are near 1, far below it both are about λ/α: both methods suppress the small-eigenvalue directions. At t = 50 (α = 0.04) the small direction gets 0.636 and 0.5.",
  },
};

const W = 300, H = 260;
const f1 = (v: number) => v.toFixed(1);
const pow10 = (u: number) => Math.exp(u * Math.LN10);
const LOG_L2 = -1.39794; // log₁₀ 0.04

export default function EarlyStopRidge({ lang }: { lang: Lang }) {
  const [lt, setLt] = useState(1.7);
  const t = Math.round(pow10(lt));
  const alpha = alphaFor(t);
  const sx = scaleLinear().domain([-0.4, 2.4]).range([24, W - 8]);
  const sy = scaleLinear().domain([-0.3, 3.1]).range([H - 20, 8]);
  const pt = ([x, y]: number[]) => `${f1(sx(x))},${f1(sy(y))}`;
  const ellipses = useMemo(() => {
    const { Q, wStar, lam } = ES;
    return [0.005, 0.02, 0.045, 0.08].map((L) => {
      const a = Math.sqrt((2 * L) / lam[0]), b = Math.sqrt((2 * L) / lam[1]);
      return Array.from({ length: 73 }, (_, i) => {
        const th = (2 * Math.PI * i) / 72, c = detCos(th), s = detCos(th - Math.PI / 2);
        return [wStar[0] + Q[0][0] * a * c + Q[0][1] * b * s, wStar[1] + Q[1][0] * a * c + Q[1][1] * b * s];
      });
    });
  }, []);
  const gdPath = useMemo(() => Array.from({ length: 401 }, (_, i) => gdAt(i)), []);
  const ridgePath = useMemo(() => Array.from({ length: 121 }, (_, i) => ridgeAt(pow10(3 - i / 20))), []);
  const g = gdAt(t), r = ridgeAt(alpha);
  const gap = Math.hypot(g[0] - r[0], g[1] - r[1]) / Math.hypot(...ES.wStar);
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="dg-two">
        <div>
          <div className="dg-sub">{TEXT.plane[lang]}</div>
          <svg className="lm-svg" viewBox={`0 0 ${W} ${H}`} role="img">
            <line x1={f1(sx(0))} x2={f1(sx(0))} y1={8} y2={H - 20} className="lm-axis" />
            <line x1={24} x2={W - 8} y1={f1(sy(0))} y2={f1(sy(0))} className="lm-axis" />
            {ellipses.map((e, i) => <polyline key={i} points={e.map(pt).join(" ")} className="dg-contour" />)}
            <polyline points={ridgePath.map(pt).join(" ")} className="dg-ridge" />
            <polyline points={gdPath.slice(0, t + 1).map(pt).join(" ")} className="dg-gd" />
            <circle cx={f1(sx(ES.wStar[0]))} cy={f1(sy(ES.wStar[1]))} r={3.5} className="dg-star" />
            <text x={f1(sx(ES.wStar[0]) + 6)} y={f1(sy(ES.wStar[1]) - 6)} className="dg-label">w*</text>
            <circle cx={f1(sx(r[0]))} cy={f1(sy(r[1]))} r={6} className="dg-pt-ridge" />
            <circle cx={f1(sx(g[0]))} cy={f1(sy(g[1]))} r={4} className="dg-pt-gd" />
          </svg>
        </div>
        <div>
          <div className="dg-sub">{TEXT.filt[lang]}</div>
          <CurvePlot xs={[]} ys={[]} xDomain={[-3, 0.3]} yDomain={[0, 1.05]} height={240}
            vlines={[{ x: LOG_L2, color: "var(--text-muted)", label: "λ₂" }, { x: 0, color: "var(--text-muted)", label: "λ₁" }]}
            curves={[
              { f: (u) => gdFilter(pow10(u), ES.eta, t), color: "var(--accent)", width: 2.5 },
              { f: (u) => ridgeFilter(pow10(u), alpha), color: "var(--accent-2)", dashed: true, width: 2 },
            ]} />
        </div>
      </div>
      <Slider label={TEXT.t[lang]} value={lt} min={0} max={2.6} step={0.05} onChange={setLt} format={() => `t = ${t}`} />
      <div className="w-stats">
        <Stat label={TEXT.alpha[lang]} value={alpha.toPrecision(3)} tone="accent" />
        <Stat label={TEXT.small[lang]} value={`${gdFilter(ES.lam[1], ES.eta, t).toFixed(3)} / ${ridgeFilter(ES.lam[1], alpha).toFixed(3)}`} />
        <Stat label={TEXT.gap[lang]} value={gap.toFixed(3)} />
      </div>
    </Widget>
  );
}
