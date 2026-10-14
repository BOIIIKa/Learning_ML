import { useRef, useState } from "react";
import { dot3, project3, type Vec3 } from "../../../engines/linreg";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch03.css";

const TEXT = {
  title: { zh: "最小二乘就是投影", en: "Least squares is a projection" },
  y: { zh: "第 {i} 位学生的成绩 y{i}", en: "Score of student {i}, y{i}" },
  coef: { zh: "拟合：截距 b，斜率 w", en: "Fit: intercept b, slope w" },
  dots: { zh: "残差 · 两列", en: "Residual · both columns" },
  rss: { zh: "残差平方和 ‖y − ŷ‖²", en: "Residual sum of squares ‖y − ŷ‖²" },
  drag: { zh: "← 拖动画面旋转 →", en: "← drag to rotate →" },
  caption: {
    zh: "只有 3 位学生，所以 y 是三维空间里的一个向量（橙色），每个坐标轴对应一位学生。设计矩阵的两列——全 1 列（截距）和学习时长列 x = (−1, 0, 1)——张成一个平面（蓝色），平面上每一点都是某条直线 b + wx 给出的一组预测。最小二乘找的是平面上离 y 最近的点 ŷ：残差（虚线）与平面垂直，所以它与两列的内积都为 0——这正是正规方程 Xᵀ(y − Xβ) = 0。",
    en: "With only 3 students, y is a vector in 3-D space (orange), one axis per student. The two columns of the design matrix, the all-ones column (intercept) and the study-hours column x = (−1, 0, 1), span a plane (blue), and every point in it is the set of predictions of some line b + wx. Least squares finds the point ŷ in the plane closest to y: the residual (dashed) is perpendicular to the plane, so its inner product with both columns is 0. That is exactly the normal equations Xᵀ(y − Xβ) = 0.",
  },
};

const W = 460;
const H = 380;
const S = 56;
const EL = 0.42;
const f1 = (v: number) => v.toFixed(1);
const A1: Vec3 = [1, 1, 1];
const A2: Vec3 = [-1, 0, 1];

export default function ProjectionViz({ lang }: { lang: Lang }) {
  const [y, setY] = useState<Vec3>([0.6, 2.4, 1.4]);
  const [az, setAz] = useState(0.75);
  const drag = useRef<number | null>(null);
  const { coef, yhat } = project3(A1, A2, y);
  const r = [0, 1, 2].map((i) => y[i] - yhat[i]) as Vec3;

  const P = (p: Vec3) => {
    const xr = p[0] * Math.cos(az) - p[1] * Math.sin(az);
    const yr = p[0] * Math.sin(az) + p[1] * Math.cos(az);
    return [W / 2 + S * xr, H * 0.72 - S * (p[2] * Math.cos(EL) + yr * Math.sin(EL))] as const;
  };
  const pt = (p: Vec3) => { const [a, b] = P(p); return `${f1(a)},${f1(b)}`; };
  const lin = (s: number, t: number) => [0, 1, 2].map((i) => s * A1[i] + t * A2[i]) as Vec3;
  const plane = [lin(-0.2, -1.2), lin(2.4, -1.2), lin(2.4, 1.2), lin(-0.2, 1.2)];
  const gridS = [0.4, 1, 1.6].map((s) => `M${pt(lin(s, -1.2))}L${pt(lin(s, 1.2))}`).join("");
  const gridT = [-0.6, 0, 0.6].map((t) => `M${pt(lin(-0.2, t))}L${pt(lin(2.4, t))}`).join("");

  // Right-angle marker at ŷ, along the residual and an in-plane direction.
  const nr = Math.hypot(...r);
  const u = nr > 1e-6 ? (r.map((v) => (v / nr) * 0.28) as Vec3) : ([0, 0, 0] as Vec3);
  const vv = A2.map((v) => (v / Math.SQRT2) * 0.28) as Vec3;
  const add = (...ps: Vec3[]) => [0, 1, 2].map((i) => ps.reduce((s, p) => s + p[i], 0)) as Vec3;
  const marker = `M${pt(add(yhat, u))}L${pt(add(yhat, u, vv))}L${pt(add(yhat, vv))}`;

  const axis = (i: number) => { const e: Vec3 = [0, 0, 0]; e[i] = 2.8; return e; };
  const O: Vec3 = [0, 0, 0];

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <svg className="lr-svg" viewBox={`0 0 ${W} ${H}`} role="img"
        onPointerDown={(e) => { drag.current = e.clientX; (e.target as Element).setPointerCapture?.(e.pointerId); }}
        onPointerMove={(e) => { if (drag.current !== null) { setAz((a) => a + (e.clientX - drag.current!) * 0.01); drag.current = e.clientX; } }}
        onPointerUp={() => (drag.current = null)} onPointerLeave={() => (drag.current = null)}>
        <rect width={W} height={H} className="plot-bg" />
        {[0, 1, 2].map((i) => (
          <g key={i}>
            <path d={`M${pt(O)}L${pt(axis(i))}`} className="lr-axis" />
            <text x={f1(P(axis(i))[0] + 4)} y={f1(P(axis(i))[1] - 4)} className="lr-label">{`y${i + 1}`}</text>
          </g>
        ))}
        <path d={`M${plane.map(pt).join("L")}Z`} className="lr-plane" />
        <path d={gridS + gridT} className="lr-plane-grid" />
        <path d={`M${pt(O)}L${pt(A1)}`} className="lr-col" />
        <text x={f1(P(A1)[0] + 6)} y={f1(P(A1)[1])} className="lr-label">1</text>
        <path d={`M${pt(O)}L${pt(A2)}`} className="lr-col" />
        <text x={f1(P(A2)[0] + 6)} y={f1(P(A2)[1])} className="lr-label">x</text>
        <path d={`M${pt(O)}L${pt(yhat)}`} className="lr-yhat" />
        <path d={`M${pt(yhat)}L${pt(y)}`} className="lr-resid" />
        <path d={marker} className="lr-right" />
        <path d={`M${pt(O)}L${pt(y)}`} className="lr-y" />
        <circle cx={f1(P(y)[0])} cy={f1(P(y)[1])} r={5} className="lr-y-dot" />
        <circle cx={f1(P(yhat)[0])} cy={f1(P(yhat)[1])} r={4.5} className="lr-yhat-dot" />
        <text x={f1(P(y)[0] + 8)} y={f1(P(y)[1] - 6)} className="lr-label strong">y</text>
        <text x={f1(P(yhat)[0] + 8)} y={f1(P(yhat)[1] + 14)} className="lr-label strong">ŷ</text>
        <text x={W / 2} y={H - 10} textAnchor="middle" className="lr-hint">{TEXT.drag[lang]}</text>
      </svg>
      <div className="w-row">
        {[0, 1, 2].map((i) => (
          <Slider key={i} label={TEXT.y[lang].replaceAll("{i}", String(i + 1))} value={y[i]} min={-1} max={3} step={0.05}
            onChange={(v) => setY((cur) => cur.map((c, j) => (j === i ? v : c)) as Vec3)} />
        ))}
      </div>
      <div className="w-stats">
        <Stat label={TEXT.coef[lang]} value={`b = ${coef[0].toFixed(2)}, w = ${coef[1].toFixed(2)}`} />
        <Stat label={TEXT.dots[lang]} value={`${Math.abs(dot3(r, A1)).toFixed(3)}, ${Math.abs(dot3(r, A2)).toFixed(3)}`} tone="good" />
        <Stat label={TEXT.rss[lang]} value={(nr * nr).toFixed(3)} tone="accent" />
      </div>
    </Widget>
  );
}
