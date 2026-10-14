import { scaleLinear } from "d3-scale";
import { useMemo, useRef, useState } from "react";
import { rng } from "../../../engines/data";
import { eigSym, type Mat } from "../../../engines/linalg";
import { quadForm, quadLoss, solveL1Ball, solveL2Ball, type Vec2 } from "../../../engines/linreg";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch03.css";

type Norm = "l1" | "l2";

const TEXT = {
  title: { zh: "L1 与 L2 的几何：菱形的尖角", en: "L1 vs L2 geometry: the diamond's corners" },
  l1: { zh: "L1 菱形（Lasso）", en: "L1 diamond (lasso)" },
  l2: { zh: "L2 圆盘（岭回归）", en: "L2 disc (ridge)" },
  t: { zh: "预算 t", en: "Budget t" },
  corr: { zh: "等高线倾斜", en: "Contour tilt" },
  sol: { zh: "约束解 w*", en: "Constrained w*" },
  sparse: { zh: "落在坐标轴上？", en: "On an axis?" },
  yes: { zh: "是：一个系数 = 0", en: "yes: one coef = 0" },
  no: { zh: "否", en: "no" },
  inside: { zh: "无约束解已在域内", en: "OLS already inside" },
  spray: { zh: "随机撒 400 个中心", en: "Scatter 400 centres" },
  rate: { zh: "落在轴上的比例", en: "Share landing on an axis" },
  caption: {
    zh: "横纵轴是两个系数 (w₁, w₂)。橙点是不加约束的最小二乘解，橙色椭圆是平方损失的等高线；拖动橙点移动它。蓝色区域是约束 ‖w‖ ≤ t。约束解 w* 就是等高线从中心向外扩张、第一次碰到约束域的那一点（实线椭圆与边界相切）。圆盘处处光滑，切点几乎从不落在坐标轴上；菱形的尖角恰好在轴上，而且向外突出，等高线很容易先碰到尖角——这时有一个系数恰好为 0（黄点）。点“随机撒”统计两种约束下落在轴上的比例。",
    en: "The axes are two coefficients (w₁, w₂). The orange dot is the unconstrained least-squares solution and the orange ellipses are contours of the squared loss; drag the dot to move it. The blue region is the constraint ‖w‖ ≤ t. The constrained solution w* is where the contours, growing outward from the centre, first touch the region (the solid ellipse is tangent to the boundary). The disc is smooth everywhere, so the touching point almost never lies on an axis; the diamond's corners sit exactly on the axes and stick out, so the contours often hit a corner first, and then one coefficient is exactly 0 (yellow dot). “Scatter” counts how often each constraint lands on an axis.",
  },
};

const W = 460;
const H = 380;
const XD: [number, number] = [-2.6, 2.6];
const YD: [number, number] = [-2.15, 2.15];
const f1 = (v: number) => v.toFixed(1);

const solve = (norm: Norm, A: number[][], c: Vec2, t: number) => {
  if (norm === "l1") return solveL1Ball(A, c, t);
  const { w } = solveL2Ball(A, c, t);
  return { w, vertex: Math.abs(w[0]) < 1e-6 || Math.abs(w[1]) < 1e-6 };
};

export default function NormBall({ lang, initial = "l1" }: { lang: Lang; initial?: Norm }) {
  const [norm, setNorm] = useState<Norm>(initial);
  const [c, setC] = useState<Vec2>([2.0, 0.4]);
  const [t, setT] = useState(1);
  const [corr, setCorr] = useState(-0.3);
  const [spray, setSpray] = useState(0);
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef(false);
  const sx = scaleLinear().domain(XD).range([0, W]);
  const sy = scaleLinear().domain(YD).range([H, 0]);
  const A = quadForm(corr, 2);
  const { values, vectors } = eigSym(A as Mat);
  const inside = norm === "l1" ? Math.abs(c[0]) + Math.abs(c[1]) <= t : Math.hypot(c[0], c[1]) <= t;
  const { w, vertex } = solve(norm, A, c, t);
  const sparse = !inside && vertex;
  const level = quadLoss(A, c, w);

  // The contour ½ dᵀAd = L is an ellipse with semi-axes √(2L/λᵢ) along the eigenvectors.
  const ellipse = (L: number) => {
    const pts: string[] = [];
    for (let i = 0; i <= 72; i++) {
      const a = (2 * Math.PI * i) / 72;
      const u = Math.sqrt((2 * L) / values[0]) * Math.cos(a);
      const v = Math.sqrt((2 * L) / values[1]) * Math.sin(a);
      pts.push(`${f1(sx(c[0] + u * vectors[0][0] + v * vectors[1][0]))},${f1(sy(c[1] + u * vectors[0][1] + v * vectors[1][1]))}`);
    }
    return "M" + pts.join("L") + "Z";
  };
  const ball = (which: Norm) => which === "l1"
    ? `M${f1(sx(t))},${f1(sy(0))}L${f1(sx(0))},${f1(sy(t))}L${f1(sx(-t))},${f1(sy(0))}L${f1(sx(0))},${f1(sy(-t))}Z`
    : `M${f1(sx(t))},${f1(sy(0))}A${f1(sx(t) - sx(0))},${f1(sy(0) - sy(t))} 0 1 0 ${f1(sx(-t))},${f1(sy(0))}A${f1(sx(t) - sx(0))},${f1(sy(0) - sy(t))} 0 1 0 ${f1(sx(t))},${f1(sy(0))}Z`;

  const cloud = useMemo(() => {
    if (!spray) return null;
    const rand = rng(spray);
    const pts: { c: Vec2; w: Vec2; sparse: boolean }[] = [];
    const hits = { l1: 0, l2: 0 };
    let n = 0;
    while (n < 400) {
      const cc: Vec2 = [XD[0] + (XD[1] - XD[0]) * rand(), YD[0] + (YD[1] - YD[0]) * rand()];
      if (Math.abs(cc[0]) + Math.abs(cc[1]) <= t) continue; // keep centres outside the diamond
      n++;
      const r1 = solve("l1", A, cc, t);
      const r2 = Math.hypot(cc[0], cc[1]) <= t ? { w: cc, vertex: false } : solve("l2", A, cc, t);
      if (r1.vertex) hits.l1++;
      if (r2.vertex) hits.l2++;
      const r = norm === "l1" ? r1 : r2;
      pts.push({ c: cc, w: r.w, sparse: r.vertex });
    }
    return { pts, rate: { l1: hits.l1 / n, l2: hits.l2 / n } };
  }, [spray, t, corr, norm]);

  const move = (e: React.PointerEvent) => {
    if (!drag.current) return;
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.current!.getScreenCTM()!.inverse());
    setC([Math.max(XD[0] + 0.1, Math.min(XD[1] - 0.1, sx.invert(p.x))), Math.max(YD[0] + 0.1, Math.min(YD[1] - 0.1, sy.invert(p.y)))]);
  };

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="chips">
        {(["l1", "l2"] as Norm[]).map((q) => (
          <button key={q} className={`chip${q === norm ? " on" : ""}`} onClick={() => setNorm(q)}>{TEXT[q][lang]}</button>
        ))}
      </div>
      <svg ref={svg} className="c3-svg" viewBox={`0 0 ${W} ${H}`} role="img" onPointerMove={move} onPointerUp={() => (drag.current = false)} onPointerLeave={() => (drag.current = false)}>
        <rect width={W} height={H} rx={12} className="c3-bg" />
        <line x1={0} x2={W} y1={sy(0)} y2={sy(0)} className="c3-axis" />
        <line x1={sx(0)} x2={sx(0)} y1={0} y2={H} className="c3-axis" />
        <text x={W - 10} y={sy(0) - 8} textAnchor="end" className="c3-lab muted">w₁</text>
        <text x={sx(0) + 8} y={16} className="c3-lab muted">w₂</text>
        <path d={ball(norm === "l1" ? "l2" : "l1")} className="nb-ball ghost" />
        <path d={ball(norm)} className="nb-ball" />
        {cloud ? (
          cloud.pts.map((q, i) => <circle key={i} cx={f1(sx(q.w[0]))} cy={f1(sy(q.w[1]))} r={3} className={`nb-cloud ${q.sparse ? "sparse" : "dense"}`} />)
        ) : (
          <>
            {!inside && [0.12, 0.4].map((s) => <path key={s} d={ellipse(level * s)} className="nb-level" />)}
            {!inside && [1.8, 3].map((s) => <path key={s} d={ellipse(level * s)} className="nb-level" />)}
            {inside ? [0.05, 0.2, 0.5].map((L) => <path key={L} d={ellipse(L)} className="nb-level" />) : <path d={ellipse(level)} className="nb-tangent" />}
            <circle cx={f1(sx(w[0]))} cy={f1(sy(w[1]))} r={7} className={`nb-opt${sparse ? " sparse" : ""}`} />
            <text x={sx(w[0]) + 10} y={sy(w[1]) + 18} className="c3-lab">w*</text>
          </>
        )}
        <circle cx={f1(sx(c[0]))} cy={f1(sy(c[1]))} r={5} className="nb-centre" />
        <circle cx={f1(sx(c[0]))} cy={f1(sy(c[1]))} r={11} className="nb-centre c3-handle" opacity={0.35}
          onPointerDown={(e) => { (e.target as Element).setPointerCapture?.(e.pointerId); drag.current = true; setSpray(0); }} />
        <text x={sx(c[0]) + 12} y={sy(c[1]) - 10} className="c3-lab">ŵ_OLS</text>
      </svg>
      <div className="w-row">
        <Slider label={TEXT.t[lang]} value={t} min={0.3} max={1.8} step={0.01} onChange={setT} format={(v) => v.toFixed(2)} />
        <Slider label={TEXT.corr[lang]} value={corr} min={-0.8} max={0.8} step={0.01} onChange={setCorr} format={(v) => v.toFixed(2)} />
        <button className="btn" onClick={() => setSpray(spray ? 0 : 1)}>{spray ? "↺" : TEXT.spray[lang]}</button>
      </div>
      <div className="w-stats">
        <Stat label={TEXT.sol[lang]} value={`(${w[0].toFixed(2)}, ${w[1].toFixed(2)})`} />
        <Stat label={TEXT.sparse[lang]} value={inside ? TEXT.inside[lang] : sparse ? TEXT.yes[lang] : TEXT.no[lang]} tone={sparse ? "accent" : undefined} />
        {cloud && <Stat label={`${TEXT.rate[lang]} · L1 / L2`} value={`${Math.round(100 * cloud.rate.l1)}% / ${Math.round(100 * cloud.rate.l2)}%`} tone="good" />}
      </div>
    </Widget>
  );
}
