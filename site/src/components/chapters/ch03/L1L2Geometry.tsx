import { scaleLinear } from "d3-scale";
import { useMemo, useRef, useState } from "react";
import { eigSym, type Mat } from "../../../engines/linalg";
import { quadPenalized, type Penalty } from "../../../engines/linreg";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch03.css";

const TEXT = {
  title: { zh: "L1 的菱形与 L2 的圆", en: "The L1 diamond and the L2 disc" },
  ridge: { zh: "L2（岭回归）", en: "L2 (ridge)" },
  lasso: { zh: "L1（Lasso）", en: "L1 (lasso)" },
  lam: { zh: "惩罚强度 λ", en: "Penalty λ" },
  sol: { zh: "正则化解 ŵ", en: "Regularized solution ŵ" },
  zeros: { zh: "恰好为 0 的系数", en: "Coefficients exactly 0" },
  caption: {
    zh: "空心点是不加惩罚的最小二乘解（可拖动），灰色椭圆是平方损失的等高线。正则化解（橙点）就是等高线第一次碰到约束域（蓝色）的地方，虚线是 λ 从 0 增大时解走过的路径。圆没有尖角，切点一般不在坐标轴上；菱形的尖角正好在坐标轴上，椭圆往往先碰到尖角——于是某个系数恰好等于 0。把空心点拖到靠近某条坐标轴的位置，对比两种惩罚。",
    en: "The hollow point is the unpenalized least-squares solution (drag it); the grey ellipses are contours of the squared loss. The regularized solution (orange) is where a contour first touches the constraint region (blue), and the dashed curve is the path the solution follows as λ grows from 0. A disc has no corners, so the touching point is generally off the axes; a diamond has its corners exactly on the axes, and the ellipse often hits a corner first, which sets a coefficient exactly to 0. Drag the hollow point near one axis and compare the two penalties.",
  },
};

const W = 420;
const H = 420;
const A: Mat = [[1, 0.45], [0.45, 0.55]];
const f1 = (v: number) => v.toFixed(1);
const loss = (w: [number, number], c: [number, number]) => {
  const d = [w[0] - c[0], w[1] - c[1]];
  return A[0][0] * d[0] * d[0] + 2 * A[0][1] * d[0] * d[1] + A[1][1] * d[1] * d[1];
};

export default function L1L2Geometry({ lang }: { lang: Lang }) {
  const [pen, setPen] = useState<Penalty>("lasso");
  const [lam, setLam] = useState(1.2);
  const [c, setC] = useState<[number, number]>([2.1, 0.5]);
  const svg = useRef<SVGSVGElement>(null);
  const dragging = useRef(false);
  const sx = scaleLinear().domain([-1.2, 3.2]).range([0, W]);
  const sy = scaleLinear().domain([-1.2, 3.2]).range([H, 0]);
  const P = (w: [number, number]) => `${f1(sx(w[0]))},${f1(sy(w[1]))}`;

  const eig = useMemo(() => eigSym(A), []);
  const ellipse = (level: number) => {
    const pts = Array.from({ length: 73 }, (_, k) => {
      const t = (2 * Math.PI * k) / 72;
      const a = Math.sqrt(level / eig.values[0]) * Math.cos(t);
      const b = Math.sqrt(level / eig.values[1]) * Math.sin(t);
      return [c[0] + a * eig.vectors[0][0] + b * eig.vectors[1][0], c[1] + a * eig.vectors[0][1] + b * eig.vectors[1][1]] as [number, number];
    });
    return "M" + pts.map(P).join("L") + "Z";
  };

  const w = quadPenalized(A as number[][], c, pen, lam);
  const path = Array.from({ length: 61 }, (_, k) => quadPenalized(A as number[][], c, pen, (k / 60) ** 2 * 8));
  const level = loss(w, c);
  const size = pen === "ridge" ? Math.hypot(w[0], w[1]) : Math.abs(w[0]) + Math.abs(w[1]);
  const region = pen === "ridge"
    ? `M${f1(sx(size))},${f1(sy(0))}A${f1(sx(size) - sx(0))},${f1(sy(0) - sy(size))} 0 1 0 ${f1(sx(-size))},${f1(sy(0))}A${f1(sx(size) - sx(0))},${f1(sy(0) - sy(size))} 0 1 0 ${f1(sx(size))},${f1(sy(0))}Z`
    : `M${P([size, 0])}L${P([0, size])}L${P([-size, 0])}L${P([0, -size])}Z`;

  const toData = (e: React.PointerEvent): [number, number] => {
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.current!.getScreenCTM()!.inverse());
    return [Math.max(-0.8, Math.min(3, sx.invert(pt.x))), Math.max(-0.8, Math.min(3, sy.invert(pt.y)))];
  };
  const zeros = w.filter((v) => Math.abs(v) < 1e-9).length;

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="chips">
        {(["ridge", "lasso"] as Penalty[]).map((p) => (
          <button key={p} className={`chip${pen === p ? " on" : ""}`} onClick={() => setPen(p)}>{TEXT[p][lang]}</button>
        ))}
      </div>
      <svg ref={svg} className="lm-svg" viewBox={`0 0 ${W} ${H}`} role="img"
        onPointerMove={(e) => dragging.current && setC(toData(e))}
        onPointerUp={() => (dragging.current = false)} onPointerLeave={() => (dragging.current = false)}>
        <rect width={W} height={H} className="plot-bg" />
        <line x1={f1(sx(-1.2))} x2={f1(sx(3.2))} y1={f1(sy(0))} y2={f1(sy(0))} className="lm-axis" />
        <line x1={f1(sx(0))} x2={f1(sx(0))} y1={f1(sy(-1.2))} y2={f1(sy(3.2))} className="lm-axis" />
        <text x={f1(sx(3.05))} y={f1(sy(0) - 6)} className="lr-label">w₁</text>
        <text x={f1(sx(0) + 6)} y={f1(sy(3.05))} className="lr-label">w₂</text>
        {[0.25, 0.5, 1, 2].map((m) => <path key={m} d={ellipse(level * m + 0.02)} className="l12-level" />)}
        {size > 1e-6 && <path d={region} className="l12-region" />}
        <path d={"M" + path.map(P).join("L")} className="l12-path" />
        <path d={ellipse(Math.max(level, 1e-4))} className="l12-touch" />
        <circle cx={f1(sx(c[0]))} cy={f1(sy(c[1]))} r={7} className="l12-center"
          onPointerDown={(e) => { dragging.current = true; svg.current!.setPointerCapture(e.pointerId); }} />
        <circle cx={f1(sx(w[0]))} cy={f1(sy(w[1]))} r={6} className="l12-sol" />
      </svg>
      <Slider label={TEXT.lam[lang]} value={lam} min={0} max={4} step={0.05} onChange={setLam} />
      <div className="w-stats">
        <Stat label={TEXT.sol[lang]} value={`(${w[0].toFixed(2)}, ${w[1].toFixed(2)})`} />
        <Stat label={TEXT.zeros[lang]} value={zeros} tone={zeros ? "accent" : undefined} />
      </div>
    </Widget>
  );
}
