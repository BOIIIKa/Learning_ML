import { useRef, useState } from "react";
import { camera, projectOntoColumns, type Vec3 } from "../../../engines/linreg";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch03.css";

const TEXT = {
  title: { zh: "最小二乘即投影", en: "Least squares is a projection" },
  view: { zh: "旋转视角", en: "Rotate view" },
  alt: { zh: "换一个斜率 w₁ + δ", en: "Try slope w₁ + δ" },
  coef: { zh: "ŵ = (截距, 斜率)", en: "ŵ = (intercept, slope)" },
  rss: { zh: "‖e‖²（最小）", en: "‖e‖² (smallest)" },
  orth: { zh: "1ᵀe, xᵀe", en: "1ᵀe, xᵀe" },
  pyth: { zh: "‖y − Xw‖² = ‖e‖² + ‖ŷ − Xw‖²", en: "‖y − Xw‖² = ‖e‖² + ‖ŷ − Xw‖²" },
  caption: {
    zh: "三次测量 x = (1, 2, 3)，观测 y 是三维空间里的一个向量（粉）。设计矩阵的两列 1 = (1,1,1) 和 x 张成一个平面（蓝）——所有直线 w₀ + w₁x 的预测值 Xw 都在这个平面上。最小二乘就是在平面上找离 y 最近的点：ŷ（绿），残差 e = y − ŷ（黄色虚线）与平面垂直，所以 1ᵀe = xᵀe = 0，这正是正规方程 Xᵀ(y − Xw) = 0。拖动粉色箭头的端点改变 y，用滑块旋转视角；再拖动 δ，换一个不是最优的斜率，灰点到 y 的距离满足勾股定理，总比 ‖e‖ 长。",
    en: "Three measurements at x = (1, 2, 3): the observations y form one vector in 3-D space (pink). The two columns of the design matrix, 1 = (1,1,1) and x, span a plane (blue); every line w₀ + w₁x has its predictions Xw on that plane. Least squares finds the point of the plane closest to y: ŷ (green). The residual e = y − ŷ (yellow, dashed) is perpendicular to the plane, so 1ᵀe = xᵀe = 0, which is exactly the normal equations Xᵀ(y − Xw) = 0. Drag the tip of the pink arrow to change y and rotate the view with the slider. Then move δ to try a slope that is not optimal: the grey point's distance to y obeys Pythagoras and is always longer than ‖e‖.",
  },
};

const W = 460;
const H = 330;
const SC = 80;
const COLS = [[1, 1, 1], [1, 2, 3]];
// An orthonormal frame adapted to the column space: two in-plane directions and the normal,
// so the plane is drawn as a tilted floor and the residual points straight up.
const M1: Vec3 = [1 / Math.sqrt(3), 1 / Math.sqrt(3), 1 / Math.sqrt(3)];
const M2: Vec3 = [-1 / Math.sqrt(2), 0, 1 / Math.sqrt(2)];
const NRM: Vec3 = [-1 / Math.sqrt(6), 2 / Math.sqrt(6), -1 / Math.sqrt(6)];
const inPlane = (s: number, t: number) => [0, 1, 2].map((i) => s * M1[i] + t * M2[i]);
const ANCHOR = [0, 1, 2].map((i) => 2.1 * M1[i] + 0.2 * M2[i] + 0.35 * NRM[i]);
const f1 = (v: number) => v.toFixed(1);
const add = (a: number[], b: number[], s = 1) => a.map((v, i) => v + s * b[i]);
const clamp = (v: number) => Math.max(-0.5, Math.min(4.5, v));

export default function ProjectionView({ lang }: { lang: Lang }) {
  const [y, setY] = useState<Vec3>([1, 3, 2]);
  const [theta, setTheta] = useState(-1.3);
  const [delta, setDelta] = useState(0);
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<[number, number] | null>(null);
  const cam = camera(theta, 0.5, [M1, M2, NRM]);
  const [ax, ay] = cam.project(ANCHOR);
  const S = (p: number[]) => {
    const [a, b] = cam.project(p);
    return [W / 2 + SC * (a - ax), H / 2 - SC * (b - ay)] as const;
  };
  const P = (p: number[]) => S(p).map(f1).join(",");
  const { w, yhat, e, xte, rss } = projectOntoColumns(COLS, y);
  const alt = COLS[0].map((_, k) => w[0] * COLS[0][k] + (w[1] + delta) * COLS[1][k]);
  const altD = y.reduce((s, v, k) => s + (v - alt[k]) ** 2, 0);
  const gap = yhat.reduce((s, v, k) => s + (v - alt[k]) ** 2, 0);

  // Plane patch in in-plane coordinates s ∈ [−0.5, 4.4] (along 1/√3), t ∈ [−1.3, 1.7] (orthogonal to it).
  const patch = [inPlane(-0.3, -1.3), inPlane(4.4, -1.3), inPlane(4.4, 1.7), inPlane(-0.3, 1.7)].map(P).join(" ");
  const gridLines = [
    ...[0, 1, 2, 3, 4].map((a) => [inPlane(a, -1.3), inPlane(a, 1.7)]),
    ...[-1, 0, 1].map((b) => [inPlane(-0.3, b), inPlane(4.4, b)]),
  ];

  // Right-angle marker at ŷ: unit residual direction and a unit in-plane direction.
  const en = Math.hypot(...e);
  const u = en > 1e-6 ? e.map((v) => (v / en) * 0.28) : [0, 0, 0];
  const cn = Math.hypot(...COLS[1].map((v, k) => v - 2 * COLS[0][k]));
  const v = COLS[1].map((x, k) => ((x - 2 * COLS[0][k]) / cn) * 0.28);
  const corner = [yhat, add(yhat, u), add(add(yhat, u), v), add(yhat, v)].slice(1).map(P).join(" ");

  const arrow = (from: number[], to: number[], cls: string, id: string) => {
    const [x1, y1] = S(from);
    const [x2, y2] = S(to);
    return <line x1={f1(x1)} y1={f1(y1)} x2={f1(x2)} y2={f1(y2)} className={cls} markerEnd={`url(#${id})`} />;
  };
  const label = (p: number[], text: string, dx = 8, dy = -8, cls = "c3-lab", anchor: "start" | "end" = "start") => {
    const [x, yy] = S(p);
    return <text x={f1(x + dx)} y={f1(yy + dy)} textAnchor={anchor} className={cls}>{text}</text>;
  };

  const onMove = (ev: React.PointerEvent) => {
    if (!drag.current) return;
    const p = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(svg.current!.getScreenCTM()!.inverse());
    const [px, py] = drag.current;
    drag.current = [p.x, p.y];
    const d = cam.unproject((p.x - px) / SC, -(p.y - py) / SC);
    setY((cur) => [clamp(cur[0] + d[0]), clamp(cur[1] + d[1]), clamp(cur[2] + d[2])]);
  };
  const tip = S(y);

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <svg ref={svg} className="c3-svg" viewBox={`0 0 ${W} ${H}`} role="img" onPointerMove={onMove} onPointerUp={() => (drag.current = null)} onPointerLeave={() => (drag.current = null)}>
        <defs>
          {["pv-a1", "pv-a2", "pv-a3"].map((id) => (
            <marker key={id} id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M0,0L10,5L0,10z" fill="context-stroke" />
            </marker>
          ))}
        </defs>
        <rect width={W} height={H} rx={12} className="c3-bg" />
        {[0, 1, 2].map((i) => {
          const end = [0, 0, 0];
          end[i] = 1.6;
          return (
            <g key={i}>
              {arrow([0, 0, 0], end, "c3-axis", "pv-a1")}
              {label(end, `y${"₁₂₃"[i]}`, 4, 4, "c3-lab muted small")}
            </g>
          );
        })}
        <polygon points={patch} className="pv-plane" />
        {gridLines.map(([a, b], i) => <line key={i} x1={f1(S(a)[0])} y1={f1(S(a)[1])} x2={f1(S(b)[0])} y2={f1(S(b)[1])} className="pv-planegrid" />)}
        {arrow([0, 0, 0], COLS[0], "pv-col", "pv-a1")}
        {arrow([0, 0, 0], COLS[1], "pv-col", "pv-a1")}
        {label(COLS[0], "1", 2, 18)}
        {label(COLS[1], "x", 4, -10)}
        {delta !== 0 && (
          <>
            <line x1={f1(S(alt)[0])} y1={f1(S(alt)[1])} x2={f1(tip[0])} y2={f1(tip[1])} className="pv-alt" />
            <circle cx={f1(S(alt)[0])} cy={f1(S(alt)[1])} r={4.5} className="pv-altpt" />
            {label(alt, "Xw", 6, 14, "c3-lab muted")}
          </>
        )}
        {arrow([0, 0, 0], yhat, "pv-yhat", "pv-a2")}
        {arrow([0, 0, 0], y, "pv-y", "pv-a3")}
        <line x1={f1(S(yhat)[0])} y1={f1(S(yhat)[1])} x2={f1(tip[0])} y2={f1(tip[1])} className="pv-res" />
        {en > 0.05 && <polyline points={corner} className="pv-right" />}
        <circle cx={f1(S(yhat)[0])} cy={f1(S(yhat)[1])} r={4} className="pv-hatpt" />
        {label(yhat, "ŷ = Xŵ", 8, 16)}
        {label(add(yhat, e, 0.5), "e", -10, 4, "c3-lab", "end")}
        <circle cx={f1(tip[0])} cy={f1(tip[1])} r={10} className="pv-tip c3-handle" opacity={0.85}
          onPointerDown={(ev) => {
            (ev.target as Element).setPointerCapture?.(ev.pointerId);
            const p = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(svg.current!.getScreenCTM()!.inverse());
            drag.current = [p.x, p.y];
          }} />
        {label(y, `y = (${y.map((v) => v.toFixed(1)).join(", ")})`, -14, -12, "c3-lab", "end")}
      </svg>
      <div className="w-row">
        <Slider label={TEXT.view[lang]} value={theta} min={-2.4} max={-0.2} step={0.01} onChange={setTheta} format={(v) => `${Math.round((v * 180) / Math.PI)}°`} />
        <Slider label={TEXT.alt[lang]} value={delta} min={-0.6} max={0.6} step={0.01} onChange={setDelta} format={(v) => v.toFixed(2)} />
      </div>
      <div className="w-stats">
        <Stat label={TEXT.coef[lang]} value={`(${w[0].toFixed(2)}, ${w[1].toFixed(2)})`} />
        <Stat label={TEXT.rss[lang]} value={rss.toFixed(3)} tone="accent" />
        <Stat label={TEXT.orth[lang]} value={xte.map((v) => (Math.abs(v) < 5e-4 ? 0 : v).toFixed(3)).join(", ")} tone="good" />
        {delta !== 0 && <Stat label={TEXT.pyth[lang]} value={`${altD.toFixed(3)} = ${rss.toFixed(3)} + ${gap.toFixed(3)}`} />}
      </div>
    </Widget>
  );
}
