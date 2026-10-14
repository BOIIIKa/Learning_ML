import { scaleLinear } from "d3-scale";
import { useRef, useState } from "react";
import { projectHalfplane, type Vec } from "../../../engines/linalg";
import type { Lang } from "../../../i18n/ui";
import Widget, { Stat } from "../../ui/Widget";

const TEXT = {
  title: { zh: "约束优化与 KKT 条件", en: "Constrained optimization and KKT" },
  lambda: { zh: "乘子 λ*", en: "Multiplier λ*" },
  status: { zh: "约束状态", en: "Constraint" },
  active: { zh: "起作用（g = 0）", en: "active (g = 0)" },
  inactive: { zh: "不起作用（λ = 0）", en: "inactive (λ = 0)" },
  slack: { zh: "λ · g(x*)", en: "λ · g(x*)" },
  caption: {
    zh: "目标：离目标点 t（拖动橙点）尽量近，即 min ‖x − t‖²；约束：留在阴影半平面 a·x ≤ c 内。t 在可行域内时约束不起作用，λ = 0；t 跑出去后，最优点 x* 落在边界上，此时 ∇f（紫）与 λ∇g（青）大小相等、方向相反，正好抵消——这就是平稳性条件 ∇f + λ∇g = 0。无论哪种情况，λ·g(x*) 都等于 0：互补松弛。",
    en: "Goal: get as close as possible to the target t (drag the orange point), i.e. min ‖x − t‖², while staying in the shaded half-plane a·x ≤ c. When t is feasible the constraint is inactive and λ = 0. Once t leaves, the optimum x* sits on the boundary, where ∇f (purple) and λ∇g (cyan) have equal length and opposite directions: stationarity, ∇f + λ∇g = 0. Either way λ·g(x*) = 0: complementary slackness.",
  },
};

const W = 460;
const H = 360;
const a: Vec = [1, 1.4];
const c = 1;

export default function KKTPlayground({ lang }: { lang: Lang }) {
  const [t, setT] = useState<Vec>([1.8, 1.6]);
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef(false);
  const sx = scaleLinear().domain([-3, 3]).range([0, W]);
  const sy = scaleLinear().domain([-2.4, 2.4]).range([H, 0]);
  const { x, lambda, active } = projectHalfplane(t, a, c);

  // Boundary line a·x = c across the box, and the feasible region polygon (a·x ≤ c).
  const yAt = (xx: number) => (c - a[0] * xx) / a[1];
  const poly = [[-3, yAt(-3)], [3, yAt(3)], [3, -2.4], [-3, -2.4]].map(([px, py]) => `${sx(px).toFixed(1)},${sy(py).toFixed(1)}`).join(" ");
  const move = (e: React.PointerEvent) => {
    if (!drag.current) return;
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.current!.getScreenCTM()!.inverse());
    setT([Math.max(-2.9, Math.min(2.9, sx.invert(p.x))), Math.max(-2.3, Math.min(2.3, sy.invert(p.y)))]);
  };
  const g = a[0] * x[0] + a[1] * x[1] - c;
  // Arrows at x*: ∇f = 2(x − t) and λ∇g = λa. Stationarity says they cancel (drawn scaled).
  const s = 0.6;
  const gradF: Vec = [2 * (x[0] - t[0]) * s, 2 * (x[1] - t[1]) * s];
  const lamGrad: Vec = [lambda * a[0] * s, lambda * a[1] * s];

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <svg ref={svg} className="kkt-svg" viewBox={`0 0 ${W} ${H}`} role="img" onPointerMove={move} onPointerUp={() => (drag.current = false)} onPointerLeave={() => (drag.current = false)}>
        <defs>
          <marker id="kkt-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0L10,5L0,10z" fill="context-stroke" />
          </marker>
        </defs>
        <rect width={W} height={H} rx={12} className="plot-bg" />
        <polygon points={poly} className="kkt-feasible" />
        <line x1={sx(-3)} y1={sy(yAt(-3))} x2={sx(3)} y2={sy(yAt(3))} className="kkt-boundary" />
        {[0.5, 1, 1.5, 2, 2.5].map((r) => <circle key={r} cx={sx(t[0])} cy={sy(t[1])} r={sx(r) - sx(0)} className="kkt-level" />)}
        <line x1={sx(x[0])} y1={sy(x[1])} x2={sx(t[0])} y2={sy(t[1])} className="kkt-dist" />
        {active && (
          <>
            <line x1={sx(x[0])} y1={sy(x[1])} x2={sx(x[0] + gradF[0])} y2={sy(x[1] + gradF[1])} className="kkt-negf" markerEnd="url(#kkt-arrow)" />
            <line x1={sx(x[0])} y1={sy(x[1])} x2={sx(x[0] + lamGrad[0])} y2={sy(x[1] + lamGrad[1])} className="kkt-lamg" markerEnd="url(#kkt-arrow)" />
          </>
        )}
        <circle cx={sx(x[0])} cy={sy(x[1])} r={7} className="kkt-opt" />
        <circle cx={sx(t[0])} cy={sy(t[1])} r={9} className="kkt-target" onPointerDown={(e) => { (e.target as Element).setPointerCapture?.(e.pointerId); drag.current = true; }} />
        <text x={sx(t[0]) + 12} y={sy(t[1]) - 10} className="kkt-lab">t</text>
        <text x={sx(x[0]) + 10} y={sy(x[1]) + 18} className="kkt-lab">x*</text>
        <text x={14} y={H - 14} className="kkt-lab muted">a·x ≤ c</text>
      </svg>
      <div className="w-stats">
        <Stat label={TEXT.status[lang]} value={active ? TEXT.active[lang] : TEXT.inactive[lang]} tone={active ? "accent" : "good"} />
        <Stat label={TEXT.lambda[lang]} value={lambda.toFixed(3)} />
        <Stat label={TEXT.slack[lang]} value={(lambda * g).toFixed(3)} />
      </div>
    </Widget>
  );
}
