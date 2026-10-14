import { scaleLinear } from "d3-scale";
import { useMemo, useRef, useState } from "react";
import { accuracy, crossEntropy, fitLogistic, newtonStep, patients, sigmoid } from "../../../engines/linclass";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import Widget, { Slider, Stat } from "../../ui/Widget";
import { Arrow, f1, lineInBox, type Box } from "./geom";
import "./ch04.css";

const TEXT = {
  title: { zh: "逻辑回归的概率曲面", en: "The logistic probability surface" },
  ce: { zh: "平均交叉熵（nats）", en: "Mean cross-entropy (nats)" },
  best: { zh: "极大似然处的最小值", en: "Minimum, at the MLE" },
  acc: { zh: "训练准确率", en: "Training accuracy" },
  norm: { zh: "‖w‖（曲面陡峭程度）", en: "‖w‖ (steepness)" },
  newton: { zh: "牛顿法走一步", en: "One Newton step" },
  reset: { zh: "重置", en: "Reset" },
  slice: {
    zh: "沿 w 方向切开曲面：横轴 z = w·x + b，纵轴 P(y = 1 | x)。点画在 0（健康）或 1（患病）处，竖线是它离 σ(z) 的差距。",
    en: "The surface sliced along w: horizontal axis z = w·x + b, vertical axis P(y = 1 | x). Points sit at 0 (healthy) or 1 (diabetic); the sticks show their gap to σ(z).",
  },
  caption: {
    zh: "颜色深浅是模型给出的患病概率 σ(w·x + b)：越橙越可能患病，越蓝越可能健康。拖动青色把手改变 w（方向决定边界的朝向，长度决定概率过渡的陡峭程度），用滑块改变 b（平移边界）。实线是 P = 0.5 的决策边界，两条虚线是 P = 0.1 与 0.9 的等高线——它们都是直线，因为概率只依赖于 w·x + b。点“牛顿法走一步”，几步之内就能到达交叉熵的最小值。",
    en: "Shading is the model's probability of diabetes σ(w·x + b): more orange means more likely diabetic, more blue more likely healthy. Drag the cyan handle to change w (its direction orients the boundary, its length sets how sharply the probability changes) and use the slider to shift b. The solid line is the P = 0.5 decision boundary; the dashed lines are the P = 0.1 and 0.9 contours. All of them are straight because the probability depends on x only through w·x + b. Press “One Newton step”: a handful of steps reach the minimum cross-entropy.",
  },
};

const W = 460;
const H = 360;
const BOX: Box = { x: [-3, 3], y: [-2.35, 2.35] };
const ARROW = 0.6; // data units of arrow per unit of ‖w‖
const NX = 46;
const NY = 36;
const W0: [number, number] = [1.4, -0.7];

export default function LogisticSurface({ lang }: { lang: Lang }) {
  const { X, y } = useMemo(() => patients(), []);
  const best = useMemo(() => fitLogistic(X, y, 10).at(-1)!, [X, y]);
  const [w, setW] = useState<[number, number]>(W0);
  const [b, setB] = useState(0.6);
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef(false);

  const sx = scaleLinear().domain(BOX.x).range([0, W]);
  const sy = scaleLinear().domain(BOX.y).range([H, 0]);
  const cw = W / NX;
  const ch = H / NY;

  const cells = [];
  for (let r = 0; r < NY; r++) {
    for (let c = 0; c < NX; c++) {
      const px = sx.invert((c + 0.5) * cw);
      const py = sy.invert((r + 0.5) * ch);
      const p = sigmoid(w[0] * px + w[1] * py + b);
      const op = (0.04 + 0.5 * Math.abs(2 * p - 1)).toFixed(2);
      cells.push(<rect key={r * NX + c} x={f1(c * cw)} y={f1(r * ch)} width={cw} height={ch} fillOpacity={op} className={`lc-cell ${p >= 0.5 ? "pos" : "neg"}`} />);
    }
  }

  const levels = [
    { p: 0.1, cls: "neg" },
    { p: 0.5, cls: "mid" },
    { p: 0.9, cls: "pos" },
  ].map((l) => ({ ...l, seg: lineInBox(w, b - Math.log(l.p / (1 - l.p)), BOX) }));

  const move = (e: React.PointerEvent) => {
    if (!drag.current) return;
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.current!.getScreenCTM()!.inverse());
    const dx = Math.max(BOX.x[0], Math.min(BOX.x[1], sx.invert(pt.x)));
    const dy = Math.max(BOX.y[0], Math.min(BOX.y[1], sy.invert(pt.y)));
    setW([+(dx / ARROW).toFixed(3), +(dy / ARROW).toFixed(3)]);
  };

  const z = X.map((x) => w[0] * x[0] + w[1] * x[1] + b);
  const t = y.map((v, i) => (v > 0 ? 1 : 0) + ((i % 5) - 2) * 0.012);
  const ce = crossEntropy(w, b, X, y);
  const tip: [number, number] = [w[0] * ARROW, w[1] * ARROW];

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <svg ref={svg} className="lc-svg" viewBox={`0 0 ${W} ${H}`} role="img" onPointerMove={move} onPointerUp={() => (drag.current = false)} onPointerLeave={() => (drag.current = false)}>
        <rect width={W} height={H} rx={12} className="plot-bg" />
        <g shapeRendering="crispEdges">{cells}</g>
        {levels.map((l) => l.seg && (
          <g key={l.p}>
            <line x1={f1(sx(l.seg[0][0]))} y1={f1(sy(l.seg[0][1]))} x2={f1(sx(l.seg[1][0]))} y2={f1(sy(l.seg[1][1]))} className={l.p === 0.5 ? "lc-sep" : `lc-level ${l.cls}`} />
          </g>
        ))}
        {X.map((p, i) => (
          <g key={i} transform={`translate(${f1(sx(p[0]))},${f1(sy(p[1]))})`}>
            {y[i] * z[i] <= 0 && <circle r={9} className="lc-wrong" />}
            <circle r={5.2} className={`lc-pt ${y[i] > 0 ? "pos" : "neg"}`} />
          </g>
        ))}
        <Arrow from={[sx(0), sy(0)]} to={[sx(tip[0]), sy(tip[1])]} cls="w" />
        <text x={f1(sx(tip[0]) + 12)} y={f1(sy(tip[1]) + 4)} className="lc-vlab w">w</text>
        <circle cx={f1(sx(tip[0]))} cy={f1(sy(tip[1]))} r={9} className="lc-handle"
          onPointerDown={(e) => { (e.target as Element).setPointerCapture?.(e.pointerId); drag.current = true; }} />
        {levels.map((l) => l.seg && (
          <text key={`t${l.p}`} x={f1(Math.min(W - 54, Math.max(6, sx(l.seg[1][0]) + 4)))} y={f1(Math.min(H - 8, Math.max(14, sy(l.seg[1][1]) + 14)))} className={`lc-level-lab ${l.cls}`}>P={l.p}</text>
        ))}
      </svg>

      <CurvePlot xs={z} ys={t} xDomain={[-6, 6]} yDomain={[-0.12, 1.12]} height={190}
        curves={[{ f: sigmoid, color: "var(--accent)" }]} residualsTo={sigmoid} xLabel="z = w·x + b" />
      <p className="lc-sub">{TEXT.slice[lang]}</p>

      <div className="w-row">
        <Slider label="w₁" value={w[0]} min={-6} max={6} step={0.01} onChange={(v) => setW([v, w[1]])} format={(v) => v.toFixed(2)} />
        <Slider label="w₂" value={w[1]} min={-6} max={6} step={0.01} onChange={(v) => setW([w[0], v])} format={(v) => v.toFixed(2)} />
        <Slider label="b" value={b} min={-4} max={4} step={0.01} onChange={setB} format={(v) => v.toFixed(2)} />
      </div>
      <div className="w-row">
        <button className="btn btn-primary" onClick={() => {
          // A damped Newton step: halve it until the loss does not increase (far from the optimum a full step can overshoot).
          const n = newtonStep(w, b, X, y);
          let s = 1;
          const at = (s: number) => [w[0] + s * (n.w[0] - w[0]), w[1] + s * (n.w[1] - w[1]), b + s * (n.b - b)];
          while (s > 1e-3 && crossEntropy([at(s)[0], at(s)[1]], at(s)[2], X, y) > ce) s /= 2;
          const [a0, a1, a2] = at(s);
          setW([+a0.toFixed(3), +a1.toFixed(3)]);
          setB(+a2.toFixed(3));
        }}>{TEXT.newton[lang]}</button>
        <button className="btn" onClick={() => { setW(W0); setB(0.6); }}>{TEXT.reset[lang]}</button>
      </div>
      <div className="w-stats">
        <Stat label={TEXT.ce[lang]} value={ce.toFixed(4)} tone={ce - best.loss < 5e-4 ? "good" : "accent"} />
        <Stat label={TEXT.best[lang]} value={best.loss.toFixed(4)} />
        <Stat label={TEXT.acc[lang]} value={`${(100 * accuracy(w, b, X, y)).toFixed(1)}%`} />
        <Stat label={TEXT.norm[lang]} value={Math.hypot(...w).toFixed(2)} />
      </div>
    </Widget>
  );
}
