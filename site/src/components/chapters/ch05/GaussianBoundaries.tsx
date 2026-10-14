import { contours } from "d3-contour";
import { scaleLinear } from "d3-scale";
import { useMemo, useRef, useState } from "react";
import { r6 } from "../../../engines/data";
import {
  covFromAxes, lda2, ldaFromParams, logisticNewton, placeGaussian, qda2, qdaFromParams, standardNormals,
} from "../../../engines/generative";
import type { Vec } from "../../../engines/linalg";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch05.css";

const TEXT = {
  title: { zh: "从高斯到边界：LDA、QDA 与逻辑回归", en: "From Gaussians to boundaries: LDA, QDA and logistic regression" },
  n: { zh: "每类训练样本数", en: "Training points per class" },
  shared: { zh: "两类共用一个协方差", en: "Both classes share one covariance" },
  bayes: { zh: "显示贝叶斯边界", en: "Show the Bayes boundary" },
  resample: { zh: "↻ 重新抽样", en: "↻ Resample" },
  lda: { zh: "LDA", en: "LDA" },
  qda: { zh: "QDA", en: "QDA" },
  lr: { zh: "逻辑回归", en: "Logistic" },
  bayesL: { zh: "贝叶斯（真分布）", en: "Bayes (true model)" },
  acc: { zh: "测试准确率", en: "test accuracy" },
  caption: {
    zh: "两类数据各来自一个二维高斯（蓝 = 常规，橙 = 需复查）。拖动实心圆移动均值，拖动椭圆上的空心把手旋转、拉伸协方差椭圆（虚线椭圆是 2σ 等高线）。三种方法都只看训练点：LDA 用一个共享协方差，边界是直线；QDA 每类一个协方差，边界是二次曲线；逻辑回归不建模 x 的分布，直接拟合一条直线。勾选“共用协方差”时 QDA 会几乎变直、与 LDA 重合；两类形状差得越多，QDA 越弯。样本少时，QDA 的曲线也最容易乱晃。",
    en: "Each class comes from a 2-D Gaussian (blue = routine, orange = needs follow-up). Drag a filled dot to move a mean; drag the hollow handles to rotate and stretch a covariance ellipse (the dashed ellipse is the 2σ contour). All three methods see only the training points: LDA uses one shared covariance and draws a straight line; QDA gives each class its own covariance and draws a conic; logistic regression never models the distribution of x and fits a line directly. Tick “share one covariance” and QDA nearly straightens onto LDA; the more the two shapes differ, the more QDA bends. With few points, QDA's curve is also the one that wobbles most.",
  },
};

interface Shape { m: Vec; s1: number; s2: number; th: number }
type HandleId = "m0" | "a0" | "b0" | "m1" | "a1" | "b1";

const W = 460;
const H = 345;
const DOM = { x: [-4, 4] as [number, number], y: [-3, 3] as [number, number] };
const f1 = (v: number) => v.toFixed(1);
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/** Segment of w·p + b = 0 inside the plotting box, or null. */
function clipLine(w: Vec, b: number): [Vec, Vec] | null {
  const pts: Vec[] = [];
  const [x0, x1] = DOM.x;
  const [y0, y1] = DOM.y;
  if (Math.abs(w[1]) > 1e-12) for (const x of [x0, x1]) { const y = -(w[0] * x + b) / w[1]; if (y >= y0 && y <= y1) pts.push([x, y]); }
  if (Math.abs(w[0]) > 1e-12) for (const y of [y0, y1]) { const x = -(w[1] * y + b) / w[0]; if (x >= x0 && x <= x1) pts.push([x, y]); }
  return pts.length >= 2 ? [pts[0], pts[pts.length - 1]] : null;
}

const NX = 65;
const NY = 49;
const GX = Array.from({ length: NX }, (_, i) => DOM.x[0] + ((DOM.x[1] - DOM.x[0]) * i) / (NX - 1));
const GY = Array.from({ length: NY }, (_, i) => DOM.y[0] + ((DOM.y[1] - DOM.y[0]) * i) / (NY - 1));

export default function GaussianBoundaries({ lang }: { lang: Lang }) {
  const [shapes, setShapes] = useState<[Shape, Shape]>([
    { m: [-1.3, -0.5], s1: 1.1, s2: 0.5, th: 0.5 },
    { m: [1.2, 0.6], s1: 0.8, s2: 0.45, th: -0.7 },
  ]);
  const [n, setN] = useState(40);
  const [shared, setShared] = useState(false);
  const [showBayes, setShowBayes] = useState(false);
  const [seed, setSeed] = useState(1);
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<HandleId | null>(null);

  const sx = scaleLinear().domain(DOM.x).range([0, W]);
  const sy = scaleLinear().domain(DOM.y).range([H, 0]);
  const scale = W / (DOM.x[1] - DOM.x[0]);
  const eff: [Shape, Shape] = shared ? [shapes[0], { ...shapes[0], m: shapes[1].m }] : shapes;

  const z = useMemo(() => [standardNormals(150, 10 * seed), standardNormals(150, 10 * seed + 5)], [seed]);
  const zt = useMemo(() => [standardNormals(1500, 901), standardNormals(1500, 902)], []);

  const place = (zz: Vec[], s: Shape) => placeGaussian(zz, s.m, s.s1, s.s2, s.th).map(([a, b]) => [r6(a), r6(b)] as Vec);
  const train = useMemo(() => {
    const A = place(z[0].slice(0, n), eff[0]);
    const B = place(z[1].slice(0, n), eff[1]);
    return { X: [...A, ...B], y: [...A.map(() => -1), ...B.map(() => 1)] };
  }, [z, n, eff[0], eff[1]]);

  const models = useMemo(() => {
    const { X, y } = train;
    const lda = lda2(X, y);
    const qda = qda2(X, y);
    const lr = logisticNewton(X, y.map((v) => (v > 0 ? 1 : 0)), 1e-2);
    const S0 = covFromAxes(eff[0].s1, eff[0].s2, eff[0].th);
    const S1 = covFromAxes(eff[1].s1, eff[1].s2, eff[1].th);
    const bayes = shared ? (() => { const l = ldaFromParams(eff[0].m, eff[1].m, S0, 0.5); return (p: Vec) => l.w[0] * p[0] + l.w[1] * p[1] + l.b; })()
      : qdaFromParams(eff[0].m, S0, eff[1].m, S1, 0.5);
    return { lda, qda, lr: { w: [lr.w[0], lr.w[1]] as Vec, b: lr.b }, bayes };
  }, [train, shared]);

  const test = useMemo(() => {
    const A = placeGaussian(zt[0], eff[0].m, eff[0].s1, eff[0].s2, eff[0].th);
    const B = placeGaussian(zt[1], eff[1].m, eff[1].s1, eff[1].s2, eff[1].th);
    const acc = (f: (p: Vec) => number) => (A.filter((p) => f(p) < 0).length + B.filter((p) => f(p) > 0).length) / (A.length + B.length);
    const lin = (l: { w: Vec; b: number }) => (p: Vec) => l.w[0] * p[0] + l.w[1] * p[1] + l.b;
    return { lda: acc(lin(models.lda)), qda: acc(models.qda), lr: acc(lin(models.lr)), bayes: acc(models.bayes) };
  }, [models, zt]);

  const paths = useMemo(() => {
    const dx = GX[1] - GX[0];
    const dy = GY[1] - GY[0];
    const ring = (rings: number[][][][]) => rings.flatMap((poly) => poly.map((r) =>
      "M" + r.map(([gx, gy]) => `${f1(sx(GX[0] + (gx - 0.5) * dx))},${f1(sy(GY[0] + (gy - 0.5) * dy))}`).join("L") + "Z")).join("");
    const field = (f: (p: Vec) => number) => {
      const v: number[] = [];
      for (const yy of GY) for (const xx of GX) v.push(f([xx, yy]));
      return v;
    };
    const q = field(models.qda);
    const gen = contours().size([NX, NY]).thresholds([0]);
    const pos = ring(gen(q)[0].coordinates as number[][][][]);
    const bayes = showBayes ? ring(gen(field(models.bayes))[0].coordinates as number[][][][]) : "";
    return { pos, bayes };
  }, [models, showBayes]);

  const line = (l: { w: Vec; b: number }) => {
    const seg = clipLine(l.w, l.b);
    return seg ? `M${f1(sx(seg[0][0]))},${f1(sy(seg[0][1]))}L${f1(sx(seg[1][0]))},${f1(sy(seg[1][1]))}` : "";
  };

  const toData = (e: React.PointerEvent): Vec => {
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.current!.getScreenCTM()!.inverse());
    return [sx.invert(p.x), sy.invert(p.y)];
  };
  const onMove = (e: React.PointerEvent) => {
    const id = drag.current;
    if (!id) return;
    const [x, y] = toData(e);
    const k = +id[1] as 0 | 1;
    const next: [Shape, Shape] = [{ ...shapes[0] }, { ...shapes[1] }];
    const s = next[k];
    if (id[0] === "m") s.m = [clamp(x, -3.6, 3.6), clamp(y, -2.6, 2.6)];
    else if (id[0] === "a") {
      const dx = x - s.m[0];
      const dy = y - s.m[1];
      s.th = Math.atan2(dy, dx);
      s.s1 = clamp(Math.hypot(dx, dy) / 2, 0.2, 2);
    } else {
      const proj = -(x - s.m[0]) * Math.sin(s.th) + (y - s.m[1]) * Math.cos(s.th);
      s.s2 = clamp(Math.abs(proj) / 2, 0.15, 2);
    }
    setShapes(next);
  };
  const grab = (id: HandleId) => (e: React.PointerEvent) => { (e.target as Element).setPointerCapture?.(e.pointerId); drag.current = id; };

  const ellipse = (s: Shape, k: 0 | 1) => {
    const cls = k ? "pos" : "neg";
    const cx = sx(s.m[0]);
    const cy = sy(s.m[1]);
    const a: Vec = [s.m[0] + 2 * s.s1 * Math.cos(s.th), s.m[1] + 2 * s.s1 * Math.sin(s.th)];
    const b: Vec = [s.m[0] - 2 * s.s2 * Math.sin(s.th), s.m[1] + 2 * s.s2 * Math.cos(s.th)];
    const editable = !(shared && k === 1);
    return (
      <g key={k}>
        <ellipse cx={f1(cx)} cy={f1(cy)} rx={f1(2 * s.s1 * scale)} ry={f1(2 * s.s2 * scale)} transform={`rotate(${f1((-s.th * 180) / Math.PI)} ${f1(cx)} ${f1(cy)})`} className={`g5-ell ${cls}`} />
        {editable && (
          <>
            <line x1={f1(cx)} y1={f1(cy)} x2={f1(sx(a[0]))} y2={f1(sy(a[1]))} className={`g5-axisline ${cls}`} />
            <line x1={f1(cx)} y1={f1(cy)} x2={f1(sx(b[0]))} y2={f1(sy(b[1]))} className={`g5-axisline ${cls}`} />
            <circle cx={f1(sx(a[0]))} cy={f1(sy(a[1]))} r={7} className={`g5-handle ${cls}`} onPointerDown={grab(`a${k}` as HandleId)} />
            <circle cx={f1(sx(b[0]))} cy={f1(sy(b[1]))} r={6} className={`g5-handle ${cls}`} onPointerDown={grab(`b${k}` as HandleId)} />
          </>
        )}
      </g>
    );
  };

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <svg ref={svg} className="g5-svg" viewBox={`0 0 ${W} ${H}`} role="img" onPointerMove={onMove}
        onPointerUp={() => (drag.current = null)} onPointerLeave={() => (drag.current = null)}>
        <rect width={W} height={H} className="g5-bg" />
        <rect width={W} height={H} className="g5-region neg" />
        <path d={paths.pos} className="g5-region pos" />
        <line x1={0} x2={W} y1={f1(sy(0))} y2={f1(sy(0))} className="g5-axis" />
        <line x1={f1(sx(0))} x2={f1(sx(0))} y1={0} y2={H} className="g5-axis" />
        {train.X.map((p, i) => <circle key={i} cx={f1(sx(p[0]))} cy={f1(sy(p[1]))} r={3.6} className={`g5-pt ${train.y[i] > 0 ? "pos" : "neg"}`} />)}
        {showBayes && <path d={paths.bayes} className="g5-bayes" />}
        <path d={paths.pos} className="g5-qda" />
        <path d={line(models.lda)} className="g5-lda" />
        <path d={line(models.lr)} className="g5-lr" />
        {ellipse(eff[0], 0)}
        {ellipse(eff[1], 1)}
        {[0, 1].map((k) => (
          <circle key={`m${k}`} cx={f1(sx(eff[k].m[0]))} cy={f1(sy(eff[k].m[1]))} r={9} className={`g5-mean ${k ? "pos" : "neg"}`} onPointerDown={grab(`m${k}` as HandleId)} />
        ))}
      </svg>
      <div className="g5-legend">
        <span><i className="lda" />LDA</span>
        <span><i className="qda" />QDA</span>
        <span><i className="lr" />{TEXT.lr[lang]}</span>
        {showBayes && <span><i className="bayes" />{TEXT.bayesL[lang]}</span>}
      </div>
      <div className="w-row">
        <Slider label={TEXT.n[lang]} value={n} min={5} max={150} step={1} onChange={setN} />
        <button className="btn" onClick={() => setSeed(seed + 1)}>{TEXT.resample[lang]}</button>
      </div>
      <div className="chips">
        <button className={`chip${shared ? " on" : ""}`} onClick={() => setShared(!shared)}>{TEXT.shared[lang]}</button>
        <button className={`chip${showBayes ? " on" : ""}`} onClick={() => setShowBayes(!showBayes)}>{TEXT.bayes[lang]}</button>
      </div>
      <div className="w-stats">
        <Stat label={`LDA · ${TEXT.acc[lang]}`} value={`${(100 * test.lda).toFixed(1)}%`} />
        <Stat label={`QDA · ${TEXT.acc[lang]}`} value={`${(100 * test.qda).toFixed(1)}%`} />
        <Stat label={`${TEXT.lr[lang]} · ${TEXT.acc[lang]}`} value={`${(100 * test.lr).toFixed(1)}%`} />
        <Stat label={`${TEXT.bayesL[lang]}`} value={`${(100 * test.bayes).toFixed(1)}%`} tone="accent" />
      </div>
    </Widget>
  );
}
