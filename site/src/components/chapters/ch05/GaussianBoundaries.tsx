import { useMemo, useState } from "react";
import {
  covFrom, errorRate, fitGda, ldaLine, logisticLine, qdaLogOdds, sampleClasses, stdNormals, type Gauss, type M2,
} from "../../../engines/generative-classifiers";
import { eigSym } from "../../../engines/linalg";
import type { V2 } from "../../../engines/linear-classification";
import type { Lang } from "../../../i18n/ui";
import Plot2D from "../../viz/Plot2D";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch05.css";

const TEXT = {
  title: { zh: "从两团高斯到一条边界", en: "From two Gaussians to a boundary" },
  shared: { zh: "共享协方差", en: "Shared covariance" },
  own: { zh: "各自的协方差", en: "Separate covariances" },
  angle: { zh: "红色椭圆的方向", en: "Red ellipse direction" },
  stretch: { zh: "红色椭圆的长轴", en: "Red ellipse long axis" },
  prior: { zh: "先验 P(复查)", en: "Prior P(follow-up)" },
  lda: { zh: "LDA 测试误差", en: "LDA test error" },
  qda: { zh: "QDA 测试误差", en: "QDA test error" },
  lr: { zh: "逻辑回归测试误差", en: "Logistic test error" },
  legend: { zh: "黄线 LDA · 白色曲线 QDA · 绿虚线 逻辑回归", en: "yellow LDA · white curve QDA · green dashed logistic" },
  caption: {
    zh: "先为每一类病人画一幅“画像”：一团高斯（虚线椭圆是真实分布的 2σ 等高线），再用贝叶斯公式翻转成 P(复查 | x)。只估计均值和协方差，不需要任何迭代优化。两类共享协方差时，二次项相消，边界是直线（LDA）；协方差不同时，边界是二次曲线（QDA，色带的零等高线）。虚线是第 04 章的逻辑回归：它和 LDA 同属“对数几率线性”的模型族，只是估计方法不同。拖动先验，三条边界一起平移——先验只改截距。数据为合成示意。",
    en: "First draw a portrait of each group of patients: one Gaussian each (the dashed ellipses are 2σ contours of the true distributions), then flip it with Bayes' rule into P(follow-up | x). Only means and covariances are estimated; no iterative optimization. With a shared covariance the quadratic terms cancel and the boundary is a line (LDA); with different covariances it is a conic (QDA, the zero contour of the shaded field). The dashed line is Chapter 04's logistic regression: the same family of linear log-odds as LDA, estimated differently. Drag the prior and all boundaries shift together: the prior only changes the intercept. Synthetic illustrative data.",
  },
};

const NEG_MEAN: V2 = [-0.9, -0.5];
const POS_MEAN: V2 = [0.9, 0.6];
const f1 = (v: number) => v.toFixed(1);

function ellipse(g: Gauss, sx: (v: number) => number, sy: (v: number) => number) {
  const { values, vectors } = eigSym(g.cov as M2 as [[number, number], [number, number]]);
  const pts = Array.from({ length: 49 }, (_, i) => {
    const t = (2 * Math.PI * i) / 48;
    const a = 2 * Math.sqrt(values[0]) * Math.cos(t);
    const b = 2 * Math.sqrt(Math.max(0, values[1])) * Math.sin(t);
    return `${f1(sx(g.mean[0] + a * vectors[0][0] + b * vectors[1][0]))},${f1(sy(g.mean[1] + a * vectors[0][1] + b * vectors[1][1]))}`;
  });
  return "M" + pts.join("L") + "Z";
}

export default function GaussianBoundaries({ lang }: { lang: Lang }) {
  const [own, setOwn] = useState(true);
  const [deg, setDeg] = useState(35);
  const [len, setLen] = useState(1.1);
  const [prior, setPrior] = useState(0.5);
  const z = useMemo(() => ({ p: stdNormals(70, 11), n: stdNormals(70, 12), tp: stdNormals(2000, 13), tn: stdNormals(2000, 14) }), []);

  const pos: Gauss = { mean: POS_MEAN, cov: covFrom(len, 0.4, deg) };
  const neg: Gauss = { mean: NEG_MEAN, cov: own ? covFrom(0.55, 0.55, 0) : covFrom(len, 0.4, deg) };
  const { X, y } = sampleClasses(pos, neg, z.p, z.n);
  const fit = { ...fitGda(X, y), prior: [prior, 1 - prior] as [number, number] };
  const shift = Math.log(prior / (1 - prior));
  const lda = ldaLine(fit);
  const lr0 = useMemo(() => logisticLine(X, y), [own, deg, len]); // eslint-disable-line react-hooks/exhaustive-deps
  const lr = { w: lr0.w, b: lr0.b + shift };

  const grid = useMemo(() => {
    const n = 44;
    return { x: Array.from({ length: n }, (_, i) => +(-3 + (6 * i) / (n - 1)).toFixed(4)), y: Array.from({ length: n }, (_, i) => +(-2.6 + (5.2 * i) / (n - 1)).toFixed(4)) };
  }, []);
  const field = grid.y.flatMap((gy) => grid.x.map((gx) => +qdaLogOdds(own ? fit : { ...fit, cov: [fit.pooled, fit.pooled] }, [gx, gy]).toFixed(3)));

  // Test set drawn with the class proportions given by the prior.
  const nPos = Math.round(2000 * prior);
  const T = sampleClasses(pos, neg, z.tp.slice(0, nPos), z.tn.slice(0, 2000 - nPos));
  const lin = (l: { w: V2; b: number }) => (x: V2) => l.w[0] * x[0] + l.w[1] * x[1] + l.b;
  const eL = errorRate(lin(lda), T.X, T.y);
  const eQ = errorRate((x) => qdaLogOdds(fit, x), T.X, T.y);
  const eR = errorRate(lin(lr), T.X, T.y);

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="scenario">
      <div className="chips">
        <button className={`chip${!own ? " on" : ""}`} onClick={() => setOwn(false)}>{TEXT.shared[lang]}</button>
        <button className={`chip${own ? " on" : ""}`} onClick={() => setOwn(true)}>{TEXT.own[lang]}</button>
        <span className="gb-legend">{TEXT.legend[lang]}</span>
      </div>
      <Plot2D X={X} y={y} grid={grid} step={{ field }} levels={[1e-9, 2, 4]} showMargin={false} height={330} tweenMs={0}
        domain={{ x: [-3, 3], y: [-2.6, 2.6] }}
        extraLines={[{ w: lda.w, b: lda.b, color: "var(--accent)" }, { w: lr.w, b: lr.b, color: "var(--success)", dashed: true }]}
        overlay={(sx, sy) => (
          <g className="gb-ell">
            <path d={ellipse(pos, sx, sy)} className="pos" />
            <path d={ellipse(neg, sx, sy)} className="neg" />
          </g>
        )} />
      <Slider label={TEXT.angle[lang]} value={deg} min={0} max={180} step={1} onChange={setDeg} format={(v) => `${v}°`} />
      <Slider label={TEXT.stretch[lang]} value={len} min={0.4} max={1.6} step={0.01} onChange={setLen} format={(v) => v.toFixed(2)} />
      <Slider label={TEXT.prior[lang]} value={prior} min={0.1} max={0.9} step={0.01} onChange={setPrior} format={(v) => v.toFixed(2)} />
      <div className="w-stats">
        <Stat label={TEXT.lda[lang]} value={`${(100 * eL).toFixed(1)}%`} />
        <Stat label={TEXT.qda[lang]} value={`${(100 * eQ).toFixed(1)}%`} tone="accent" />
        <Stat label={TEXT.lr[lang]} value={`${(100 * eR).toFixed(1)}%`} />
      </div>
    </Widget>
  );
}
