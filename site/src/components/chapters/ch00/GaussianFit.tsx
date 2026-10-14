import { scaleLinear } from "d3-scale";
import { useMemo, useRef, useState } from "react";
import { gaussian, r6, rng } from "../../../engines/data";
import { eigSym, gaussianLogLik, gaussianMle, type Mat, type Vec } from "../../../engines/linalg";
import type { Lang } from "../../../i18n/ui";
import Widget, { Stat } from "../../ui/Widget";

const TEXT = {
  title: { zh: "用高斯分布描述一团数据", en: "Describe a cloud of data with a Gaussian" },
  axes: { zh: "显示主方向", en: "Show principal axes" },
  reset: { zh: "重置", en: "Reset" },
  clear: { zh: "清空", en: "Clear" },
  ll: { zh: "平均对数似然", en: "Average log-likelihood" },
  n: { zh: "样本数", en: "Samples" },
  rho: { zh: "相关系数", en: "Correlation" },
  hx: { zh: "身高（标准化）", en: "Height (standardized)" },
  hy: { zh: "体重（标准化）", en: "Weight (standardized)" },
  caption: {
    zh: "点击空白处添加样本。椭圆是极大似然高斯的 1σ 与 2σ 等高线：均值是样本平均，协方差是平均外积。箭头是协方差矩阵的特征向量，长度正比于 √特征值——数据“最舒展”的方向。数据为合成示意。",
    en: "Click empty space to add samples. The ellipses are the 1σ and 2σ contours of the maximum-likelihood Gaussian: the mean is the sample average and the covariance the average outer product. Arrows are the covariance eigenvectors, scaled by √eigenvalue — the directions in which the data spreads most. Synthetic illustrative data.",
  },
};

const W = 460;
const H = 380;

function initial(): number[][] {
  const rand = rng(12);
  return Array.from({ length: 40 }, () => {
    const z1 = gaussian(rand);
    const z2 = gaussian(rand);
    return [r6(0.9 * z1), r6(0.65 * z1 + 0.45 * z2)];
  });
}

const ellipse = (mean: Vec, cov: Mat, k: number, sx: (v: number) => number, sy: (v: number) => number) => {
  const { values, vectors } = eigSym(cov);
  const a = k * Math.sqrt(Math.max(values[0], 0));
  const b = k * Math.sqrt(Math.max(values[1], 0));
  return "M" + Array.from({ length: 73 }, (_, i) => {
    const t = (i / 72) * 2 * Math.PI;
    const x = mean[0] + a * Math.cos(t) * vectors[0][0] + b * Math.sin(t) * vectors[1][0];
    const y = mean[1] + a * Math.cos(t) * vectors[0][1] + b * Math.sin(t) * vectors[1][1];
    return `${sx(x).toFixed(1)},${sy(y).toFixed(1)}`;
  }).join("L");
};

export default function GaussianFit({ lang }: { lang: Lang }) {
  const [X, setX] = useState<number[][]>(initial);
  const [axes, setAxes] = useState(true);
  const svg = useRef<SVGSVGElement>(null);
  const sx = scaleLinear().domain([-3, 3]).range([0, W]);
  const sy = scaleLinear().domain([-2.5, 2.5]).range([H, 0]);

  const fit = useMemo(() => (X.length >= 3 ? gaussianMle(X) : null), [X]);
  const eig = fit ? eigSym(fit.cov) : null;
  const ll = fit ? gaussianLogLik(X, fit.mean, fit.cov) : NaN;
  const rho = fit ? fit.cov[0][1] / Math.sqrt(fit.cov[0][0] * fit.cov[1][1]) : NaN;

  const add = (e: React.MouseEvent) => {
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.current!.getScreenCTM()!.inverse());
    setX((cur) => [...cur, [sx.invert(p.x), sy.invert(p.y)]]);
  };

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <svg ref={svg} className="gf-svg" viewBox={`0 0 ${W} ${H}`} role="img" onClick={add}>
        <rect width={W} height={H} rx={12} className="plot-bg" />
        <line x1={0} x2={W} y1={sy(0)} y2={sy(0)} className="lm-axis" />
        <line x1={sx(0)} x2={sx(0)} y1={0} y2={H} className="lm-axis" />
        <text x={W - 8} y={sy(0) - 6} textAnchor="end" className="gf-label">{TEXT.hx[lang]}</text>
        <text x={sx(0) + 6} y={14} className="gf-label">{TEXT.hy[lang]}</text>
        {fit && (
          <g>
            <path d={ellipse(fit.mean, fit.cov, 2, sx, sy)} className="gf-ell2" />
            <path d={ellipse(fit.mean, fit.cov, 1, sx, sy)} className="gf-ell1" />
          </g>
        )}
        {X.map((p, i) => <circle key={i} cx={sx(p[0]).toFixed(1)} cy={sy(p[1]).toFixed(1)} r={4} className="gf-pt" />)}
        {fit && axes && eig && eig.vectors.map((v, k) => {
          const L = 2 * Math.sqrt(Math.max(eig.values[k], 0));
          return (
            <line key={k} x1={sx(fit.mean[0])} y1={sy(fit.mean[1])} x2={sx(fit.mean[0] + L * v[0])} y2={sy(fit.mean[1] + L * v[1])}
              className={`gf-axis k${k}`} markerEnd="url(#gf-arrow)" />
          );
        })}
        {fit && <circle cx={sx(fit.mean[0])} cy={sy(fit.mean[1])} r={6} className="gf-mean" />}
        <defs>
          <marker id="gf-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0L10,5L0,10z" className="gf-arrowhead" />
          </marker>
        </defs>
      </svg>
      <div className="w-row">
        <label className="chip"><input type="checkbox" checked={axes} onChange={(e) => setAxes(e.target.checked)} /> {TEXT.axes[lang]}</label>
        <button className="btn" onClick={() => setX(initial())}>{TEXT.reset[lang]}</button>
        <button className="btn" onClick={() => setX([])}>{TEXT.clear[lang]}</button>
      </div>
      <div className="w-stats">
        <Stat label={TEXT.n[lang]} value={X.length} />
        <Stat label={TEXT.rho[lang]} value={Number.isFinite(rho) ? rho.toFixed(2) : "—"} />
        <Stat label={TEXT.ll[lang]} value={Number.isFinite(ll) ? ll.toFixed(3) : "—"} tone="accent" />
      </div>
    </Widget>
  );
}
