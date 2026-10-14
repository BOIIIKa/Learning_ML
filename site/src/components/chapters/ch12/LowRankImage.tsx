import { useEffect, useMemo, useRef, useState } from "react";
import { lowRank, scanImage, svd } from "../../../engines/matrix-factorization";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch12.css";

const TEXT = {
  title: { zh: "低秩近似：几个奇异值就够了", en: "Low-rank approximation: a few singular values suffice" },
  k: { zh: "保留的秩 k", en: "Rank kept k" },
  err: { zh: "相对误差 ‖A − A_k‖/‖A‖", en: "Relative error ‖A − A_k‖/‖A‖" },
  comp: { zh: "存储量 / 原图", en: "Storage / original" },
  spec: { zh: "奇异值谱", en: "Singular values" },
  caption: {
    zh: "一张 40 × 56 的“扫描图”（程序生成）。把它看成矩阵 A，做 SVD：A = Σ σ_j u_j v_jᵀ，每一项是一张“秩一图像”。只保留前 k 项就是截断 SVD——Eckart–Young 定理保证，在所有秩不超过 k 的矩阵中，它的 Frobenius 误差最小，误差恰为被丢掉的奇异值平方和的平方根。右边的奇异值谱下降得很快，所以 k 不大时图像已经清晰，而存储只需 k(m + n + 1) 个数。",
    en: "A 40 × 56 \"scan\" (procedurally generated). Treat it as a matrix A and take its SVD: A = Σ σ_j u_j v_jᵀ, each term a rank-one image. Keeping only the first k terms is the truncated SVD, and the Eckart–Young theorem guarantees that among all matrices of rank at most k it has the smallest Frobenius error, which is exactly the square root of the sum of the discarded squared singular values. The spectrum on the right falls fast, so the picture is already clear for small k while storage needs only k(m + n + 1) numbers.",
  },
};

export default function LowRankImage({ lang }: { lang: Lang }) {
  const A = useMemo(() => scanImage(), []);
  const d = useMemo(() => svd(A), [A]);
  const [k, setK] = useState(3);
  const canvas = useRef<HTMLCanvasElement>(null);
  const m = A.length, n = A[0].length;
  useEffect(() => {
    const c = canvas.current!;
    const ctx = c.getContext("2d")!;
    const R = lowRank(d, k);
    const img = ctx.createImageData(n, m);
    R.forEach((row, i) => row.forEach((v, j) => {
      const g = Math.round(255 * Math.max(0, Math.min(1, v)));
      const o = 4 * (i * n + j);
      img.data[o] = g; img.data[o + 1] = g; img.data[o + 2] = g; img.data[o + 3] = 255;
    }));
    ctx.putImageData(img, 0, 0);
  }, [d, k, m, n]);
  const tot = d.s.reduce((s, v) => s + v * v, 0);
  const err = Math.sqrt(d.s.slice(k).reduce((s, v) => s + v * v, 0) / tot);
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="lr-row">
        <canvas ref={canvas} width={n} height={m} className="lr-canvas" />
        <svg className="lm-svg" viewBox="0 0 200 140" role="img">
          <text x={4} y={12} className="pc-cap">{TEXT.spec[lang]}</text>
          {d.s.map((s, j) => {
            const h = (110 * s) / d.s[0];
            return <rect key={j} x={(4 + j * 4.8).toFixed(1)} y={(132 - h).toFixed(1)} width={3.6} height={h.toFixed(1)} className={j < k ? "lr-sv on" : "lr-sv"} />;
          })}
        </svg>
      </div>
      <Slider label={TEXT.k[lang]} value={k} min={1} max={m} step={1} onChange={setK} />
      <div className="w-stats">
        <Stat label={TEXT.err[lang]} value={`${(100 * err).toFixed(1)}%`} tone="accent" />
        <Stat label={TEXT.comp[lang]} value={`${((100 * k * (m + n + 1)) / (m * n)).toFixed(0)}%`} />
      </div>
    </Widget>
  );
}
