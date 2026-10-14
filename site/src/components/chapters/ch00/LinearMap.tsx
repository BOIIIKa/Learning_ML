import { scaleLinear } from "d3-scale";
import { useEffect, useRef, useState } from "react";
import { det, diag, lerpMat, matmul, mul, rot, svd2, type Mat, type Vec } from "../../../engines/linalg";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";

const TEXT = {
  title: { zh: "矩阵就是一个变换", en: "A matrix is a transformation" },
  play: { zh: "▶ 分解为 旋转 → 缩放 → 旋转", en: "▶ Decompose: rotate → scale → rotate" },
  stage: [
    { zh: "原始网格", en: "Original grid" },
    { zh: "① 旋转 Vᵀ", en: "① Rotate by Vᵀ" },
    { zh: "② 沿坐标轴缩放 Σ", en: "② Scale along the axes by Σ" },
    { zh: "③ 旋转 U", en: "③ Rotate by U" },
  ],
  det: { zh: "行列式（面积缩放）", en: "Determinant (area scale)" },
  sv: { zh: "奇异值 σ₁, σ₂", en: "Singular values σ₁, σ₂" },
  presets: { zh: "预设", en: "Presets" },
  caption: {
    zh: "拖动四个元素，看单位网格与单位圆怎样被变形：圆总是变成椭圆，椭圆的两个半轴长就是奇异值。点“分解”，任何矩阵都被拆成三个简单动作：先旋转、再沿坐标轴拉伸、最后再旋转——这就是 SVD 的几何含义。",
    en: "Drag the four entries and watch the unit grid and circle deform: a circle always becomes an ellipse whose semi-axes are the singular values. Press “decompose” and any matrix splits into three simple moves — rotate, stretch along the axes, rotate again. That is the geometry of the SVD.",
  },
};

const PRESETS: { name: { zh: string; en: string }; m: Mat }[] = [
  { name: { zh: "剪切", en: "Shear" }, m: [[1, 1], [0, 1]] },
  { name: { zh: "旋转 30°", en: "Rotate 30°" }, m: rot(Math.PI / 6) },
  { name: { zh: "反射", en: "Reflect" }, m: [[1, 0], [0, -1]] },
  { name: { zh: "秩 1（压扁）", en: "Rank 1 (flatten)" }, m: [[1, 0.5], [2, 1]] },
  { name: { zh: "协方差形状", en: "Covariance shape" }, m: [[1.4, 0.6], [0.6, 0.7]] },
];

const W = 420;
const H = 420;
const R = 3;

export default function LinearMap({ lang }: { lang: Lang }) {
  const [A, setA] = useState<Mat>([[1.2, 0.6], [0.2, 0.9]]);
  const [shown, setShown] = useState<Mat>(A);
  const [stage, setStage] = useState(-1); // −1: show A directly; 0..3: decomposition stages
  const frame = useRef(0);

  useEffect(() => {
    if (stage < 0) setShown(A);
  }, [A, stage]);

  const play = () => {
    cancelAnimationFrame(frame.current);
    const s = svd2(A);
    const I: Mat = [[1, 0], [0, 1]];
    const M1 = rot(-s.theta);
    const M2 = matmul(diag(s.sigma[0], s.sigma[1]), M1);
    const M3 = matmul(rot(s.phi), M2);
    const keys = [I, M1, M2, M3];
    const start = performance.now();
    const SEG = 1100;
    const tick = (now: number) => {
      const t = (now - start) / SEG;
      const k = Math.min(3, Math.floor(t));
      const f = Math.min(1, t - k);
      const ease = f < 0.5 ? 2 * f * f : 1 - (-2 * f + 2) ** 2 / 2;
      if (k >= 3) { setShown(M3); setStage(3); setTimeout(() => setStage(-1), 900); return; }
      setStage(k + 1);
      setShown(lerpMat(keys[k], keys[k + 1], ease));
      frame.current = requestAnimationFrame(tick);
    };
    setStage(0);
    setShown(I);
    frame.current = requestAnimationFrame(tick);
  };

  const sx = scaleLinear().domain([-R, R]).range([0, W]);
  const sy = scaleLinear().domain([-R, R]).range([H, 0]);
  const P = (v: Vec) => { const w = mul(shown, v); return `${sx(w[0]).toFixed(1)},${sy(w[1]).toFixed(1)}`; };
  const gridLines: [Vec, Vec][] = [];
  for (let k = -4; k <= 4; k++) { gridLines.push([[k, -4], [k, 4]]); gridLines.push([[-4, k], [4, k]]); }
  const circle = "M" + Array.from({ length: 73 }, (_, i) => P([Math.cos((i / 72) * 2 * Math.PI), Math.sin((i / 72) * 2 * Math.PI)])).join("L");
  const s = svd2(A);
  const set = (i: number, j: number, v: number) => { setStage(-1); setA((cur) => cur.map((row, r) => row.map((x, c) => (r === i && c === j ? v : x))) as Mat); };

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="split-2">
        <svg className="lm-svg" viewBox={`0 0 ${W} ${H}`} role="img">
          <rect width={W} height={H} rx={12} className="plot-bg" />
          <line x1={sx(-R)} x2={sx(R)} y1={sy(0)} y2={sy(0)} className="lm-axis" />
          <line x1={sx(0)} x2={sx(0)} y1={sy(-R)} y2={sy(R)} className="lm-axis" />
          {gridLines.map(([a, b], i) => <path key={i} d={`M${P(a)}L${P(b)}`} className="lm-grid" />)}
          <path d={circle} className="lm-circle" />
          <path d={`M${P([0, 0])}L${P([1, 0])}`} className="lm-e1" />
          <path d={`M${P([0, 0])}L${P([0, 1])}`} className="lm-e2" />
          <circle cx={sx(mul(shown, [1, 0])[0])} cy={sy(mul(shown, [1, 0])[1])} r={5} className="lm-e1-dot" />
          <circle cx={sx(mul(shown, [0, 1])[0])} cy={sy(mul(shown, [0, 1])[1])} r={5} className="lm-e2-dot" />
          {stage >= 0 && <text x={14} y={26} className="lm-stage">{TEXT.stage[stage][lang]}</text>}
        </svg>
        <div className="lm-controls">
          <div className="lm-matrix">
            <span className="lm-bracket">[</span>
            <div className="lm-cells">
              {[0, 1].map((i) => [0, 1].map((j) => (
                <Slider key={`${i}${j}`} label={`a${i + 1}${j + 1}`} value={A[i][j]} min={-2} max={2} step={0.05} onChange={(v) => set(i, j, v)} format={(v) => v.toFixed(2)} />
              )))}
            </div>
            <span className="lm-bracket">]</span>
          </div>
          <div className="chips">
            {PRESETS.map((p) => (
              <button key={p.name.en} className="chip" onClick={() => { setStage(-1); setA(p.m.map((r) => r.map((v) => +v.toFixed(3))) as Mat); }}>{p.name[lang]}</button>
            ))}
          </div>
          <button className="btn btn-primary" onClick={play}>{TEXT.play[lang]}</button>
          <div className="w-stats">
            <Stat label={TEXT.det[lang]} value={det(A).toFixed(2)} tone={Math.abs(det(A)) < 1e-6 ? "bad" : undefined} />
            <Stat label={TEXT.sv[lang]} value={`${s.sigma[0].toFixed(2)}, ${Math.abs(s.sigma[1]).toFixed(2)}`} tone="accent" />
          </div>
        </div>
      </div>
    </Widget>
  );
}
