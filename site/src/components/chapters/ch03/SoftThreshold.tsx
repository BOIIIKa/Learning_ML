import { contours } from "d3-contour";
import { scaleLinear } from "d3-scale";
import { useMemo, useState } from "react";
import { lassoCD2, quadForm, quadLoss, softThreshold, type Vec2 } from "../../../engines/linreg";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch03.css";

const TEXT = {
  title: { zh: "软阈值与坐标下降", en: "Soft thresholding and coordinate descent" },
  lam: { zh: "λ", en: "λ" },
  z: { zh: "最小二乘系数 z", en: "Least-squares coef z" },
  steps: { zh: "坐标更新次数", en: "Coordinate updates" },
  map: { zh: "一维：z ↦ 收缩后的系数", en: "1-D: z ↦ shrunken coefficient" },
  cd: { zh: "二维：Lasso 目标的坐标下降", en: "2-D: coordinate descent on the lasso objective" },
  ridge: { zh: "岭回归 z/(1+λ)", en: "Ridge z/(1+λ)" },
  lasso: { zh: "Lasso S_λ(z)", en: "Lasso S_λ(z)" },
  w: { zh: "当前 w", en: "Current w" },
  obj: { zh: "目标值", en: "Objective" },
  caption: {
    zh: "左图：设计矩阵的列正交且已标准化时，最小二乘系数为 z，岭回归把它按比例缩小为 z/(1+λ)，永远不会等于 0；Lasso 用软阈值 S_λ(z) = sign(z)·max(|z|−λ, 0)：先整体平移 λ，|z| ≤ λ 的部分（黄色死区）直接变成 0。右图：列不正交时没有闭式解，但每次只动一个坐标、固定其他坐标，就又变成一维问题，用同一个软阈值就能精确求解。紫色是 ½(w−c)ᵀA(w−c) + λ‖w‖₁ 的等高线，黄色折线是坐标下降的轨迹（只沿坐标轴方向走），虚线圈是最优点。λ 足够大时，w₂ 一步就被压到 0 并停在那里。",
    en: "Left: when the design columns are orthonormal (after scaling), the least-squares coefficient is z. Ridge scales it to z/(1+λ), which is never exactly 0; lasso applies soft thresholding S_λ(z) = sign(z)·max(|z|−λ, 0): shift towards 0 by λ, and everything with |z| ≤ λ (the yellow dead zone) becomes exactly 0. Right: with correlated columns there is no closed form, but moving one coordinate at a time with the others fixed gives a 1-D problem again, solved exactly by the same soft threshold. Purple lines are contours of ½(w−c)ᵀA(w−c) + λ‖w‖₁, the yellow polyline is the coordinate-descent path (it only moves parallel to the axes), and the dashed ring is the optimum. With λ large enough, w₂ is pushed to 0 in one step and stays there.",
  },
};

const W = 300;
const H = 260;
const f1 = (v: number) => v.toFixed(1);
const A = quadForm(0.6, 1.5);
const C: Vec2 = [1.5, 0.35];
const XD: [number, number] = [-0.5, 2.1];
const YD: [number, number] = [-0.8, 1.4];
const NG = 52;

export default function SoftThreshold({ lang }: { lang: Lang }) {
  const [lam, setLam] = useState(0.5);
  const [z, setZ] = useState(1.6);
  const [k, setK] = useState(2);

  // Left: shrinkage maps.
  const mx = scaleLinear().domain([-3, 3]).range([24, W - 8]);
  const my = scaleLinear().domain([-3, 3]).range([H - 20, 8]);
  const zs = Array.from({ length: 121 }, (_, i) => -3 + i * 0.05);
  const curve = (g: (v: number) => number) => "M" + zs.map((v) => `${f1(mx(v))},${f1(my(g(v)))}`).join("L");

  // Right: coordinate descent on a correlated 2-D lasso.
  const cx = scaleLinear().domain(XD).range([0, W]);
  const cy = scaleLinear().domain(YD).range([H, 0]);
  const obj = (w: Vec2) => quadLoss(A, C, w) + lam * (Math.abs(w[0]) + Math.abs(w[1]));
  const path = useMemo(() => lassoCD2(A, C, lam, 30), [lam]);
  const opt = path.at(-1)!;
  const shown = path.slice(0, k + 1);
  const levels = useMemo(() => {
    const gx = Array.from({ length: NG }, (_, i) => XD[0] + ((XD[1] - XD[0]) * i) / (NG - 1));
    const gy = Array.from({ length: NG }, (_, i) => YD[0] + ((YD[1] - YD[0]) * i) / (NG - 1));
    const field: number[] = [];
    for (const yy of gy) for (const xx of gx) field.push(obj([xx, yy]));
    const base = obj(opt);
    const th = [0.02, 0.08, 0.2, 0.4, 0.7, 1.1].map((d) => base + d);
    const dx = (XD[1] - XD[0]) / (NG - 1);
    const dy = (YD[1] - YD[0]) / (NG - 1);
    return contours().size([NG, NG]).thresholds(th)(field).map((m) =>
      m.coordinates.flatMap((poly) => poly.map((ring) => "M" + ring.map(([a, b]) => `${f1(cx(XD[0] + (a - 0.5) * dx))},${f1(cy(YD[0] + (b - 0.5) * dy))}`).join("L") + "Z")).join(""));
  }, [lam]);

  const cur = shown.at(-1)!;
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="c3-pair">
        <figure className="c3-sub">
          <svg className="c3-svg" viewBox={`0 0 ${W} ${H}`} role="img">
            <rect width={W} height={H} rx={10} className="c3-bg" />
            <rect x={f1(mx(-lam))} y={8} width={f1(mx(lam) - mx(-lam))} height={H - 28} className="st-dead" />
            <line x1={mx(-3)} x2={mx(3)} y1={my(0)} y2={my(0)} className="c3-axis" />
            <line x1={mx(0)} x2={mx(0)} y1={8} y2={H - 20} className="c3-axis" />
            {[-2, -1, 1, 2].map((t) => <text key={t} x={mx(t)} y={H - 6} textAnchor="middle" className="c3-tick">{t}</text>)}
            <path d={curve((v) => v)} className="st-ols" />
            <path d={curve((v) => v / (1 + lam))} className="st-ridge" />
            <path d={curve((v) => softThreshold(v, lam))} className="st-lasso" />
            <line x1={f1(mx(z))} x2={f1(mx(z))} y1={8} y2={H - 20} className="st-zline" />
            <circle cx={f1(mx(z))} cy={f1(my(z / (1 + lam)))} r={5} className="st-dot ridge" />
            <circle cx={f1(mx(z))} cy={f1(my(softThreshold(z, lam)))} r={5} className="st-dot lasso" />
            <text x={mx(-2.9)} y={20} className="c3-lab small">w</text>
            <text x={mx(2.9)} y={my(0) + 14} textAnchor="end" className="c3-lab small">z</text>
          </svg>
          <figcaption className="rp-legend">
            <span><i style={{ background: "var(--class-neg)" }} />{TEXT.ridge[lang]}</span>
            <span><i style={{ background: "var(--class-pos)" }} />{TEXT.lasso[lang]}</span>
          </figcaption>
        </figure>
        <figure className="c3-sub">
          <svg className="c3-svg" viewBox={`0 0 ${W} ${H}`} role="img">
            <rect width={W} height={H} rx={10} className="c3-bg" />
            <line x1={0} x2={W} y1={cy(0)} y2={cy(0)} className="c3-axis" />
            <line x1={cx(0)} x2={cx(0)} y1={0} y2={H} className="c3-axis" />
            {levels.map((d, i) => <path key={i} d={d} className="cd-level" />)}
            <polyline points={shown.map((p) => `${f1(cx(p[0]))},${f1(cy(p[1]))}`).join(" ")} className="cd-path" />
            {shown.map((p, i) => <circle key={i} cx={f1(cx(p[0]))} cy={f1(cy(p[1]))} r={2.6} className="cd-pt" />)}
            <circle cx={f1(cx(opt[0]))} cy={f1(cy(opt[1]))} r={9} className="cd-opt" />
            <circle cx={f1(cx(cur[0]))} cy={f1(cy(cur[1]))} r={5.5} className="cd-now" />
            <text x={W - 8} y={cy(0) - 6} textAnchor="end" className="c3-lab muted small">w₁</text>
            <text x={cx(0) + 6} y={14} className="c3-lab muted small">w₂</text>
          </svg>
          <figcaption>{TEXT.cd[lang]}</figcaption>
        </figure>
      </div>
      <div className="w-row">
        <Slider label={TEXT.lam[lang]} value={lam} min={0} max={2} step={0.01} onChange={setLam} format={(v) => v.toFixed(2)} />
        <Slider label={TEXT.z[lang]} value={z} min={-3} max={3} step={0.01} onChange={setZ} format={(v) => v.toFixed(2)} />
        <Slider label={TEXT.steps[lang]} value={k} min={0} max={12} step={1} onChange={setK} format={(v) => String(v)} />
      </div>
      <div className="w-stats">
        <Stat label={`${TEXT.ridge[lang]}`} value={(z / (1 + lam)).toFixed(3)} />
        <Stat label={`${TEXT.lasso[lang]}`} value={softThreshold(z, lam).toFixed(3)} tone={softThreshold(z, lam) === 0 ? "accent" : undefined} />
        <Stat label={TEXT.w[lang]} value={`(${cur[0].toFixed(3)}, ${cur[1].toFixed(3)})`} />
        <Stat label={TEXT.obj[lang]} value={obj(cur).toFixed(4)} />
      </div>
    </Widget>
  );
}
