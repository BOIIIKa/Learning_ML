import { useEffect, useMemo, useRef, useState } from "react";
import {
  crossEntropy, fromPolar, gdIterations, newton, newtonIterations, overlapping, screening, sigmoid, toPolar, type Theta,
} from "../../../engines/linear-classification";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import Plot2D from "../../viz/Plot2D";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch04.css";

const TEXT = {
  title: { zh: "逻辑回归：把距离变成概率", en: "Logistic regression: turning distance into probability" },
  overlap: { zh: "重叠数据", en: "Overlapping data" },
  separable: { zh: "可分数据", en: "Separable data" },
  angle: { zh: "方向 θ", en: "Direction θ" },
  steep: { zh: "陡峭度 ‖w‖", en: "Steepness ‖w‖" },
  offset: { zh: "边界位置 c", en: "Boundary offset c" },
  fit: { zh: "牛顿法拟合", en: "Fit with Newton" },
  ce: { zh: "平均交叉熵", en: "Mean cross-entropy" },
  acc: { zh: "训练准确率", en: "Training accuracy" },
  iters: { zh: "收敛步数：牛顿 / 梯度下降", en: "Steps to converge: Newton / GD" },
  never: { zh: "不收敛：‖w‖ → ∞", en: "never: ‖w‖ → ∞" },
  side: { zh: "沿 w 方向的截面：P(复查 | x)", en: "Cross-section along w: P(follow-up | x)" },
  tAxis: { zh: "到边界的有向距离", en: "signed distance to the boundary" },
  caption: {
    zh: "左图的色带是 P(需要复查 | x) = σ(w·x + b) 的等高线：0.5（边界）、0.75、0.9。右图把所有病人投影到 w 方向上：横轴是到边界的有向距离，曲线是 sigmoid，竖线是每个人“没被说中”的概率 1 − p(真实标签)。交叉熵是这些概率取负对数后的平均。转动方向、移动边界、改变陡峭度，看交叉熵怎样变化；再点“牛顿法拟合”。切到可分数据再拟合：交叉熵可以无限接近 0，‖w‖ 一路增大，极大似然估计不存在。数据为合成示意。",
    en: "The bands on the left are contours of P(follow-up | x) = σ(w·x + b) at 0.5 (the boundary), 0.75 and 0.9. On the right every patient is projected onto the w direction: the horizontal axis is the signed distance to the boundary, the curve is the sigmoid, and each vertical stick is the probability 1 − p(true label) that the model gave to the wrong answer. Cross-entropy averages the negative logs of these probabilities. Rotate the direction, move the boundary, change the steepness and watch the cross-entropy; then press “Fit with Newton”. Switch to separable data and fit again: the cross-entropy creeps towards 0 while ‖w‖ keeps growing, so no maximum-likelihood estimate exists. Synthetic illustrative data.",
  },
};

const S_MAX = 12;
const LV = [1e-9, Math.log(3), Math.log(9)]; // logit of 0.5, 0.75, 0.9

export default function LogisticSurface({ lang }: { lang: Lang }) {
  const sets = useMemo(() => [overlapping(), screening()], []);
  const [ds, setDs] = useState(0);
  const { X, y } = sets[ds];
  const [deg, setDeg] = useState(20);
  const [s, setS] = useState(1);
  const [c, setC] = useState(0.8);
  const timer = useRef<number | null>(null);
  useEffect(() => () => { if (timer.current) window.clearInterval(timer.current); }, []);

  const theta: Theta = fromPolar(deg, s, c);
  const ce = crossEntropy(X, y, theta);
  const acc = X.reduce((a, x, i) => a + (y[i] * (theta[0] * x[0] + theta[1] * x[1] + theta[2]) > 0 ? 1 : 0), 0) / X.length;
  const iters = useMemo(() => sets.map((d, k) => (k === 0 ? `${newtonIterations(d.X, d.y)} / ${gdIterations(d.X, d.y, 0.5)}` : null)), [sets]);

  const grid = useMemo(() => {
    const all = sets.flatMap((d) => d.X);
    const lo = [0, 1].map((k) => Math.min(...all.map((p) => p[k])) - 0.5);
    const hi = [0, 1].map((k) => Math.max(...all.map((p) => p[k])) + 0.5);
    const n = 46;
    const axis = (k: number) => Array.from({ length: n }, (_, i) => +(lo[k] + ((hi[k] - lo[k]) * i) / (n - 1)).toFixed(4));
    return { x: axis(0), y: axis(1) };
  }, [sets]);
  const field = useMemo(() => grid.y.flatMap((gy) => grid.x.map((gx) => +(theta[0] * gx + theta[1] * gy + theta[2]).toFixed(4))), [grid, theta[0], theta[1], theta[2]]);

  // Cross-section: signed distance t_i = (w·x_i + b)/‖w‖, probability σ(s t).
  const u = [Math.cos((deg * Math.PI) / 180), Math.sin((deg * Math.PI) / 180)];
  const ts = X.map((x) => +(u[0] * x[0] + u[1] * x[1] - c).toFixed(4));
  const ys01 = y.map((v) => (v > 0 ? 1 : 0));

  const fit = () => {
    if (timer.current) window.clearInterval(timer.current);
    const path = newton(X, y, ds === 0 ? 12 : 30, 1e-10);
    let k = 1;
    timer.current = window.setInterval(() => {
      if (k >= path.length) { window.clearInterval(timer.current!); timer.current = null; return; }
      const p = toPolar(path[k]);
      setDeg(Math.round(((p.deg + 540) % 360) - 180));
      setS(+Math.min(S_MAX, p.s).toFixed(2));
      setC(+p.c.toFixed(2));
      k++;
    }, ds === 0 ? 420 : 160);
  };

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="chips">
        {[TEXT.overlap, TEXT.separable].map((t, i) => (
          <button key={i} className={`chip${ds === i ? " on" : ""}`} onClick={() => setDs(i)}>{t[lang]}</button>
        ))}
      </div>
      <div className="lg-panels">
        <Plot2D X={X} y={y} grid={grid} levels={LV} step={{ field }} showMargin={false} height={320} tweenMs={0} />
        <div>
          <div className="lg-side-title">{TEXT.side[lang]}</div>
          <CurvePlot xs={ts} ys={ys01} xDomain={[-3.5, 3.5]} yDomain={[-0.08, 1.08]} height={290}
            residualsTo={(t) => sigmoid(s * t)}
            pointColors={y.map((v) => (v > 0 ? "var(--class-pos)" : "var(--class-neg)"))}
            curves={[{ f: (t) => sigmoid(s * t), color: "var(--accent)" }]}
            vlines={[{ x: 0, color: "var(--text-muted)" }]} xLabel={TEXT.tAxis[lang]} />
        </div>
      </div>
      <Slider label={TEXT.angle[lang]} value={deg} min={-180} max={180} step={1} onChange={setDeg} format={(v) => `${v}°`} />
      <Slider label={TEXT.steep[lang]} value={s} min={0.1} max={S_MAX} step={0.01} onChange={setS} format={(v) => v.toFixed(2)} />
      <Slider label={TEXT.offset[lang]} value={c} min={-2} max={2} step={0.01} onChange={setC} format={(v) => v.toFixed(2)} />
      <div className="lg-foot">
        <button className="btn btn-primary" onClick={fit}>▶ {TEXT.fit[lang]}</button>
        <div className="w-stats">
          <Stat label={TEXT.ce[lang]} value={ce.toFixed(3)} tone="accent" />
          <Stat label={TEXT.acc[lang]} value={`${Math.round(100 * acc)}%`} />
          <Stat label={TEXT.iters[lang]} value={iters[ds] ?? TEXT.never[lang]} />
        </div>
      </div>
    </Widget>
  );
}
