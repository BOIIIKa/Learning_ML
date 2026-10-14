import { useEffect, useMemo, useRef, useState } from "react";
import { correlated, principalAngle, projStats } from "../../../engines/matrix-factorization";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch12.css";

const TEXT = {
  title: { zh: "寻找主方向：方差最大 = 重构误差最小", en: "Finding the principal direction: max variance = min reconstruction error" },
  angle: { zh: "投影方向", en: "Projection direction" },
  snap: { zh: "吸附到主方向", en: "Snap to the principal direction" },
  proj: { zh: "投影方差", en: "Projected variance" },
  resid: { zh: "重构误差", en: "Reconstruction error" },
  total: { zh: "两者之和（总方差）", en: "Sum (total variance)" },
  caption: {
    zh: "60 位病人的两项指标高度相关（已中心化）。把每个点投影到一条过原点的直线上（短线是投影的“残差”）。转动直线：投影后点的方差（黄色）与残差平方的平均（红色）此消彼长，二者之和恒等于总方差——勾股定理。所以“让投影尽量分散”和“让重构尽量准确”是同一个问题，答案都是协方差矩阵最大特征值对应的特征向量。点按钮吸附过去。数据为合成示意。",
    en: "Two strongly correlated markers for 60 patients (centred). Project every point onto a line through the origin (the short segments are the residuals). Rotate the line: the variance of the projected points (yellow) and the mean squared residual (red) trade off, and their sum is always the total variance, by Pythagoras. So \"spread the projections as much as possible\" and \"reconstruct as accurately as possible\" are the same problem, solved by the eigenvector of the covariance matrix with the largest eigenvalue. Press the button to snap to it. Synthetic illustrative data.",
  },
};

const f1 = (v: number) => v.toFixed(1);
const S = 62;
const sx = (x: number) => 230 + x * S;
const sy = (y: number) => 150 - y * S;

export default function PcaRotate({ lang }: { lang: Lang }) {
  const P = useMemo(() => correlated(), []);
  const best = useMemo(() => Math.round((principalAngle(P) * 180) / Math.PI), [P]);
  const [deg, setDeg] = useState(-40);
  const timer = useRef<number | null>(null);
  useEffect(() => () => { if (timer.current) window.clearInterval(timer.current); }, []);
  const t = (deg * Math.PI) / 180;
  const u = [Math.cos(t), Math.sin(t)];
  const st = projStats(P, t);
  const snap = () => {
    if (timer.current) window.clearInterval(timer.current);
    timer.current = window.setInterval(() => setDeg((d) => {
      if (d === best) { window.clearInterval(timer.current!); timer.current = null; return d; }
      return d + Math.sign(best - d) * Math.min(3, Math.abs(best - d));
    }), 30);
  };
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="scenario">
      <svg className="lm-svg pr-svg" viewBox="0 0 460 300" role="img">
        <line x1={f1(sx(-4 * u[0]))} y1={f1(sy(-4 * u[1]))} x2={f1(sx(4 * u[0]))} y2={f1(sy(4 * u[1]))} className="pr-axis" />
        {P.map((p, i) => {
          const z = p[0] * u[0] + p[1] * u[1];
          return (
            <g key={i}>
              <line x1={f1(sx(p[0]))} y1={f1(sy(p[1]))} x2={f1(sx(z * u[0]))} y2={f1(sy(z * u[1]))} className="pr-res" />
              <circle cx={f1(sx(z * u[0]))} cy={f1(sy(z * u[1]))} r={2.5} className="pr-proj" />
              <circle cx={f1(sx(p[0]))} cy={f1(sy(p[1]))} r={3.5} className="pr-pt" />
            </g>
          );
        })}
      </svg>
      <div className="pr-bar" aria-hidden="true">
        <span className="pr-bar-p" style={{ width: `${(100 * st.proj / st.total).toFixed(1)}%` }} />
      </div>
      <Slider label={TEXT.angle[lang]} value={deg} min={-90} max={90} step={1} onChange={setDeg} format={(v) => `${v}°`} />
      <div className="pr-foot">
        <button className="btn btn-primary" onClick={snap}>⤓ {TEXT.snap[lang]}</button>
        <div className="w-stats">
          <Stat label={TEXT.proj[lang]} value={st.proj.toFixed(3)} tone="accent" />
          <Stat label={TEXT.resid[lang]} value={st.resid.toFixed(3)} tone="bad" />
          <Stat label={TEXT.total[lang]} value={st.total.toFixed(3)} />
        </div>
      </div>
    </Widget>
  );
}
