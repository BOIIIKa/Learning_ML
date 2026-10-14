import { scaleLinear, scaleLog } from "d3-scale";
import { useMemo, useRef, useState } from "react";
import { learningCurves } from "../../../engines/generative";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch05.css";

const TEXT = {
  title: { zh: "样本效率：生成式 vs 判别式", en: "Sample efficiency: generative vs discriminative" },
  rho: { zh: "冗余特征的相关系数 ρ", en: "Correlation ρ of the redundant features" },
  m: { zh: "训练样本量 m", en: "Training size m" },
  nb: { zh: "朴素贝叶斯（生成式）", en: "Naive Bayes (generative)" },
  lr: { zh: "逻辑回归（判别式）", en: "Logistic regression (discriminative)" },
  nbInf: { zh: "朴素贝叶斯的极限", en: "naive Bayes limit" },
  bayes: { zh: "贝叶斯错误率", en: "Bayes error" },
  cross: { zh: "交叉点", en: "Crossing" },
  none: { zh: "无交叉", en: "none" },
  errNb: { zh: "朴素贝叶斯测试误差", en: "Naive Bayes test error" },
  errLr: { zh: "逻辑回归测试误差", en: "Logistic test error" },
  xl: { zh: "训练样本量 m（对数刻度）", en: "training size m (log scale)" },
  yl: { zh: "测试误差", en: "test error" },
  caption: {
    zh: "14 个特征的两类高斯数据，其中 4 个是特征 1 的“冗余副本”，彼此相关系数为 ρ。朴素贝叶斯（共享对角方差的高斯 NB）和逻辑回归属于同一个模型族——后验都是线性 sigmoid——只是参数的估计方式不同。每个点是 24 次随机训练集的平均测试误差（误差按真实分布精确计算）。ρ = 0 时独立性假设成立，朴素贝叶斯处处不差；ρ 增大后它会重复计算冗余证据，极限误差（橙色虚线）高于贝叶斯错误率，但在样本少时仍然收敛得快得多——两条曲线因此交叉。逻辑回归在 m 接近特征数时的“鼓包”来自几乎线性可分的数据。",
    en: "Two Gaussian classes with 14 features, 4 of which are “redundant copies” of feature 1, correlated with each other by ρ. Naive Bayes (Gaussian NB with a shared diagonal variance) and logistic regression belong to the same model family — both posteriors are linear sigmoids — and differ only in how the parameters are estimated. Each point averages the test error over 24 random training sets (errors are computed exactly from the true distribution). At ρ = 0 the independence assumption holds and naive Bayes is never worse; as ρ grows it double-counts the redundant evidence, so its limiting error (orange dashed) sits above the Bayes error, yet it still converges much faster with little data — so the curves cross. The bump in the logistic curve where m approaches the number of features comes from nearly separable training sets.",
  },
};

const W = 460;
const H = 300;
const PAD = { l: 44, r: 12, t: 12, b: 34 };
const f1 = (v: number) => v.toFixed(1);

export default function LearningCurves({ lang }: { lang: Lang }) {
  const [rho, setRho] = useState(0.6);
  const [mi, setMi] = useState(1);
  const cache = useRef(new Map<number, ReturnType<typeof learningCurves>>());
  const c = useMemo(() => {
    const key = Math.round(rho * 10);
    if (!cache.current.has(key)) cache.current.set(key, learningCurves(key / 10));
    return cache.current.get(key)!;
  }, [rho]);

  const sx = scaleLog().domain([c.sizes[0], c.sizes[c.sizes.length - 1]]).range([PAD.l, W - PAD.r]);
  const lo = Math.max(0, Math.floor((c.bayes - 0.03) * 20) / 20);
  const sy = scaleLinear().domain([lo, 0.4]).range([H - PAD.b, PAD.t]);
  const cy = (v: number) => Math.max(PAD.t, sy(v));
  const path = (v: number[]) => "M" + v.map((e, i) => `${f1(sx(c.sizes[i]))},${f1(cy(e))}`).join("L");

  // First crossing from "naive Bayes ahead" to "logistic ahead", interpolated in log m.
  const cross = useMemo(() => {
    for (let i = 1; i < c.sizes.length; i++) {
      const d0 = c.nb[i - 1] - c.lr[i - 1];
      const d1 = c.nb[i] - c.lr[i];
      if (d0 < 0 && d1 >= 0) {
        const t = d0 / (d0 - d1);
        const lm = Math.log(c.sizes[i - 1]) + t * (Math.log(c.sizes[i]) - Math.log(c.sizes[i - 1]));
        return { m: Math.exp(lm), e: c.nb[i - 1] + t * (c.nb[i] - c.nb[i - 1]) };
      }
    }
    return null;
  }, [c]);

  const m = c.sizes[mi];
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <svg className="g5-svg" viewBox={`0 0 ${W} ${H}`} role="img">
        <rect width={W} height={H} rx={12} className="g5-bg" />
        {sy.ticks(5).map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={W - PAD.r} y1={f1(sy(t))} y2={f1(sy(t))} className="g5-grid" />
            <text x={PAD.l - 6} y={f1(sy(t))} textAnchor="end" dominantBaseline="middle" className="g5-tick">{t.toFixed(2)}</text>
          </g>
        ))}
        {c.sizes.filter((s) => [12, 32, 64, 128, 256, 512].includes(s)).map((s) => (
          <text key={s} x={f1(sx(s))} y={H - PAD.b + 15} textAnchor="middle" className="g5-tick">{s}</text>
        ))}
        <text x={W - PAD.r} y={H - 4} textAnchor="end" className="g5-lab">{TEXT.xl[lang]}</text>
        <text x={W - PAD.r - 4} y={PAD.t + 12} textAnchor="end" className="g5-lab">{TEXT.yl[lang]}</text>
        <line x1={f1(sx(m))} x2={f1(sx(m))} y1={PAD.t} y2={H - PAD.b} className="g5-cursor" />
        <line x1={PAD.l} x2={W - PAD.r} y1={f1(sy(c.bayes))} y2={f1(sy(c.bayes))} className="g5-asym bayes" />
        <text x={W - PAD.r - 4} y={f1(sy(c.bayes) + 13)} textAnchor="end" className="g5-alab bayes">{TEXT.bayes[lang]} {c.bayes.toFixed(3)}</text>
        {c.nbInf - c.bayes > 0.004 && (
          <>
            <line x1={PAD.l} x2={W - PAD.r} y1={f1(sy(c.nbInf))} y2={f1(sy(c.nbInf))} className="g5-asym nb" />
            <text x={PAD.l + 6} y={f1(sy(c.nbInf) + 13)} className="g5-alab nb">{TEXT.nbInf[lang]} {c.nbInf.toFixed(3)}</text>
          </>
        )}
        <path d={path(c.lr)} className="g5-curve lr" />
        <path d={path(c.nb)} className="g5-curve nb" />
        {c.sizes.map((s, i) => (
          <g key={s}>
            <circle cx={f1(sx(s))} cy={f1(cy(c.lr[i]))} r={3} className="g5-dot lr" />
            <circle cx={f1(sx(s))} cy={f1(cy(c.nb[i]))} r={3} className="g5-dot nb" />
          </g>
        ))}
        {cross && <circle cx={f1(sx(cross.m))} cy={f1(sy(cross.e))} r={7} className="g5-cross" />}
      </svg>
      <div className="g5-legend">
        <span><i className="nb" />{TEXT.nb[lang]}</span>
        <span><i className="lrc" />{TEXT.lr[lang]}</span>
      </div>
      <div className="w-row">
        <Slider label={TEXT.rho[lang]} value={rho} min={0} max={0.9} step={0.1} onChange={setRho} format={(v) => v.toFixed(1)} />
        <Slider label={TEXT.m[lang]} value={mi} min={0} max={c.sizes.length - 1} step={1} onChange={setMi} format={() => String(m)} />
      </div>
      <div className="w-stats">
        <Stat label={`${TEXT.errNb[lang]} (m = ${m})`} value={c.nb[mi].toFixed(3)} tone={c.nb[mi] <= c.lr[mi] ? "good" : undefined} />
        <Stat label={`${TEXT.errLr[lang]} (m = ${m})`} value={c.lr[mi].toFixed(3)} tone={c.lr[mi] < c.nb[mi] ? "good" : undefined} />
        <Stat label={TEXT.cross[lang]} value={cross ? `m ≈ ${Math.round(cross.m)}` : TEXT.none[lang]} tone="accent" />
      </div>
    </Widget>
  );
}
