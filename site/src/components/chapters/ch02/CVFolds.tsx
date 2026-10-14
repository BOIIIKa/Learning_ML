import { useEffect, useMemo, useState } from "react";
import { cvError, folds, polyFit, sample } from "../../../engines/generalization";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import MetricChart from "../../viz/MetricChart";
import Widget, { Slider, Stat } from "../../ui/Widget";

const TEXT = {
  title: { zh: "k 折交叉验证：轮流当“考官”", en: "k-fold cross-validation: take turns as the examiner" },
  k: { zh: "折数 k", en: "Folds k" },
  loo: { zh: "留一法", en: "Leave-one-out" },
  degree: { zh: "多项式次数", en: "Polynomial degree" },
  cv: { zh: "交叉验证误差", en: "CV error" },
  best: { zh: "CV 选出的次数", en: "Degree chosen by CV" },
  heldOut: { zh: "当前留出第 {f} 折", en: "Holding out fold {f}" },
  caption: {
    zh: "30 位学生被随机分成 k 折（颜色）。每一轮留出一折当“明年的学生”（大圆点），用其余数据拟合（橙线），记下留出折上的误差；k 轮之后取平均，就是 CV 误差。下方曲线对每个次数都做了一遍，最低点就是 CV 推荐的模型复杂度。",
    en: "Thirty students are randomly split into k folds (colours). Each round holds out one fold as “next year's students” (large dots), fits on the rest (orange) and records the error on the held-out fold; the average over the k rounds is the CV error. The chart repeats this for every degree, and its minimum is the complexity CV recommends.",
  },
};

const PALETTE = ["#4fd1ff", "#ff8a4c", "#b388ff", "#3ddc97", "#ffb547", "#ff5ca8", "#4f9dff", "#e0e0e0", "#9ae66e", "#f78fb3"];

export default function CVFolds({ lang }: { lang: Lang }) {
  const data = useMemo(() => sample(30, 2), []);
  const [k, setK] = useState(5);
  const [d, setD] = useState(3);
  const [cur, setCur] = useState(0);
  const fold = useMemo(() => folds(30, k), [k]);
  const kk = Math.min(k, 30);
  const cvs = useMemo(() => Array.from({ length: 11 }, (_, deg) => cvError(data.xs, data.ys, deg, fold, kk)), [data, fold, kk]);
  const best = cvs.reduce((bi, v, i) => (v < cvs[bi] ? i : bi), 0);

  useEffect(() => {
    setCur(0);
    const id = setInterval(() => setCur((c) => (c + 1) % kk), 1100);
    return () => clearInterval(id);
  }, [kk]);

  const trainIdx = data.xs.map((_, i) => i).filter((i) => fold[i] !== cur);
  const fit = useMemo(() => polyFit(trainIdx.map((i) => data.xs[i]), trainIdx.map((i) => data.ys[i]), d), [cur, d, fold]);
  const colors = fold.map((f) => (f === cur ? "var(--text)" : PALETTE[f % PALETTE.length]));

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="chips">
        {[2, 5, 10, 30].map((v) => (
          <button key={v} className={`chip${v === k ? " on" : ""}`} onClick={() => setK(v)}>{v === 30 ? `${TEXT.loo[lang]} (k=30)` : `k = ${v}`}</button>
        ))}
        <span className="muted">{TEXT.heldOut[lang].replace("{f}", String(cur + 1))}</span>
      </div>
      <CurvePlot xs={data.xs} ys={data.ys} pointColors={colors} xDomain={[0, 10]} yDomain={[0, 110]} height={270}
        curves={[{ f: fit, color: "var(--class-pos)" }]}
        extraPoints={{ xs: data.xs.filter((_, i) => fold[i] === cur), ys: data.ys.filter((_, i) => fold[i] === cur) }} />
      <Slider label={TEXT.degree[lang]} value={d} min={0} max={10} step={1} onChange={setD} />
      <MetricChart zero={false} cursor={d} height={120} series={[{ key: "cv", label: `log₁₀ ${TEXT.cv[lang]}`, values: cvs.map((v) => Math.log10(v)), color: "var(--accent)" }]} />
      <div className="w-stats">
        <Stat label={TEXT.cv[lang]} value={cvs[d].toFixed(1)} />
        <Stat label={TEXT.best[lang]} value={best} tone="accent" />
      </div>
    </Widget>
  );
}
