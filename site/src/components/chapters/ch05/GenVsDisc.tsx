import { useMemo, useState } from "react";
import { learningCurves, SIZES } from "../../../engines/generative-classifiers";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch05.css";

const TEXT = {
  title: { zh: "样本效率：生成式先跑，判别式后追", en: "Sample efficiency: generative starts fast, discriminative catches up" },
  indep: { zh: "特征条件独立", en: "Independent features" },
  dup: { zh: "混入两个重复特征", en: "Two duplicated features" },
  n: { zh: "训练样本数 n", en: "Training size n" },
  nb: { zh: "朴素贝叶斯误差", en: "Naive Bayes error" },
  lr: { zh: "逻辑回归误差", en: "Logistic error" },
  x: { zh: "log₂ n", en: "log₂ n" },
  nbName: { zh: "朴素贝叶斯", en: "naive Bayes" },
  lrName: { zh: "逻辑回归", en: "logistic" },
  caption: {
    zh: "10 个有用的特征，测试误差随训练样本数的变化（每个 n 重复 16 次取平均）。朴素贝叶斯每个特征只估计均值和方差，几十个样本就站稳了；逻辑回归要同时估计 11 个相互牵连的系数，起步更慢。特征真的条件独立时，两者最终一样好。混入第一个特征的两个带噪副本后，朴素贝叶斯把同一份证据数了三遍，停在更高的误差上；逻辑回归自己学会给副本小权重，样本多了就反超——两条曲线交叉。数据为合成示意。",
    en: "Ten useful features; test error versus training size (16 repetitions per n). Naive Bayes estimates only a mean and a variance per feature and settles within a few dozen samples; logistic regression must estimate 11 coupled weights and starts slower. When the features really are conditionally independent, both end equally good. Add two noisy copies of the first feature and naive Bayes counts the same evidence three times and plateaus higher, while logistic regression learns small weights for the copies and overtakes it given enough data: the curves cross. Synthetic illustrative data.",
  },
};

const L = SIZES.map((n) => Math.log2(n));
const interp = (vals: number[]) => (x: number) => {
  if (x <= L[0]) return vals[0];
  for (let i = 1; i < L.length; i++) if (x <= L[i]) return vals[i - 1] + ((vals[i] - vals[i - 1]) * (x - L[i - 1])) / (L[i] - L[i - 1]);
  return vals.at(-1)!;
};

export default function GenVsDisc({ lang }: { lang: Lang }) {
  const [mode, setMode] = useState(1);
  const [k, setK] = useState(SIZES.length - 1);
  const curves = useMemo(() => learningCurves(mode === 0 ? 0 : 2), [mode]);
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="chips">
        {[TEXT.indep, TEXT.dup].map((t, i) => (
          <button key={i} className={`chip${mode === i ? " on" : ""}`} onClick={() => setMode(i)}>{t[lang]}</button>
        ))}
        <span className="gb-legend"><b className="gv-nb">━</b> {TEXT.nbName[lang]} · <b className="gv-lr">━</b> {TEXT.lrName[lang]}</span>
      </div>
      <CurvePlot xs={[...L, ...L]} ys={[...curves.nb, ...curves.lr]} xDomain={[2.8, 9.2]} yDomain={[0.08, 0.32]} height={280} xLabel={TEXT.x[lang]}
        pointColors={[...L.map(() => "var(--accent)"), ...L.map(() => "var(--accent-2)")]}
        curves={[{ f: interp(curves.nb), color: "var(--accent)", label: "NB" }, { f: interp(curves.lr), color: "var(--accent-2)", label: "LR" }]}
        vlines={[{ x: L[k], color: "var(--ink)" }]} />
      <Slider label={TEXT.n[lang]} value={k} min={0} max={SIZES.length - 1} step={1} onChange={setK} format={(v) => `${SIZES[v]}`} />
      <div className="w-stats">
        <Stat label={TEXT.nb[lang]} value={`${(100 * curves.nb[k]).toFixed(1)}%`} tone={curves.nb[k] <= curves.lr[k] ? "good" : undefined} />
        <Stat label={TEXT.lr[lang]} value={`${(100 * curves.lr[k]).toFixed(1)}%`} tone={curves.lr[k] < curves.nb[k] ? "good" : undefined} />
      </div>
    </Widget>
  );
}
