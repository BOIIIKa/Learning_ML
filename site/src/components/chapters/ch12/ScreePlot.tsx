import { useMemo, useState } from "react";
import { panel, pca } from "../../../engines/matrix-factorization";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "../ch10/ch10.css";
import "./ch12.css";

const TEXT = {
  title: { zh: "碎石图：保留几个主成分？", en: "The scree plot: how many components?" },
  k: { zh: "主成分个数", en: "Components kept" },
  cum: { zh: "累计解释方差", en: "Cumulative explained variance" },
  proj: { zh: "前两个主成分上的投影", en: "Projection on the first two components" },
  caption: {
    zh: "120 位病人的 8 项化验指标，其实由两个潜在因素（加噪声）驱动，病人分属三个亚型。左：每个主成分解释的方差（柱）与累计比例（折线）——前两个主成分之后断崖式下跌，像山脚的碎石，后面几乎全是噪声。右：把 8 维数据投影到前两个主成分上，三个亚型一目了然。PCA 本身不用任何标签，颜色只是为了看。数据为合成示意。",
    en: "Eight lab markers for 120 patients, actually driven by two latent factors (plus noise), with three subtypes. Left: the variance explained by each component (bars) and the cumulative share (line); after the first two it drops off a cliff, like scree at the foot of a slope, and the rest is mostly noise. Right: projecting the 8-dimensional data on the first two components makes the three subtypes plain to see. PCA uses no labels at all; the colours are only for looking. Synthetic illustrative data.",
  },
};

const f1 = (v: number) => v.toFixed(1);

export default function ScreePlot({ lang }: { lang: Lang }) {
  const { X, g } = useMemo(() => panel(), []);
  const { values, scores } = useMemo(() => pca(X), [X]);
  const [k, setK] = useState(2);
  const tot = values.reduce((s, v) => s + v, 0);
  const cum = values.map((_, i) => values.slice(0, i + 1).reduce((s, v) => s + v, 0) / tot);
  const bx = (i: number) => 24 + i * 24;
  const by = (v: number) => 180 - v * 160;
  const sc = 36 / Math.max(...scores.map((p) => Math.max(Math.abs(p[0]), Math.abs(p[1]))));
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="lr-row">
        <svg className="lm-svg" viewBox="0 0 220 200" role="img">
          {values.map((v, i) => <rect key={i} x={f1(bx(i))} y={f1(by(v / tot))} width={18} height={f1(180 - by(v / tot))} className={i < k ? "lr-sv on" : "lr-sv"} />)}
          <path d={"M" + cum.map((c, i) => `${f1(bx(i) + 9)},${f1(by(c))}`).join("L")} className="sc-cum" />
          {cum.map((c, i) => <circle key={i} cx={f1(bx(i) + 9)} cy={f1(by(c))} r={3} className="sc-dot" />)}
          <line x1={20} x2={215} y1={180} y2={180} className="cp-grid" />
        </svg>
        <svg className="lm-svg" viewBox="0 0 220 200" role="img">
          <text x={6} y={14} className="pc-cap">{TEXT.proj[lang]}</text>
          {scores.map((p, i) => <circle key={i} cx={f1(110 + p[0] * sc * 2.4)} cy={f1(105 - p[1] * sc * 2.2)} r={3.2} className={`km-pt c${g[i]}`} />)}
        </svg>
      </div>
      <Slider label={TEXT.k[lang]} value={k} min={1} max={values.length} step={1} onChange={setK} />
      <div className="w-stats"><Stat label={TEXT.cum[lang]} value={`${(100 * cum[k - 1]).toFixed(1)}%`} tone="accent" /></div>
    </Widget>
  );
}
