import { useMemo, useState } from "react";
import { chooseK, subgroups } from "../../../engines/clustering";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch10.css";

const TEXT = {
  title: { zh: "k 选多少？", en: "How many clusters?" },
  k: { zh: "簇数 k", en: "Clusters k" },
  w: { zh: "簇内平方和（肘部法）", en: "Within-cluster SS (elbow)" },
  s: { zh: "平均轮廓系数", en: "Mean silhouette" },
  caption: {
    zh: "同样的 90 位病人，对每个 k 跑 5 次 k-means++ 取最好。左：簇内平方和随 k 单调下降（k = n 时为 0），不能直接最小化，只能找“肘部”——再加一个簇收益突然变小的地方。右：轮廓系数比较每个点到本簇与到最近他簇的平均距离，(b − a)/max(a, b)，越接近 1 分得越清楚；它在 k = 3 处最高，正好是生成数据时的亚型数。真实问题里两者常常不一致，还要结合领域知识。",
    en: "The same 90 patients; for each k, the best of 5 k-means++ runs. Left: the within-cluster sum of squares falls with every k (to 0 at k = n), so it cannot simply be minimized; look for the \"elbow\" where one more cluster suddenly helps much less. Right: the silhouette compares each point's mean distance to its own cluster with that to the nearest other cluster, (b − a)/max(a, b), and is closer to 1 for cleaner splits; it peaks at k = 3, the number of subtypes used to generate the data. On real problems the two often disagree, and domain knowledge has the last word.",
  },
};

const KS = [1, 2, 3, 4, 5, 6, 7, 8];

export default function ChooseK({ lang }: { lang: Lang }) {
  const { X } = useMemo(() => subgroups(), []);
  const { wcss, sil } = useMemo(() => chooseK(X, KS), [X]);
  const [k, setK] = useState(3);
  const lin = (v: number[]) => (x: number) => { const i = Math.max(0, Math.min(KS.length - 2, Math.floor(x - 1))); return v[i] + (v[i + 1] - v[i]) * (x - 1 - i); };
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="ck-row">
        <div><div className="ck-cap">{TEXT.w[lang]}</div>
          <CurvePlot xs={KS} ys={wcss} xDomain={[0.6, 8.4]} yDomain={[0, Math.max(...wcss) * 1.05]} height={220} curves={[{ f: lin(wcss), color: "var(--accent)" }]} vlines={[{ x: k, color: "var(--ink)" }]} /></div>
        <div><div className="ck-cap">{TEXT.s[lang]}</div>
          <CurvePlot xs={KS.slice(1)} ys={sil.slice(1)} xDomain={[0.6, 8.4]} yDomain={[0, 1]} height={220} curves={[{ f: (x) => lin(sil)(Math.max(2, x)), color: "var(--accent-2)" }]} vlines={[{ x: k, color: "var(--ink)" }]} /></div>
      </div>
      <Slider label={TEXT.k[lang]} value={k} min={1} max={8} step={1} onChange={setK} />
      <div className="w-stats">
        <Stat label={TEXT.w[lang]} value={wcss[k - 1].toFixed(1)} />
        <Stat label={TEXT.s[lang]} value={k > 1 ? sil[k - 1].toFixed(3) : "—"} tone="accent" />
      </div>
    </Widget>
  );
}
