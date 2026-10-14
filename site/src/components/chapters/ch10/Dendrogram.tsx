import { useMemo, useState } from "react";
import { agglomerate, cut, subgroups, type Linkage, type Merge } from "../../../engines/clustering";
import type { Lang } from "../../../i18n/ui";
import Widget, { Slider, Stat } from "../../ui/Widget";
import "./ch10.css";

const TEXT = {
  title: { zh: "层次聚类：一棵树，任意切", en: "Hierarchical clustering: one tree, cut anywhere" },
  single: { zh: "单链接", en: "Single" },
  complete: { zh: "全链接", en: "Complete" },
  average: { zh: "平均链接", en: "Average" },
  h: { zh: "切割高度", en: "Cut height" },
  n: { zh: "簇数", en: "Clusters" },
  caption: {
    zh: "18 位病人。凝聚式层次聚类从每人一簇开始，每次合并距离最近的两簇；树状图（右）记录每次合并发生的高度。拖动切割线：线以下的合并保留，簇的个数随之变化（左图颜色）。不必事先选 k，一棵树给出所有粒度。簇间距离怎样定义决定了树的形状：单链接看最近的一对点，容易一条链地串起来；全链接看最远的一对，簇更紧凑；平均链接介于两者之间。数据为合成示意。",
    en: "18 patients. Agglomerative clustering starts with one cluster per patient and repeatedly merges the two closest clusters; the dendrogram (right) records the height of every merge. Drag the cut line: merges below it are kept, and the number of clusters changes with it (colours on the left). No need to choose k in advance: one tree gives every granularity. How the distance between clusters is defined shapes the tree: single linkage looks at the closest pair and tends to chain; complete linkage looks at the farthest pair and gives compact clusters; average linkage sits in between. Synthetic illustrative data.",
  },
};

const LINKS: Linkage[] = ["single", "complete", "average"];
const f1 = (v: number) => v.toFixed(1);

/** Leaf order and x / height of each node for drawing. */
function layout(n: number, merges: Merge[]) {
  const order: number[] = [];
  const kids = (id: number): number[] => (id < n ? [id] : [merges[id - n].a, merges[id - n].b]);
  const walk = (id: number) => (id < n ? order.push(id) : kids(id).forEach(walk));
  walk(n + merges.length - 1);
  const x = new Map<number, number>();
  order.forEach((leaf, i) => x.set(leaf, i));
  merges.forEach((m, k) => x.set(n + k, (x.get(m.a)! + x.get(m.b)!) / 2));
  const h = (id: number) => (id < n ? 0 : merges[id - n].h);
  return { order, x, h };
}

export default function Dendrogram({ lang }: { lang: Lang }) {
  const { X } = useMemo(() => subgroups(18, 9), []);
  const [li, setLi] = useState(1);
  const merges = useMemo(() => agglomerate(X, LINKS[li]), [X, li]);
  const top = merges.at(-1)!.h;
  const [frac, setFrac] = useState(0.45);
  const hcut = frac * top;
  const lab = cut(X.length, merges, hcut);
  const nC = Math.max(...lab) + 1;
  const { x, h } = layout(X.length, merges);
  const W = 230, H = 230;
  const dx = (v: number) => 12 + (v * (W - 24)) / (X.length - 1);
  const dy = (v: number) => H - 14 - (v / top) * (H - 30);
  const px = (v: number) => 115 + v * 38;
  const py = (v: number) => 115 - v * 34;
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="chips">
        {LINKS.map((l, i) => <button key={l} className={`chip${li === i ? " on" : ""}`} onClick={() => setLi(i)}>{TEXT[l][lang]}</button>)}
      </div>
      <div className="dg-row">
        <svg className="lm-svg" viewBox="0 0 230 230" role="img">
          {X.map((p, i) => <circle key={i} cx={f1(px(p[0]))} cy={f1(py(p[1]))} r={5} className={`km-pt c${lab[i] % 6}`} />)}
        </svg>
        <svg className="lm-svg" viewBox={`0 0 ${W} ${H}`} role="img">
          {merges.map((m, k) => {
            const id = X.length + k;
            const [xa, xb, ha, hb, hm] = [x.get(m.a)!, x.get(m.b)!, h(m.a), h(m.b), m.h];
            return <path key={id} d={`M${f1(dx(xa))},${f1(dy(ha))}V${f1(dy(hm))}H${f1(dx(xb))}V${f1(dy(hb))}`} className={`dg-link${hm <= hcut ? " in" : ""}`} />;
          })}
          {X.map((_, i) => <circle key={i} cx={f1(dx(x.get(i)!))} cy={f1(dy(0))} r={3.5} className={`km-pt c${lab[i] % 6}`} />)}
          <line x1={0} x2={W} y1={f1(dy(hcut))} y2={f1(dy(hcut))} className="dg-cut" />
        </svg>
      </div>
      <Slider label={TEXT.h[lang]} value={frac} min={0} max={1} step={0.01} onChange={setFrac} format={(v) => (v * top).toFixed(2)} />
      <div className="w-stats"><Stat label={TEXT.n[lang]} value={nC} tone="accent" /></div>
    </Widget>
  );
}
