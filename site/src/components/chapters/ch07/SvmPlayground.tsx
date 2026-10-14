import { useMemo, useState } from "react";
import { blobs } from "../../../engines/data";
import { smo } from "../../../engines/smo";
import type { Lang } from "../../../i18n/ui";
import Plot2D from "../../viz/Plot2D";
import Widget, { Slider, Stat } from "../../ui/Widget";

const TEXT = {
  title: { zh: "拖动数据点，看最大间隔如何响应", en: "Drag the points and watch the maximum margin respond" },
  C: { zh: "惩罚系数 C", en: "Penalty C" },
  width: { zh: "间隔宽度 2/‖w‖", en: "Margin width 2/‖w‖" },
  sv: { zh: "支持向量", en: "Support vectors" },
  err: { zh: "训练误分类", en: "Training mistakes" },
  reset: { zh: "重置数据", en: "Reset data" },
  caption: {
    zh: "实线是分离超平面，虚线是两侧的间隔边界，带旋转虚线圈的点是支持向量。试试：拖动一个远离边界的点——超平面纹丝不动；拖动支持向量——它立刻跟着变。把 C 调小，间隔变宽、容忍更多点进入间隔（软间隔）；调大则逼近硬间隔。",
    en: "The solid line is the separating hyperplane, the dashed lines are the margin boundaries, and points with a spinning ring are support vectors. Try it: drag a point far from the boundary — the hyperplane does not move; drag a support vector — it follows at once. Lower C widens the margin and tolerates points inside it (soft margin); raise it to approach the hard margin.",
  },
};

export default function SvmPlayground({ lang }: { lang: Lang }) {
  const initial = useMemo(() => blobs(36, 21, 0.85), []);
  const [X, setX] = useState(initial.X);
  const [logC, setLogC] = useState(0);
  const C = 10 ** logC;

  const r = useMemo(() => smo(X, initial.y, { C, maxSteps: 4000 }), [X, C, initial.y]);
  const norm = Math.hypot(...r.w);
  const mistakes = X.filter((x, i) => initial.y[i] * (r.w[0] * x[0] + r.w[1] * x[1] + r.b) < 0).length;
  const support = r.alpha.flatMap((a, k) => (a > 1e-6 ? [k] : []));

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <Plot2D X={X} y={initial.y} step={{ w: r.w, b: r.b, support }} tweenMs={160}
        domain={{ x: [-4, 4], y: [-3.6, 3.8] }}
        onDragPoint={(i, p) => setX((cur) => cur.map((q, k) => (k === i ? p : q)))} />
      <div className="w-row">
        <Slider label={TEXT.C[lang]} value={logC} min={-2} max={2} step={0.05} onChange={setLogC} format={() => C.toPrecision(2)} />
        <button className="btn" onClick={() => setX(initial.X)}>{TEXT.reset[lang]}</button>
      </div>
      <div className="w-stats">
        <Stat label={TEXT.width[lang]} value={norm > 0 ? (2 / norm).toFixed(2) : "∞"} tone="accent" />
        <Stat label={TEXT.sv[lang]} value={support.length} />
        <Stat label={TEXT.err[lang]} value={mistakes} tone={mistakes ? "bad" : "good"} />
      </div>
    </Widget>
  );
}
