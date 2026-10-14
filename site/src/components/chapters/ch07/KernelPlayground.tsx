import { useMemo, useState } from "react";
import { gridFor, moons } from "../../../engines/data";
import { decisionField, kernels, smo } from "../../../engines/smo";
import type { Lang } from "../../../i18n/ui";
import { renderRich } from "../../../lib/rich";
import Plot2D from "../../viz/Plot2D";
import Widget, { Slider, Stat } from "../../ui/Widget";

type K = "linear" | "poly" | "rbf";

const TEXT = {
  title: { zh: "换一个核，边界就不再是直线", en: "Change the kernel and the boundary bends" },
  linear: { zh: "线性核", en: "Linear" },
  poly: { zh: "多项式核 (d=3)", en: "Polynomial (d=3)" },
  rbf: { zh: "RBF 核", en: "RBF" },
  gamma: { zh: "带宽 γ", en: "Bandwidth γ" },
  C: { zh: "惩罚 C", en: "Penalty C" },
  acc: { zh: "训练准确率", en: "Training accuracy" },
  sv: { zh: "支持向量", en: "Support vectors" },
  caption: {
    zh: "同一个 SMO，只把内积 $x^\\top z$ 换成核函数 $K(x,z)$。背景颜色深浅是决策函数 $f(x)$ 的大小，实线是 $f(x)=0$。RBF 的 γ 越大，每个支持向量的“影响半径”越小，边界越弯曲——也越容易过拟合。",
    en: "The same SMO, with the inner product $x^\\top z$ swapped for a kernel $K(x,z)$. Background shade is the size of the decision function $f(x)$; the solid curve is $f(x)=0$. A larger RBF γ shrinks each support vector's radius of influence, so the boundary bends more — and overfits more easily.",
  },
};

export default function KernelPlayground({ lang }: { lang: Lang }) {
  const { X, y } = useMemo(() => moons(80, 3, 0.2), []);
  const grid = useMemo(() => gridFor(X, 40), [X]);
  const [kind, setKind] = useState<K>("rbf");
  const [logGamma, setLogGamma] = useState(0.3);
  const [logC, setLogC] = useState(1);
  const gamma = 10 ** logGamma;
  const C = 10 ** logC;

  const { field, support, acc } = useMemo(() => {
    const kernel = kind === "linear" ? kernels.linear() : kind === "poly" ? kernels.poly(3) : kernels.rbf(gamma);
    const r = smo(X, y, { C, kernel, maxSteps: 3000 });
    const onData = decisionField(X, y, r, kernel, X);
    return {
      field: decisionField(X, y, r, kernel, grid.points),
      support: r.alpha.flatMap((a, k) => (a > 1e-6 ? [k] : [])),
      acc: onData.filter((v, k) => (v >= 0 ? 1 : -1) === y[k]).length / y.length,
    };
  }, [X, y, grid, kind, gamma, C]);

  return (
    <Widget title={TEXT.title[lang]} caption={<span dangerouslySetInnerHTML={{ __html: renderRich(TEXT.caption[lang]) }} />} tag="extension">
      <div className="chips">
        {(["linear", "poly", "rbf"] as K[]).map((k) => (
          <button key={k} className={`chip${k === kind ? " on" : ""}`} onClick={() => setKind(k)}>{TEXT[k][lang]}</button>
        ))}
      </div>
      <Plot2D X={X} y={y} step={{ field, support }} grid={{ x: grid.x, y: grid.y }} tweenMs={300} />
      <div className="w-row">
        {kind === "rbf" && <Slider label={TEXT.gamma[lang]} value={logGamma} min={-1} max={1.5} step={0.05} onChange={setLogGamma} format={() => gamma.toPrecision(2)} />}
        <Slider label={TEXT.C[lang]} value={logC} min={-1} max={3} step={0.05} onChange={setLogC} format={() => C.toPrecision(2)} />
      </div>
      <div className="w-stats">
        <Stat label={TEXT.acc[lang]} value={`${(acc * 100).toFixed(1)}%`} tone="accent" />
        <Stat label={TEXT.sv[lang]} value={support.length} />
      </div>
    </Widget>
  );
}

